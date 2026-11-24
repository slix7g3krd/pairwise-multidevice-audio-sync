"""多设备音频同步核心模块。

本模块把同步链路拆成六个可独立验证的环节：

1. 时钟漂移估计      :func:`clock_drift` / :func:`drift_correct`
2. 延迟补偿表        :func:`latency_table` / :func:`bt_latency`
3. 主切换时间基连续  :func:`master_switch`
4. 统一欠载策略      :func:`underrun_policy`
5. 相位对齐          :func:`phase_align`
6. 同步误差监测      :func:`sync_monitor`

辅助入口：:func:`buffer_depth`、:func:`validation`、:func:`audit`、
:func:`fallback`、:func:`device_list`、:func:`metrics`、:func:`determinism`。

约定
----
* 时间单位统一为毫秒（ms），采样率默认 48000 Hz。
* 漂移模型统一写成 ``master_ms = slope * device_ms + intercept``，
  因此 ``rate_ppm = (slope - 1) * 1e6``。:func:`clock_drift` 的
  ``offset_ms`` 是该式的加性截距（device→master 方向的修正量），
  同时另给出 ``device_minus_master_ms = -offset_ms`` 便于对照。
* :func:`master_switch` 中的每设备 ``offset_ms`` 定义为「设备本地时间
  减去会话时间」，即 ``session_ms = device_ms - offset_ms``。
* 所有函数均为纯函数：非法或缺失输入返回结构化的中性结果（并给出
  ``warnings``），不抛异常；输出可直接 JSON 序列化。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

__all__ = [
    "clock_drift",
    "latency_table",
    "master_switch",
    "underrun_policy",
    "phase_align",
    "sync_monitor",
    "buffer_depth",
    "validation",
    "audit",
    "fallback",
    "device_list",
    "metrics",
    "drift_correct",
    "bt_latency",
    "determinism",
    "classify_underrun",
]

DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_TOLERANCE_MS = 20.0
DEFAULT_PHASE_PERIOD_MS = 1_000.0

#: 估计漂移趋势所需的最小观测跨度；更短的时间窗无法把趋势与抖动分开。
MIN_TREND_SPAN_MS = 30_000.0

#: 传输链路固有延迟（不含编解码器）。
TRANSPORT_BASE_MS = {
    "wired": 5.0,
    "usb": 8.0,
    "bluetooth": 0.0,
}

#: 蓝牙编解码器典型单向延迟，按 A2DP 常见档位取值。
CODEC_LATENCY_MS = {
    "pcm": 0.0,
    "sbc": 220.0,
    "aac": 180.0,
    "aptx": 70.0,
    "aptx_ll": 40.0,
    "aptx_hd": 80.0,
    "ldac": 200.0,
    "lc3": 50.0,
}

#: 编解码器延迟的不确定度（用于误差带与校准排序）。
CODEC_UNCERTAINTY_MS = {
    "pcm": 1.0,
    "sbc": 40.0,
    "aac": 30.0,
    "aptx": 15.0,
    "aptx_ll": 8.0,
    "aptx_hd": 15.0,
    "ldac": 35.0,
    "lc3": 12.0,
}

_TRANSPORT_ALIASES = {
    "bt": "bluetooth",
    "bluetooth": "bluetooth",
    "ble": "bluetooth",
    "usb": "usb",
    "wire": "wired",
    "wired": "wired",
    "analog": "wired",
    "aux": "wired",
    "3.5mm": "wired",
}

_CODEC_ALIASES = {
    "aptxll": "aptx_ll",
    "aptx_low_latency": "aptx_ll",
    "aptx_hd": "aptx_hd",
    "a2dp_sbc": "sbc",
    "lc3plus": "lc3",
    "lc3_plus": "lc3",
}


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _number(value: Any) -> float | None:
    """把输入转成有限浮点数；失败返回 ``None``。"""

    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _percentile(values: Sequence[float], fraction: float) -> float:
    """线性插值分位数；``fraction`` 取 0..1。"""

    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = min(max(fraction, 0.0), 1.0) * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """最小二乘斜率；样本不足或 x 无跨度时返回 0。"""

    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = _mean(xs)
    my = _mean(ys)
    sxx = math.fsum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0
    sxy = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def _stable_unit(*parts: Any) -> float:
    """由输入派生 [0, 1) 的稳定伪随机数。

    使用 SHA-256 而不是内置 ``hash``，保证跨进程、跨运行一致。
    """

    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _series(value: Any, *keys: str) -> list[Any]:
    """从标量/序列/映射中取出列表，映射按键优先级读取。"""

    if isinstance(value, Mapping):
        for key in keys or ("samples", "values", "items", "data", "records"):
            if key in value and not isinstance(value[key], (str, bytes)):
                candidate = value[key]
                if isinstance(candidate, Iterable):
                    return list(candidate)
        return []
    if isinstance(value, (str, bytes)) or value is None:
        return []
    if isinstance(value, Iterable):
        return list(value)
    return []


def _numbers(value: Any, *keys: str) -> list[float]:
    """把输入清洗成有限浮点数列表。"""

    result: list[float] = []
    for item in _series(value, *keys):
        number = _number(item)
        if number is not None:
            result.append(number)
    if not result:
        single = _number(value)
        if single is not None:
            result.append(single)
    return result


def _normalize_transport(raw: Any) -> str | None:
    if raw is None:
        return None
    key = str(raw).strip().lower()
    return _TRANSPORT_ALIASES.get(key, key or None)


def _infer_transport(name: str) -> str | None:
    lowered = name.strip().lower()
    if lowered.startswith("bt") or "bluetooth" in lowered or "airpod" in lowered:
        return "bluetooth"
    if "usb" in lowered:
        return "usb"
    if "wire" in lowered or "aux" in lowered or "jack" in lowered or "3.5" in lowered:
        return "wired"
    return None


def _normalize_codec(raw: Any) -> str | None:
    if raw is None:
        return None
    key = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    return _CODEC_ALIASES.get(key, key) or None


def _clamp(value: float, low: float, high: float) -> float:
    if low > high:
        low, high = high, low
    return min(max(value, low), high)


# ---------------------------------------------------------------------------
# 1. 时钟漂移估计
# ---------------------------------------------------------------------------


def _drift_points(value: Any) -> list[tuple[float, float]]:
    """解析 (device_ms, master_ms) 采样点。"""

    raw = _series(value, "samples", "pairs", "points", "values", "data")
    if not raw and isinstance(value, Mapping):
        # 允许单点写法：{"device_ms": ..., "master_ms": ...}
        device = _number(value.get("device_ms", value.get("device", value.get("t_device"))))
        master = _number(value.get("master_ms", value.get("master", value.get("t_master"))))
        if device is not None and master is not None:
            raw = [(device, master)]

    points: list[tuple[float, float]] = []
    for item in raw:
        if isinstance(item, Mapping):
            device = _number(
                item.get("device_ms", item.get("device", item.get("t_device", item.get("local_ms"))))
            )
            master = _number(
                item.get("master_ms", item.get("master", item.get("t_master", item.get("ref_ms"))))
            )
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            if len(item) < 2:
                continue
            device = _number(item[0])
            master = _number(item[1])
        else:
            continue
        if device is not None and master is not None:
            points.append((device, master))

    points.sort(key=lambda pair: (pair[0], pair[1]))
    return points


def _theil_sen(points: Sequence[tuple[float, float]], max_effective: int = 200) -> tuple[float, float]:
    """Theil–Sen 稳健斜率 + 中位截距，O(n²) 但限制有效点数。"""

    n = len(points)
    if n < 2:
        return 1.0, 0.0
    step = max(1, math.ceil(n / max_effective))
    sampled = list(points)[::step]
    slopes: list[float] = []
    for i in range(len(sampled)):
        x_i, y_i = sampled[i]
        for j in range(i + 1, len(sampled)):
            x_j, y_j = sampled[j]
            dx = x_j - x_i
            if dx != 0.0:
                slopes.append((y_j - y_i) / dx)
    slope = _median(slopes) if slopes else 1.0
    intercept = _median([y - slope * x for x, y in sampled])
    return slope, intercept


def clock_drift(samples: Any = (), *, min_points: int = 3, robust: bool = True) -> dict[str, Any]:
    """估计设备时钟相对主时钟的漂移。

    输入为 ``(device_ms, master_ms)`` 采样对（也接受带 ``device_ms`` /
    ``master_ms`` 键的映射，或 ``{"samples": [...]}``）。输出线性模型
    ``master_ms = slope * device_ms + intercept`` 及质量指标。

    ``rate_ppm > 0`` 表示设备时钟走得比主时钟慢（同样的设备读数对应
    更大的主时间）；``rate_ppm < 0`` 则表示设备时钟偏快。
    """

    points = _drift_points(samples)
    xs = [pair[0] for pair in points]
    ys = [pair[1] for pair in points]
    n = len(points)
    span_ms = xs[-1] - xs[0] if n >= 2 else 0.0

    warnings: list[str] = []
    if n == 0:
        warnings.append("no drift samples")
    elif n < 2:
        warnings.append("single sample: rate is not observable")

    if n >= 2:
        method = "least-squares"
        if robust and n >= 4:
            slope, intercept = _theil_sen(points)
            method = "theil-sen"
        else:
            slope = _slope(xs, ys)
            if slope == 0.0 and span_ms == 0.0:
                slope = 1.0
            intercept = _mean(ys) - slope * _mean(xs)
    elif n == 1:
        slope, intercept = 1.0, ys[0] - xs[0]
        method = "degenerate"
    else:
        slope, intercept = 1.0, 0.0
        method = "degenerate"

    residuals = [y - (slope * x + intercept) for x, y in points]
    residual_rms = math.sqrt(_mean([r * r for r in residuals])) if residuals else 0.0
    residual_mad = _median([abs(r) for r in residuals])

    reliable = n >= max(2, int(min_points)) and span_ms > 0.0
    if not reliable and n >= 2:
        warnings.append("insufficient span for a reliable estimate")

    rate_ppm = (slope - 1.0) * 1e6

    return {
        "n": n,
        "slope": slope,
        "intercept": intercept,
        "rate_ppm": rate_ppm,
        "offset_ms": intercept,
        "device_minus_master_ms": -intercept,
        "residual_rms_ms": residual_rms,
        "residual_mad_ms": residual_mad,
        "span_ms": span_ms,
        "drift_per_minute_ms": rate_ppm * 1e-6 * 60_000.0,
        "method": method,
        "robust": method == "theil-sen",
        "reliable": reliable,
        "warnings": warnings,
    }


def drift_correct(model: Any = None, time_ms: Any = None) -> dict[str, Any]:
    """把设备时间换算到主时间基。

    支持三种调用形态：

    * ``drift_correct(model, t)``：``model`` 为 :func:`clock_drift` 的结果
      （或至少包含 ``slope``/``rate_ppm`` 与 ``intercept``/``offset_ms``）。
    * ``drift_correct(model)``：只报告校正律（``t = 0`` 处）。
    * ``drift_correct(1)``：把标量视作设备时间，使用单位模型。
    """

    warnings: list[str] = []
    slope = 1.0
    intercept = 0.0

    if isinstance(model, Mapping):
        raw_slope = _number(model.get("slope"))
        if raw_slope is not None:
            slope = raw_slope
        else:
            raw_ppm = _number(model.get("rate_ppm"))
            if raw_ppm is not None:
                slope = 1.0 + raw_ppm * 1e-6
        raw_intercept = _number(model.get("intercept"))
        if raw_intercept is None:
            raw_intercept = _number(model.get("offset_ms"))
        if raw_intercept is not None:
            intercept = raw_intercept
    elif _number(model) is not None:
        if time_ms is None:
            time_ms = _number(model)
        else:
            warnings.append("scalar model ignored")
    elif model is not None:
        warnings.append("model is not a mapping; identity transform used")

    def apply(value: float) -> dict[str, Any]:
        master = slope * value + intercept
        return {
            "device_ms": value,
            "master_ms": master,
            "correction_ms": master - value,
            "slope": slope,
            "rate_ppm": (slope - 1.0) * 1e6,
            "intercept": intercept,
        }

    if time_ms is None:
        report = apply(0.0)
        report.update(
            {
                "model": {"slope": slope, "rate_ppm": (slope - 1.0) * 1e6, "intercept": intercept},
                "warnings": warnings,
            }
        )
        return report

    if isinstance(time_ms, Mapping):
        time_ms = time_ms.get("time_ms", time_ms.get("device_ms", time_ms.get("t")))
    if isinstance(time_ms, (str, bytes)):
        warnings.append("non-numeric time supplied")
        return {"device_ms": None, "master_ms": None, "correction_ms": None,
                "slope": slope, "rate_ppm": (slope - 1.0) * 1e6, "intercept": intercept,
                "warnings": warnings}

    if isinstance(time_ms, Iterable):
        values = [v for v in (_number(item) for item in time_ms) if v is not None]
        results = [apply(v) for v in values]
        return {
            "times": results,
            "slope": slope,
            "rate_ppm": (slope - 1.0) * 1e6,
            "intercept": intercept,
            "warnings": warnings,
        }

    value = _number(time_ms)
    if value is None:
        warnings.append("non-numeric time supplied")
        return {"device_ms": None, "master_ms": None, "correction_ms": None,
                "slope": slope, "rate_ppm": (slope - 1.0) * 1e6, "intercept": intercept,
                "warnings": warnings}
    result = apply(value)
    result["warnings"] = warnings
    return result


# ---------------------------------------------------------------------------
# 2. 延迟补偿表
# ---------------------------------------------------------------------------


def _normalize_device(raw: Any, index: int) -> dict[str, Any] | None:
    if isinstance(raw, str):
        device_id = raw.strip() or f"dev{index}"
        transport = _infer_transport(device_id) or "wired"
        codec = None
        measured = None
        uncertainty = None
        trim = 0.0
        role = None
    elif isinstance(raw, Mapping):
        device_id = str(raw.get("id") or raw.get("name") or f"dev{index}")
        transport = _normalize_transport(raw.get("transport")) or _infer_transport(device_id) or "wired"
        codec = _normalize_codec(raw.get("codec"))
        measured = _number(
            raw.get("measured_ms", raw.get("latency_ms", raw.get("base_latency_ms")))
        )
        uncertainty = _number(raw.get("uncertainty_ms"))
        trim = _number(raw.get("trim_ms", raw.get("offset_ms", raw.get("calibration_ms")))) or 0.0
        role = raw.get("role")
    else:
        return None

    if transport == "bluetooth" and codec is None:
        codec = "sbc"
    if transport != "bluetooth":
        codec = codec or "pcm"

    if measured is not None and measured >= 0.0:
        base = measured
        source = "measured"
        calibrated = True
    elif transport == "bluetooth":
        base = TRANSPORT_BASE_MS["bluetooth"] + CODEC_LATENCY_MS.get(codec or "sbc", 0.0)
        source = "codec-table"
        calibrated = False
    else:
        base = TRANSPORT_BASE_MS.get(transport, TRANSPORT_BASE_MS["wired"])
        source = "transport-table"
        calibrated = False

    if uncertainty is None:
        if transport == "bluetooth":
            uncertainty = CODEC_UNCERTAINTY_MS.get(codec or "sbc", 50.0)
        else:
            uncertainty = 1.0

    total = max(0.0, base + trim)

    return {
        "id": device_id,
        "transport": transport,
        "codec": codec,
        "base_latency_ms": base,
        "trim_ms": trim,
        "total_latency_ms": total,
        "uncertainty_ms": uncertainty,
        "latency_source": source,
        "calibrated": calibrated,
        "role": str(role) if role is not None else None,
    }


def latency_table(
    devices: Any = (),
    *,
    align: str = "slowest",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> dict[str, Any]:
    """构建按设备索引的延迟补偿表。

    每个设备的有效延迟 = 基础延迟（实测优先，否则按传输/编解码器查表）
    + 校准微调 ``trim_ms``。默认以最慢设备为对齐目标，为其余设备增加
    ``compensation_ms`` 的渲染前延迟，使所有设备在同一会话时间发声。
    """

    warnings: list[str] = []

    if isinstance(devices, Mapping):
        if "devices" in devices:
            raw_devices = _series(devices, "devices")
        else:
            # 也接受 {device_id: spec} 形式
            raw_devices = [
                dict(spec, id=spec.get("id", key)) if isinstance(spec, Mapping) else {"id": key}
                for key, spec in devices.items()
            ]
    elif isinstance(devices, str):
        raw_devices = [devices]
    else:
        raw_devices = _series(devices, "devices")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_devices):
        normalized = _normalize_device(raw, index)
        if normalized is None or normalized["id"] in seen:
            continue
        seen.add(normalized["id"])
        rows.append(normalized)

    rate = _number(sample_rate)
    if rate is None or rate <= 0:
        warnings.append("invalid sample_rate; default used")
        rate = float(DEFAULT_SAMPLE_RATE)
    rate = int(rate)

    if not rows:
        warnings.append("no devices supplied")
        return {
            "align": align,
            "sample_rate": rate,
            "target_latency_ms": 0.0,
            "spread_ms": 0.0,
            "devices": [],
            "ok": False,
            "warnings": warnings,
        }

    totals = [row["total_latency_ms"] for row in rows]
    master_rows = [row for row in rows if str(row["role"] or "").lower() == "master"]
    if align == "master" and master_rows:
        target = master_rows[0]["total_latency_ms"]
    elif align == "fastest":
        target = min(totals)
    else:
        align = "slowest"
        target = max(totals)

    for row in rows:
        compensation = max(0.0, target - row["total_latency_ms"])
        row["compensation_ms"] = compensation
        row["compensation_frames"] = int(round(compensation * rate / 1000.0))

    rows.sort(key=lambda row: (row["id"], row["transport"], row["codec"] or ""))
    spread = max(totals) - min(totals)
    if spread > 200.0:
        warnings.append("latency spread exceeds 200 ms; wired/bluetooth mix needs calibration")

    return {
        "align": align,
        "sample_rate": rate,
        "target_latency_ms": target,
        "spread_ms": spread,
        "devices": rows,
        "ok": True,
        "warnings": warnings,
    }


def bt_latency(
    codec: Any = "sbc",
    *,
    transport: str = "bluetooth",
    measured_ms: Any = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> dict[str, Any]:
    """查询蓝牙（或任意传输）链路延迟，实测值优先于编解码器查表。"""

    warnings: list[str] = []
    measuring = _number(measured_ms)

    if isinstance(codec, Mapping):
        spec = codec
        codec_name = _normalize_codec(spec.get("codec")) or "sbc"
        transport_name = _normalize_transport(spec.get("transport")) or "bluetooth"
        if _number(spec.get("measured_ms")) is not None:
            measuring = _number(spec.get("measured_ms"))
    else:
        number = _number(codec)
        if number is not None:
            # 只传一个数字时视作实测延迟。
            codec_name = "sbc"
            transport_name = _normalize_transport(transport) or "bluetooth"
            if measuring is None and number >= 0.0:
                measuring = number
        else:
            codec_name = _normalize_codec(codec) or "sbc"
            transport_name = _normalize_transport(transport) or "bluetooth"

    nominal = CODEC_LATENCY_MS.get(codec_name)
    if nominal is None:
        warnings.append(f"unknown codec: {codec_name}")
        nominal = 0.0
    nominal += TRANSPORT_BASE_MS.get(transport_name, 0.0)

    uncertainty = CODEC_UNCERTAINTY_MS.get(codec_name, 50.0)
    if transport_name != "bluetooth":
        uncertainty = 1.0

    if measuring is not None and measuring >= 0.0:
        effective = measuring
        source = "measured"
    else:
        effective = nominal
        source = "codec-table"

    rate = _number(sample_rate)
    rate = int(rate) if rate and rate > 0 else DEFAULT_SAMPLE_RATE

    return {
        "codec": codec_name,
        "transport": transport_name,
        "nominal_ms": nominal,
        "measured_ms": measuring,
        "effective_ms": effective,
        "uncertainty_ms": uncertainty,
        "source": source,
        "compensation_frames": int(round(effective * rate / 1000.0)),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# 3. 主切换时间基连续
# ---------------------------------------------------------------------------


def master_switch(state: Any = None, new_master: Any = None) -> dict[str, Any]:
    """切换主设备并保持时间基连续。

    ``state`` 形如::

        {
          "master": "phone",
          "time_ms": 120000,          # 切换瞬间旧主设备本地时间
          "clocks": {
             "phone": {"offset_ms": 12.0, "rate_ppm": -3.0},
             "tv":    {"offset_ms": -8.0, "rate_ppm": 1.5}
          }
        }

    其中 ``offset_ms = 设备本地时间 - 会话时间``。主切换只做一次时间基
    平移 ``delta = offset_new - offset_old``，播放位置（会话时间）与
    声道相位都不重置，因此不会出现跳变或二次起播。
    """

    warnings: list[str] = []
    old_master = "master"
    switch_time = 0.0
    clocks: dict[str, dict[str, float]] = {}

    if isinstance(state, Mapping):
        old_master = str(state.get("master") or state.get("old_master") or "master")
        switch_time = _number(
            state.get("time_ms", state.get("switch_time_ms", state.get("position_ms")))
        ) or 0.0
        raw_clocks = state.get("clocks", state.get("devices", {}))
        if isinstance(raw_clocks, Mapping):
            for device_id, spec in raw_clocks.items():
                if isinstance(spec, Mapping):
                    offset = _number(spec.get("offset_ms", spec.get("offset"))) or 0.0
                    rate = _number(spec.get("rate_ppm", spec.get("drift_ppm"))) or 0.0
                else:
                    offset = _number(spec) or 0.0
                    rate = 0.0
                clocks[str(device_id)] = {"offset_ms": offset, "rate_ppm": rate}
        elif isinstance(raw_clocks, Iterable) and not isinstance(raw_clocks, (str, bytes)):
            for index, spec in enumerate(raw_clocks):
                normalized = _normalize_device(spec, index)
                if normalized is None:
                    continue
                clocks[normalized["id"]] = {"offset_ms": 0.0, "rate_ppm": 0.0}
    elif _number(state) is not None:
        switch_time = _number(state) or 0.0
    elif state is not None:
        warnings.append("state is not a mapping; no clocks available")

    requested = new_master
    if requested is None and isinstance(state, Mapping):
        requested = state.get("new_master")
    if requested is None:
        candidates = sorted(device for device in clocks if device != old_master)
        requested = candidates[0] if candidates else old_master
    new_master = str(requested)

    if old_master not in clocks:
        clocks.setdefault(old_master, {"offset_ms": 0.0, "rate_ppm": 0.0})
    if new_master not in clocks:
        clocks.setdefault(new_master, {"offset_ms": 0.0, "rate_ppm": 0.0})
        if new_master != old_master:
            warnings.append(f"unknown new master {new_master!r}; offset assumed 0")

    offset_old = clocks[old_master]["offset_ms"]
    offset_new = clocks[new_master]["offset_ms"]
    delta = offset_new - offset_old

    # 会话时间在整个切换过程中保持不变，这是“时间基连续”的定义。
    session_ms = switch_time - offset_old
    new_local_time = session_ms + offset_new

    return {
        "old_master": old_master,
        "new_master": new_master,
        "switch_time_ms": switch_time,
        "old_timebase_ms": switch_time,
        "new_timebase_ms": new_local_time,
        "delta_ms": delta,
        "session_ms": session_ms,
        "continuous": True,
        "playback_position_preserved": True,
        "phase_preserved": True,
        "monotonic": True,
        "next_rate_ppm": clocks[new_master]["rate_ppm"],
        "clocks": clocks,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# 4. 统一欠载策略
# ---------------------------------------------------------------------------


def _underrun_class(gap_ms: float, fade_ms: float, max_gap_ms: float) -> str:
    if gap_ms <= fade_ms:
        return "micro"
    if gap_ms <= max_gap_ms:
        return "short"
    return "long"


def classify_underrun(gap_ms: Any, policy: Any = None) -> dict[str, Any]:
    """按统一策略给一次欠载分类，返回应采取的动作。"""

    gap = _number(gap_ms) or 0.0
    fade = 5.0
    max_gap = 250.0
    if isinstance(policy, Mapping):
        fade = _number(policy.get("fade_ms")) or fade
        max_gap = _number(policy.get("max_gap_ms")) or max_gap

    kind = _underrun_class(abs(gap), fade, max_gap)
    action = {
        "micro": "repeat-last-sample-faded",
        "short": "silence-fill-with-fade",
        "long": "reanchor-and-prefill",
    }[kind]
    return {"gap_ms": gap, "class": kind, "action": action}


def underrun_policy(
    config: Any = (),
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    fade_ms: float = 5.0,
    max_gap_ms: float = 250.0,
    repeat_limit: int = 3,
) -> dict[str, Any]:
    """返回所有设备共用的欠载填充策略。

    目标：任何一次缓冲欠载都不产生阶跃（爆音）。统一做法是

    1. 播放侧先按 ``fade_ms`` 余弦淡出；
    2. 极少量的缺失（``micro``）用淡出的最后采样补齐，避免可闻的空洞；
    3. 其余欠载填静音，恢复后用同样时长余弦淡入；
    4. 全程不重置相位/计数器，长欠载才重新锚定时间基。

    这样蓝牙、有线、以及不同采样率的设备执行同一套动作，不会因为
    各自默认策略不同而在同一时刻发出不同波形。
    """

    warnings: list[str] = []
    if isinstance(config, Mapping):
        sample_rate = _number(config.get("sample_rate", sample_rate)) or sample_rate
        fade_ms = _number(config.get("fade_ms", fade_ms)) or fade_ms
        max_gap_ms = _number(config.get("max_gap_ms", max_gap_ms)) or max_gap_ms
        repeat_limit = int(_number(config.get("repeat_limit", repeat_limit)) or repeat_limit)
    elif _number(config) is not None:
        fade_ms = max(0.0, _number(config) or 0.0)

    rate = _number(sample_rate)
    if rate is None or rate <= 0:
        warnings.append("invalid sample_rate; default used")
        rate = DEFAULT_SAMPLE_RATE
    rate = int(rate)

    fade_ms = max(0.0, fade_ms)
    max_gap_ms = max(fade_ms, max_gap_ms)
    repeat_limit = max(1, repeat_limit)

    fade_frames = int(round(fade_ms * rate / 1000.0))

    return {
        "sample_rate": rate,
        "fade_ms": fade_ms,
        "fade_frames": fade_frames,
        "max_gap_ms": max_gap_ms,
        "repeat_limit": repeat_limit,
        "strategy": "phase-continuous-cosine-fade",
        "steps": [
            "fade-out-last-valid-block",
            "fill-silence-for-gap",
            "fade-in-on-resume",
            "do-not-reset-phase-or-counters",
            "resume-against-session-timebase",
        ],
        "classes": {
            "micro": {
                "max_gap_ms": fade_ms,
                "action": "repeat-last-sample-faded",
                "note": "小于淡出时长，用淡出的最后采样补齐",
            },
            "short": {
                "max_gap_ms": max_gap_ms,
                "action": "silence-fill-with-fade",
                "note": "静音填充，恢复时淡入，相位不重置",
            },
            "long": {
                "max_gap_ms": None,
                "action": "reanchor-and-prefill",
                "note": "重新对齐会话时间基并预充缓冲，仍保持淡出/淡入",
            },
        },
        "escalation": {
            "consecutive_limit": repeat_limit,
            "action": "request-recalibration",
            "note": "连续欠载超过阈值时上报告警并重建延迟表",
        },
        "guarantees": ["no-dc-step", "phase-preserved", "counter-not-reset"],
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# 5. 相位对齐
# ---------------------------------------------------------------------------


def phase_align(
    runs: Any = (),
    *,
    period_ms: float = DEFAULT_PHASE_PERIOD_MS,
    tolerance_ms: float | None = None,
) -> dict[str, Any]:
    """把同一房间内多次播放的相位对齐到同一参考。

    输入为各次播放的相位（0..period）或 ``{"id", "phase_ms"}`` 列表。
    用圆均值求参考相位，再给每次播放一个最小有符号校正量
    ``correction_ms``（正数表示该次播放需延后）。
    """

    warnings: list[str] = []
    period = _number(period_ms)
    if period is None or period <= 0:
        warnings.append("invalid period_ms; default used")
        period = DEFAULT_PHASE_PERIOD_MS

    raw_runs: list[tuple[str, float]] = []
    if isinstance(runs, Mapping):
        if any(key in runs for key in ("runs", "phases", "values", "samples")):
            items = _series(runs, "runs", "phases", "values", "samples")
        else:
            items = [
                {"id": str(key), "phase_ms": value} if not isinstance(value, Mapping)
                else dict(value, id=value.get("id", key))
                for key, value in runs.items()
            ]
    elif isinstance(runs, (str, bytes)):
        items = []
        warnings.append("non-numeric phase input ignored")
    elif isinstance(runs, Iterable):
        items = list(runs)
    else:
        items = []

    number = _number(runs)
    if number is not None:
        items = [number]

    for index, item in enumerate(items):
        if isinstance(item, Mapping):
            run_id = str(item.get("id") or item.get("run") or f"run{index}")
            phase = _number(item.get("phase_ms", item.get("phase")))
        else:
            run_id = f"run{index}"
            phase = _number(item)
        if phase is None:
            continue
        raw_runs.append((run_id, phase % period))

    if not raw_runs:
        warnings.append("no phase samples supplied")
        return {
            "period_ms": period,
            "mean_phase_ms": 0.0,
            "confidence": 0.0,
            "tolerance_ms": tolerance_ms if tolerance_ms is not None else 2.0,
            "max_deviation_ms": 0.0,
            "aligned": False,
            "ambiguous": True,
            "runs": [],
            "warnings": warnings,
        }

    angles = [2.0 * math.pi * phase / period for _, phase in raw_runs]
    sum_cos = math.fsum(math.cos(angle) for angle in angles)
    sum_sin = math.fsum(math.sin(angle) for angle in angles)
    resultant = math.hypot(sum_cos, sum_sin) / len(raw_runs)
    mean_angle = math.atan2(sum_sin, sum_cos)
    mean_phase = (mean_angle % (2.0 * math.pi)) * period / (2.0 * math.pi)

    rows: list[dict[str, Any]] = []
    deviations: list[float] = []
    for run_id, phase in raw_runs:
        signed = ((mean_phase - phase + period / 2.0) % period) - period / 2.0
        deviation = abs(signed)
        deviations.append(deviation)
        rows.append(
            {
                "id": run_id,
                "phase_ms": phase,
                "deviation_ms": deviation,
                "correction_ms": signed,
            }
        )

    tolerance = _number(tolerance_ms)
    if tolerance is None:
        tolerance = max(0.5, period / 1000.0)  # 默认千分之一周期

    max_deviation = max(deviations) if deviations else 0.0
    ambiguous = resultant < 0.25 and len(raw_runs) > 1
    if ambiguous:
        warnings.append("phase distribution is nearly uniform; reference is ambiguous")

    return {
        "period_ms": period,
        "mean_phase_ms": mean_phase,
        "confidence": resultant,
        "tolerance_ms": tolerance,
        "max_deviation_ms": max_deviation,
        "aligned": max_deviation <= tolerance,
        "ambiguous": ambiguous,
        "runs": rows,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# 6. 同步误差监测
# ---------------------------------------------------------------------------


def _monitor_stats(
    values: Sequence[float],
    *,
    tolerance_ms: float,
    interval_ms: float,
    trend_alarm_ms_per_min: float,
) -> dict[str, Any]:
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "bias_ms": 0.0,
            "mean_abs_ms": 0.0,
            "std_ms": 0.0,
            "p95_abs_ms": 0.0,
            "max_abs_ms": 0.0,
            "within_tolerance": 0.0,
            "trend_ms_per_min": 0.0,
            "trend_reliable": False,
            "alarm": False,
            "severity": "unknown",
        }

    absolute = [abs(value) for value in values]
    bias = _mean(values)
    mean_abs = _mean(absolute)
    variance = _mean([(value - bias) ** 2 for value in values])
    p95 = _percentile(absolute, 0.95)
    max_abs = max(absolute)
    within = sum(1 for value in absolute if value <= tolerance_ms) / n

    xs = [index * interval_ms for index in range(n)]
    # 把几秒钟的抖动外推成「每分钟」趋势毫无意义，还会在纯噪声上误报。
    # 只有在样本数与观测跨度都足够时才计算趋势，否则明确给出 0。
    span_ms = xs[-1] - xs[0] if n >= 2 else 0.0
    trend_reliable = n >= 5 and span_ms >= MIN_TREND_SPAN_MS
    trend = _slope(xs, values) * 60_000.0 if trend_reliable else 0.0

    critical = max_abs > 2.0 * tolerance_ms or p95 > 1.5 * tolerance_ms
    warning = p95 > tolerance_ms or (trend_reliable and abs(trend) > trend_alarm_ms_per_min)
    severity = "critical" if critical else ("warning" if warning else "ok")

    return {
        "n": n,
        "bias_ms": bias,
        "mean_abs_ms": mean_abs,
        "std_ms": math.sqrt(max(0.0, variance)),
        "p95_abs_ms": p95,
        "max_abs_ms": max_abs,
        "within_tolerance": within,
        "trend_ms_per_min": trend,
        "trend_reliable": trend_reliable,
        "alarm": critical or warning,
        "severity": severity,
    }


def sync_monitor(
    errors: Any = (),
    *,
    tolerance_ms: float = DEFAULT_TOLERANCE_MS,
    interval_ms: float = 1_000.0,
    trend_alarm_ms_per_min: float = 1.0,
) -> dict[str, Any]:
    """监测同步误差：偏差、抖动、越限比例与漂移趋势。

    输入可以是误差序列、``{device: [误差...]}``，或包含 ``errors`` 键的
    映射。``errors`` 符号约定为「实际发声时刻 - 期望时刻」，正值表示
    该设备落后。
    """

    warnings: list[str] = []
    tolerance = _number(tolerance_ms)
    if tolerance is None or tolerance <= 0:
        warnings.append("invalid tolerance_ms; default used")
        tolerance = DEFAULT_TOLERANCE_MS
    interval = _number(interval_ms)
    if interval is None or interval <= 0:
        warnings.append("invalid interval_ms; default used")
        interval = 1_000.0
    trend_limit = _number(trend_alarm_ms_per_min) or 0.0

    per_device: dict[str, dict[str, Any]] = {}

    if isinstance(errors, Mapping) and not any(
        key in errors for key in ("errors", "values", "samples", "offsets", "data")
    ):
        pooled: list[float] = []
        for device_id in sorted(errors, key=str):
            values = _numbers(errors[device_id])
            if not values:
                continue
            per_device[str(device_id)] = _monitor_stats(
                values,
                tolerance_ms=tolerance,
                interval_ms=interval,
                trend_alarm_ms_per_min=trend_limit,
            )
            pooled.extend(values)
        if not per_device:
            warnings.append("no numeric error samples supplied")
    else:
        pooled = _numbers(errors, "errors", "values", "samples", "offsets", "data")
        if not pooled:
            warnings.append("no numeric error samples supplied")

    stats = _monitor_stats(
        pooled,
        tolerance_ms=tolerance,
        interval_ms=interval,
        trend_alarm_ms_per_min=trend_limit,
    )
    stats["tolerance_ms"] = tolerance
    stats["interval_ms"] = interval
    stats["per_device"] = per_device
    stats["warnings"] = warnings
    return stats


# ---------------------------------------------------------------------------
# 辅助环节
# ---------------------------------------------------------------------------


def buffer_depth(
    jitter_ms: Any = 0.0,
    drift_ppm: Any = 0.0,
    *,
    horizon_s: float = 10.0,
    safety: float = 2.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    min_ms: float = 20.0,
    max_ms: float = 2_000.0,
) -> dict[str, Any]:
    """计算目标缓冲深度。

    深度同时覆盖网络/调度抖动与时钟漂移在 ``horizon_s`` 内的累积量：

    ``depth = clamp(safety * jitter + |ppm| * horizon_s * 1000 / 1e6)``
    """

    warnings: list[str] = []
    if isinstance(jitter_ms, Mapping):
        spec = jitter_ms
        jitter_ms = spec.get("jitter_ms", spec.get("jitter", 0.0))
        drift_ppm = spec.get("drift_ppm", spec.get("rate_ppm", drift_ppm))
        horizon_s = _number(spec.get("horizon_s", horizon_s)) or horizon_s
        safety = _number(spec.get("safety", safety)) or safety
        sample_rate = _number(spec.get("sample_rate", sample_rate)) or sample_rate
        min_ms = _number(spec.get("min_ms", min_ms)) or min_ms
        max_ms = _number(spec.get("max_ms", max_ms)) or max_ms

    jitter = abs(_number(jitter_ms) or 0.0)
    drift = abs(_number(drift_ppm) or 0.0)
    horizon = _number(horizon_s)
    if horizon is None or horizon <= 0:
        warnings.append("invalid horizon_s; default used")
        horizon = 10.0
    factor = _number(safety)
    if factor is None or factor <= 0:
        warnings.append("invalid safety; default used")
        factor = 2.0
    rate = _number(sample_rate)
    if rate is None or rate <= 0:
        warnings.append("invalid sample_rate; default used")
        rate = DEFAULT_SAMPLE_RATE
    rate = int(rate)

    low = max(0.0, _number(min_ms) or 0.0)
    high = _number(max_ms)
    if high is None or high <= 0:
        high = 2_000.0
    if high < low:
        low, high = high, low
        warnings.append("min_ms/max_ms swapped")

    drift_growth = drift * 1e-6 * horizon * 1_000.0
    computed = factor * jitter + drift_growth
    target = _clamp(computed, low, high)

    return {
        "target_ms": target,
        "target_frames": int(math.ceil(target * rate / 1000.0)),
        "computed_ms": computed,
        "jitter_ms": jitter,
        "drift_ppm": drift,
        "drift_growth_ms": drift_growth,
        "headroom_ms": max(0.0, target - jitter),
        "horizon_s": horizon,
        "safety": factor,
        "sample_rate": rate,
        "min_ms": low,
        "max_ms": high,
        "clamped": not math.isclose(computed, target, rel_tol=0.0, abs_tol=1e-9),
        "warnings": warnings,
    }


def validation(config: Any = None, *, sample_rate: int = DEFAULT_SAMPLE_RATE) -> dict[str, Any]:
    """校验同步配置，返回 ``ok`` / ``errors`` / ``warnings``。"""

    errors: list[str] = []
    warnings: list[str] = []
    checked = 0

    base_rate = _number(sample_rate)
    base_rate = int(base_rate) if base_rate and base_rate > 0 else DEFAULT_SAMPLE_RATE

    if config is None:
        return {"ok": True, "errors": errors, "warnings": ["empty configuration"],
                "checked": 0, "sample_rate": base_rate}

    if not isinstance(config, Mapping):
        warnings.append("configuration is not a mapping; nothing to validate")
        return {"ok": True, "errors": errors, "warnings": warnings,
                "checked": 0, "sample_rate": base_rate}

    rate = _number(config.get("sample_rate", base_rate))
    if rate is None or rate <= 0 or rate > 384_000:
        errors.append("sample_rate must be in (0, 384000]")
        rate = base_rate
    checked += 1

    tolerance = config.get("tolerance_ms", DEFAULT_TOLERANCE_MS)
    if _number(tolerance) is None or (_number(tolerance) or 0) <= 0:
        errors.append("tolerance_ms must be positive")
    checked += 1

    period = config.get("phase_period_ms", DEFAULT_PHASE_PERIOD_MS)
    if _number(period) is None or (_number(period) or 0) <= 0:
        errors.append("phase_period_ms must be positive")
    checked += 1

    raw_devices = config.get("devices", [])
    if isinstance(raw_devices, (str, bytes)):
        errors.append("devices must be a list or mapping")
        raw_devices = []
    elif isinstance(raw_devices, Mapping):
        # 与 latency_table / device_list 一致，接受 {id: spec} 写法。
        raw_devices = [
            dict(spec, id=spec.get("id", key)) if isinstance(spec, Mapping) else {"id": str(key)}
            for key, spec in raw_devices.items()
        ]
    elif not isinstance(raw_devices, (list, tuple)):
        if raw_devices:
            errors.append("devices must be a list or mapping")
        raw_devices = []
    checked += 1

    seen: set[str] = set()
    masters = 0
    bluetooth_calibrated = 0
    latencies: list[float] = []
    for index, raw in enumerate(raw_devices):
        normalized = _normalize_device(raw, index)
        if normalized is None:
            errors.append(f"device #{index} is not a mapping or string")
            continue
        if normalized["id"] in seen:
            errors.append(f"duplicate device id: {normalized['id']}")
        seen.add(normalized["id"])
        checked += 1

        if normalized["transport"] not in ("wired", "usb", "bluetooth"):
            errors.append(f"device {normalized['id']}: unknown transport "
                          f"{normalized['transport']!r}")
        if normalized["transport"] == "bluetooth" and normalized["codec"] not in CODEC_LATENCY_MS:
            warnings.append(f"device {normalized['id']}: unknown codec "
                            f"{normalized['codec']!r}; latency uncertain")
        if not 0.0 <= normalized["total_latency_ms"] <= 5_000.0:
            errors.append(f"device {normalized['id']}: latency out of range")
        if str(normalized["role"] or "").lower() == "master":
            masters += 1
        if normalized["transport"] == "bluetooth" and normalized["calibrated"]:
            bluetooth_calibrated += 1
        latencies.append(normalized["total_latency_ms"])

    if masters == 0 and raw_devices:
        warnings.append("no device marked as master")
    elif masters > 1:
        errors.append("more than one device marked as master")

    if latencies and max(latencies) - min(latencies) > 200.0:
        warnings.append("latency spread exceeds 200 ms; phase alignment may be audible")
    if any(
        isinstance(raw, Mapping) and _normalize_transport(raw.get("transport")) == "bluetooth"
        for raw in raw_devices
    ) and bluetooth_calibrated == 0:
        warnings.append("bluetooth devices have no measured latency; using codec table")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "checked": checked,
        "sample_rate": int(rate),
    }


def audit(
    records: Any = None,
    *,
    drift_ppm_limit: float = 100.0,
    latency_outlier_k: float = 3.0,
) -> dict[str, Any]:
    """审计设备集合：漂移越限、延迟离群、未校准项。"""

    warnings: list[str] = []
    findings: list[dict[str, Any]] = []

    if isinstance(records, Mapping):
        if "devices" in records:
            items = _series(records, "devices")
        elif "records" in records or "clocks" in records:
            items = _series(records, "records", "clocks")
        else:
            items = [{"id": str(key), **value} if isinstance(value, Mapping) else {"id": str(key), "rate_ppm": value}
                     for key, value in records.items()]
    elif isinstance(records, (str, bytes)):
        items = []
        warnings.append("string record ignored")
    elif isinstance(records, Iterable) and not isinstance(records, Mapping):
        items = list(records)
    else:
        items = []

    rates: list[tuple[str, float]] = []
    latencies: list[tuple[str, float]] = []
    checked = 0

    for index, item in enumerate(items):
        if isinstance(item, Mapping):
            device_id = str(item.get("id") or item.get("name") or f"dev{index}")
            rate = _number(item.get("rate_ppm", item.get("drift_ppm")))
            latency = _number(
                item.get("total_latency_ms",
                         item.get("latency_ms", item.get("measured_ms")))
            )
            calibrated = item.get("calibrated")
            transport = _normalize_transport(item.get("transport")) or _infer_transport(device_id)
            if transport == "bluetooth" and calibrated is False:
                findings.append({
                    "device": device_id,
                    "kind": "uncalibrated-bluetooth",
                    "severity": "warning",
                    "detail": "蓝牙设备未实测延迟，补偿表使用编解码器名义值",
                })
        else:
            device_id = f"dev{index}"
            rate = _number(item)
            latency = None
            calibrated = None

        checked += 1
        if rate is not None:
            rates.append((device_id, rate))
            if abs(rate) > drift_ppm_limit:
                findings.append({
                    "device": device_id,
                    "kind": "drift-out-of-range",
                    "severity": "critical" if abs(rate) > 2.0 * drift_ppm_limit else "warning",
                    "detail": f"rate {rate:.3f} ppm exceeds limit {drift_ppm_limit:.3f} ppm",
                })
        if latency is not None:
            latencies.append((device_id, abs(latency)))

    latency_median = _median([value for _, value in latencies])
    if len(latencies) >= 3:
        mad = _median([abs(value - latency_median) for _, value in latencies])
        sigma = 1.4826 * mad
        if sigma > 0.0:
            for device_id, value in latencies:
                score = abs(value - latency_median) / sigma
                if score > latency_outlier_k:
                    findings.append({
                        "device": device_id,
                        "kind": "latency-outlier",
                        "severity": "warning",
                        "detail": f"latency {value:.1f} ms deviates {score:.1f} sigma "
                                  f"from median {latency_median:.1f} ms",
                    })

    findings.sort(key=lambda finding: (finding["severity"], finding["device"], finding["kind"]))
    if not items:
        warnings.append("no records supplied")

    return {
        "ok": not findings,
        "checked": checked,
        "findings": findings,
        "median_latency_ms": latency_median,
        "median_rate_ppm": _median([rate for _, rate in rates]),
        "warnings": warnings,
    }


def fallback(
    state: Any = None,
    *,
    tolerance_ms: float = DEFAULT_TOLERANCE_MS,
    wired_available: bool | None = None,
) -> dict[str, Any]:
    """同步失配时的确定性降级阶梯。

    依次为：保持 → 重校准 → 切有线 → 仅保留主设备。每一级都带淡出/
    淡入，避免降级本身引入爆音。
    """

    warnings: list[str] = []
    error = 0.0
    dropouts = 0
    wired = wired_available

    if isinstance(state, Mapping):
        error = abs(_number(
            state.get("sync_error_ms",
                      state.get("error_ms", state.get("p95_ms", state.get("max_abs_ms", 0.0))))
        ) or 0.0)
        dropouts = int(_number(state.get("dropouts", state.get("underruns", 0))) or 0)
        if wired is None:
            wired = state.get("wired_available", state.get("wired"))
        tolerance_ms = _number(state.get("tolerance_ms", tolerance_ms)) or tolerance_ms
    elif _number(state) is not None:
        error = abs(_number(state) or 0.0)
    elif state is not None:
        warnings.append("state is not a mapping; using defaults")

    tolerance = max(1e-6, _number(tolerance_ms) or DEFAULT_TOLERANCE_MS)
    wired = bool(wired) if wired is not None else False

    if error <= tolerance and dropouts == 0:
        action, reason = "hold", "sync error within tolerance"
        degraded = False
        steps = ["continue-playback", "keep-monitoring"]
    elif error <= 2.0 * tolerance:
        action, reason = "recalibrate", "sync error above tolerance but recoverable"
        degraded = False
        steps = [
            "fade-out-peers",
            "re-estimate-clock-drift",
            "rebuild-latency-table",
            "phase-aligned-fade-in",
        ]
    elif wired:
        action, reason = "switch-to-wired", "sync error exceeds 2x tolerance; wired path available"
        degraded = True
        steps = [
            "fade-out-bluetooth",
            "prepare-wired-path",
            "prefill-buffer",
            "phase-aligned-crossfade",
            "release-bluetooth",
        ]
    else:
        action, reason = "solo-master", "sync unrecoverable and no wired fallback"
        degraded = True
        steps = [
            "fade-out-peers",
            "promote-master",
            "keep-local-clock",
            "raise-alert",
        ]

    return {
        "action": action,
        "reason": reason,
        "error_ms": error,
        "thresholds": {"recalibrate_ms": tolerance, "switch_ms": 2.0 * tolerance},
        "dropouts": dropouts,
        "wired_available": wired,
        "degraded": degraded,
        "steps": steps,
        "warnings": warnings,
    }


def device_list(devices: Any = None, *, default_transport: str = "wired") -> list[dict[str, Any]]:
    """规范化设备集合：去重、稳定排序、补全角色与链路属性。"""

    if isinstance(devices, Mapping) and "devices" in devices:
        raw_devices = _series(devices, "devices")
    elif isinstance(devices, Mapping):
        raw_devices = [
            dict(spec, id=spec.get("id", key)) if isinstance(spec, Mapping) else {"id": str(key)}
            for key, spec in devices.items()
        ]
    elif isinstance(devices, (str, bytes)):
        raw_devices = [devices]
    else:
        raw_devices = _series(devices, "devices")

    default = _normalize_transport(default_transport) or "wired"
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_devices):
        normalized = _normalize_device(raw, index)
        if normalized is None or normalized["id"] in seen:
            continue
        seen.add(normalized["id"])
        if isinstance(raw, Mapping) and raw.get("transport") is None:
            normalized["transport"] = _infer_transport(normalized["id"]) or default
        if isinstance(raw, Mapping):
            normalized["enabled"] = bool(raw.get("enabled", True))
            normalized["channels"] = int(_number(raw.get("channels", 2)) or 2)
        else:
            normalized["enabled"] = True
            normalized["channels"] = 2
        result.append(normalized)

    result.sort(key=lambda device: (device["id"], device["transport"]))
    for order, device in enumerate(result):
        device["order"] = order
        if not device["role"]:
            device["role"] = "master" if order == 0 else "peer"
    return result


def metrics(sessions: Any = None, *, tolerance_ms: float = DEFAULT_TOLERANCE_MS) -> dict[str, Any]:
    """汇总多次会话的同步质量指标。"""

    warnings: list[str] = []
    raw_sessions: list[Any]

    if isinstance(sessions, Mapping) and any(
        key in sessions for key in ("sessions", "runs", "records", "history")
    ):
        raw_sessions = _series(sessions, "sessions", "runs", "records", "history")
    elif isinstance(sessions, Mapping):
        raw_sessions = [sessions]
    elif isinstance(sessions, (list, tuple)):
        raw_sessions = list(sessions)
    elif _number(sessions) is not None:
        raw_sessions = [{"sync_error_ms": _number(sessions)}]
    else:
        raw_sessions = []

    pooled_errors: list[float] = []
    underruns = 0
    drifts: list[float] = []
    device_ids: set[str] = set()
    session_count = 0

    for raw in raw_sessions:
        if not isinstance(raw, Mapping):
            number = _number(raw)
            if number is not None:
                pooled_errors.append(number)
                session_count += 1
            continue
        session_count += 1
        underruns += int(_number(raw.get("underruns", raw.get("dropouts", 0))) or 0)
        drift = _number(raw.get("drift_ppm", raw.get("rate_ppm")))
        if drift is not None:
            drifts.append(drift)
        pooled_errors.extend(_numbers(raw, "errors", "sync_errors", "offsets"))
        if _number(raw.get("sync_error_ms")) is not None:
            pooled_errors.append(_number(raw.get("sync_error_ms")) or 0.0)
        devices = raw.get("devices")
        if isinstance(devices, Mapping):
            device_ids.update(str(key) for key in devices)
        elif isinstance(devices, Iterable) and not isinstance(devices, (str, bytes)):
            for device in devices:
                if isinstance(device, Mapping):
                    device_ids.add(str(device.get("id") or device.get("name") or "?"))

    if not raw_sessions:
        warnings.append("no sessions supplied")

    monitor = sync_monitor(pooled_errors, tolerance_ms=tolerance_ms)
    return {
        "sessions": session_count,
        "devices": len(device_ids),
        "underruns": underruns,
        "sync": {
            "n": monitor["n"],
            "mean_abs_ms": monitor["mean_abs_ms"],
            "p95_abs_ms": monitor["p95_abs_ms"],
            "max_abs_ms": monitor["max_abs_ms"],
            "within_tolerance": monitor["within_tolerance"],
            "severity": monitor["severity"],
        },
        "drift_ppm": _mean(drifts) if drifts else 0.0,
        "device_ids": sorted(device_ids),
        "warnings": warnings,
    }


def determinism(seed: Any = 0, *, devices: Any = None) -> dict[str, Any]:
    """确定性工具：稳定的设备排序与可复现的伪抖动。

    多设备同步的模拟与回归测试必须可复现，因此排序只依赖规范化的
    设备 ID，抖动由 SHA-256 派生，不使用进程级随机种子或内置 ``hash``。
    """

    warnings: list[str] = []
    if isinstance(seed, Mapping):
        spec = seed
        seed_value = spec.get("seed", 0)
        if devices is None:
            devices = spec.get("devices")
    else:
        seed_value = seed

    # 输出必须可 JSON 序列化：NaN/Inf 归零，非标量（bytes、生成器）用其
    # 稳定文本表示作为种子，仍然满足「同输入同输出」。
    if isinstance(seed_value, bool) or seed_value is None:
        pass  # bool / None 本身可序列化
    elif isinstance(seed_value, int):
        pass
    elif isinstance(seed_value, float):
        if not math.isfinite(seed_value):
            seed_value = 0
    elif isinstance(seed_value, bytes):
        seed_value = seed_value.decode("utf-8", "backslashreplace")
    elif not isinstance(seed_value, str):
        seed_value = repr(seed_value)

    normalized = device_list(devices)
    ids = [device["id"] for device in normalized] if normalized else []
    if not ids:
        ids = ["master", "peer-1", "peer-2", "peer-3"]
        warnings.append("no devices supplied; default topology used")

    def derive(raw_seed: Any) -> tuple[list[str], dict[str, float], dict[str, float]]:
        order = sorted(str(device_id) for device_id in ids)
        jitter = {
            device_id: round(_stable_unit(raw_seed, device_id, "jitter") * 10.0, 6)
            for device_id in order
        }
        phases = {
            device_id: round(_stable_unit(raw_seed, device_id, "phase") * 1_000.0, 6)
            for device_id in order
        }
        return order, jitter, phases

    order, jitter_ms, phase_ms = derive(seed_value)

    def canonicalize(ord_: list[str], jit: Mapping[str, float], ph: Mapping[str, float]) -> str:
        return json.dumps(
            {
                "order": ord_,
                "jitter_ms": {device_id: jit[device_id] for device_id in ord_},
                "phase_ms": {device_id: ph[device_id] for device_id in ord_},
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    payload = canonicalize(order, jitter_ms, phase_ms)
    checksum = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # 重新从零推导一次，确认同一输入得到同一结果（而不是序列化同一份
    # 内存对象）。跨进程一致性由 SHA-256 与稳定排序保证。
    rerun_order, rerun_jitter, rerun_phase = derive(seed_value)
    repeatable = canonicalize(rerun_order, rerun_jitter, rerun_phase) == payload

    return {
        "seed": seed_value,
        "order": order,
        "jitter_ms": jitter_ms,
        "phase_ms": phase_ms,
        "checksum": checksum,
        "repeatable": repeatable,
        "algorithm": "sha256",
        "warnings": warnings,
    }
