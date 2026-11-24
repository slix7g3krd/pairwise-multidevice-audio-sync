# pairwise-multidevice-audio-sync

## 领域

游戏、图形与多媒体

## 主题

多设备音频同步

## 说明

时钟漂移估计、主切换连续与相位对齐。

实现位于 `media/multidevice_audio_sync.py`，按六个环节组织：

| 环节 | 入口 | 要点 |
| --- | --- | --- |
| 时钟漂移估计 | `clock_drift` / `drift_correct` | Theil–Sen 稳健斜率，输出 ppm、截距与残差 |
| 延迟补偿表 | `latency_table` / `bt_latency` | 实测优先，否则按传输/编解码器查表；统一对齐到最慢设备 |
| 主切换时间基连续 | `master_switch` | 只平移时间基，会话时间、播放位置与相位不变 |
| 统一欠载策略 | `underrun_policy` / `classify_underrun` | 余弦淡出/淡入 + 静音填充，相位与计数器不重置 |
| 相位对齐 | `phase_align` | 圆均值求参考相位，给出最小有符号校正量 |
| 同步误差监测 | `sync_monitor` | 偏置、抖动、P95、越限比例与漂移趋势告警 |

辅助入口：`buffer_depth`、`validation`、`audit`、`fallback`、`device_list`、
`metrics`、`determinism`。

所有入口都是纯函数，非法输入返回带 `warnings` 的中性结果而不抛异常，
输出可直接 JSON 序列化。

运行测试：

```bash
python -m pytest -q
```

期望全部通过。
