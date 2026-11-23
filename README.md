# pairwise-multidevice-audio-sync

## 领域

游戏、图形与多媒体

## 主题

多设备音频同步

## 说明

时钟漂移估计、主切换连续与相位对齐。

初始状态下 `media/multidevice_audio_sync.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
