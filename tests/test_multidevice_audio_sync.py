import pytest
import media.multidevice_audio_sync as mod

CASES = [
    ("clock_drift", (1,)),
    ("latency_table", (1,)),
    ("master_switch", (1,)),
    ("underrun_policy", (1,)),
    ("phase_align", (1,)),
    ("sync_monitor", (1,)),
    ("buffer_depth", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("device_list", (1,)),
    ("metrics", (1,)),
    ("drift_correct", (1,)),
    ("bt_latency", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
