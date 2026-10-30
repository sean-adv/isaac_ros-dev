import pytest

from dinov3_cases.timing import ResultRate


def test_first_result_and_context_change_start_a_new_measurement():
    rate = ResultRate()
    assert rate.update(1.0, "camera") is None
    assert rate.update(1.1, "camera") == pytest.approx(10)
    assert rate.update(20.0, "upload") is None
    assert rate.update(20.5, "upload") == pytest.approx(2)


def test_sam3_nvdcf_cycle_uses_total_elapsed_time_not_average_instant_fps():
    rate = ResultRate()
    rate.update(0.0, 1)
    now = 0.0
    # One 0.4s detection + four 0.025s tracking frames = 10 results/s.
    for _ in range(8):
        for duration in [.4, .025, .025, .025, .025]:
            now += duration
            fps = rate.update(now, 1)
    assert fps == pytest.approx(10)


def test_window_adapts_after_a_sustained_speed_change():
    rate = ResultRate()
    for tick in range(41):
        rate.update(tick / 10, 1)
    for tick in range(1, 81):
        fps = rate.update(4 + tick / 20, 1)
    assert fps == pytest.approx(20)
    assert len(rate.completed) <= 62


def test_long_interval_is_not_discarded():
    rate = ResultRate()
    rate.update(0.0, 1)
    rate.update(.1, 1)
    assert rate.update(10.1, 1) == pytest.approx(.1)


def test_nonincreasing_clock_restarts_measurement():
    rate = ResultRate()
    rate.update(2.0, 1)
    assert rate.update(2.0, 1) is None
    assert rate.update(1.0, 1) is None
    assert rate.update(1.5, 1) == pytest.approx(2)
