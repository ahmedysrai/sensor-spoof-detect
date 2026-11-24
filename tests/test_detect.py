import json

import numpy as np
import pytest

from spoofdet.detect import TRACK_COLUMNS, DetectConfig, detect
from spoofdet.logio import LogError
from spoofdet.logsim import simulate_log


def test_auto_picks_the_filter_the_sensors_support():
    full = simulate_log(duration_s=20.0)
    assert detect(full).filter == "gyro"
    assert detect(simulate_log(duration_s=20.0, with_gyro=False)).filter == "gnss-odo"
    assert detect(
        simulate_log(duration_s=20.0, with_gyro=False, with_speed=False)
    ).filter == "gnss"


def test_asking_for_a_filter_the_log_cannot_feed_says_what_is_missing():
    with pytest.raises(ValueError, match="no 'yaw_rate' column"):
        detect(simulate_log(duration_s=20.0, with_gyro=False), DetectConfig(filter="gyro"))
    with pytest.raises(ValueError, match="no 'speed' column"):
        detect(
            simulate_log(duration_s=20.0, with_speed=False), DetectConfig(filter="gnss-odo")
        )
    with pytest.raises(ValueError, match="filter must be one of"):
        detect(simulate_log(duration_s=20.0), DetectConfig(filter="ukf"))


def test_a_clean_run_does_not_alarm():
    detection = detect(simulate_log(duration_s=60.0, seed=1))
    assert detection.detector == "cusum"
    assert detection.columns == TRACK_COLUMNS + ("heading",)
    assert detection.alarm_steps == 0
    assert detection.first_alarm_s is None
    # a matched 4-dof filter has a mean NIS of 4
    assert detection.mean_nis == pytest.approx(4.0, abs=0.25)


def test_a_slow_sideways_drag_is_found_after_it_starts():
    # 0.1 m/s across the track from 40 s: every single innovation stays small,
    # the lean on the across column is what the CUSUM accumulates
    log = simulate_log(duration_s=200.0, seed=2, attack="drag-across", rate_mps=0.1, start_s=40.0)
    detection = detect(log)
    assert detection.first_alarm_s is not None
    assert 40.0 <= detection.first_alarm_s < 120.0
    assert detection.first_alarm_by_column["across"] == pytest.approx(detection.first_alarm_s)


def test_without_the_magnetometer_the_heading_column_is_absent():
    with_mag = detect(simulate_log(duration_s=60.0, seed=1))
    without = detect(simulate_log(duration_s=60.0, seed=1, with_mag=False))
    assert with_mag.dof == 4 and with_mag.nu_track.shape[1] == 4
    assert without.dof == 3 and without.nu_track.shape[1] == 3
    assert "heading" not in without.columns
    assert any("no magnetometer" in note for note in without.warnings)


def test_turning_the_magnetometer_off_keeps_the_row_out():
    detection = detect(
        simulate_log(duration_s=60.0, seed=1), DetectConfig(mag_std_rad=None)
    )
    assert detection.dof == 3
    assert any("mag_std_rad is off" in note for note in detection.warnings)


def test_the_offset_state_is_estimated_when_asked_for():
    config = DetectConfig(bias_std_radps=0.0175, bias_rw_radps=1.1e-4)
    log = simulate_log(duration_s=120.0, seed=4, gyro_bias_radps=0.0175)
    detection = detect(log, config)
    assert detection.xs.shape[1] == 5
    # the fifth state is the offset, and the run is told it is about 1 deg/s
    # the state holds the offset as the filter models it, x = rate - b
    assert np.mean(detection.xs[60:, 4]) == pytest.approx(0.0175, abs=0.008)
    assert detection.alarm_steps == 0


def test_the_constant_velocity_paths_use_the_nis_gate():
    odo = detect(simulate_log(duration_s=60.0, seed=1, with_gyro=False))
    assert odo.detector == "nis gate"
    assert odo.columns == ()
    assert odo.dof == 3
    assert odo.mean_nis == pytest.approx(3.0, abs=0.3)

    gnss = detect(simulate_log(duration_s=60.0, seed=1, with_gyro=False, with_speed=False))
    assert gnss.detector == "nis gate" and gnss.dof == 2
    assert gnss.mean_nis == pytest.approx(2.0, abs=0.3)


def test_the_nis_gate_catches_a_jump_and_the_rate_it_reports_matches_its_p():
    log = simulate_log(
        duration_s=60.0, seed=2, attack="jump", jump_m=15.0, start_s=30.0,
        with_gyro=False, with_speed=False,
    )
    detection = detect(log)
    assert detection.first_alarm_s is not None
    assert detection.first_alarm_s >= 30.0
    assert detection.expected_clean_alarms_per_hour == pytest.approx(
        detection.config.nis_p_false_alarm * 3600.0 / log.dt_s
    )


def test_a_dropped_fix_is_skipped_and_the_run_keeps_going():
    log = simulate_log(duration_s=60.0, seed=1)
    log.fixes_m[300] = np.nan
    detection = detect(log)
    # the two position rows are gone on that step, the others still update
    assert np.isnan(detection.nu_track[300, :2]).all()
    assert np.isfinite(detection.nu_track[300, 2])
    assert np.isfinite(detection.xs[301:]).all()
    # the track around the dropout stays where the same run without it was
    clean = detect(simulate_log(duration_s=60.0, seed=1))
    assert np.linalg.norm(detection.xs[350, :2] - clean.xs[350, :2]) < 2.0


def test_a_dropout_inside_the_start_window_is_refused():
    log = simulate_log(duration_s=60.0, seed=1)
    log.fixes_m[10] = np.nan
    with pytest.raises(LogError, match="the fix of sample 10 is missing"):
        detect(log)


def test_a_run_too_short_for_the_start_window_is_refused():
    log = simulate_log(duration_s=3.0, seed=1)
    with pytest.raises(LogError, match="the filter starts after 50 samples"):
        detect(log)


def test_a_bad_parameter_is_named():
    log = simulate_log(duration_s=20.0, seed=1)
    with pytest.raises(ValueError, match="gnss_std_m must be positive"):
        detect(log, DetectConfig(gnss_std_m=0.0))
    with pytest.raises(ValueError, match="n_init must be at least 2"):
        detect(log, DetectConfig(n_init=1))


def test_the_report_is_json_ready_and_says_what_ran():
    detection = detect(simulate_log(duration_s=30.0, seed=1, with_mag=False))
    report = detection.to_dict()
    text = json.dumps(report)  # no arrays, no NaN tokens
    assert "NaN" not in text
    assert report["filter"] == "gyro"
    assert report["detector"] == "cusum"
    assert report["log"]["samples"] == 300
    assert report["nis"]["dof"] == 3
    assert report["alarms"]["steps"] == detection.alarm_steps
    assert set(report["alarms"]["by_column"]) <= set(TRACK_COLUMNS)


def test_the_summary_mentions_the_first_alarm_and_the_rate():
    witness = simulate_log(
        duration_s=120.0, seed=3, attack="drag-across", rate_mps=0.2, start_s=30.0
    )
    summary = detect(witness).summary()
    assert "alarm steps" in summary and "alarms/h" in summary
    assert "filter gyro" in summary
