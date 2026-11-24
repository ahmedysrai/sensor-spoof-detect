import numpy as np
import pytest

from spoofdet.logio import Log, LogError, format_log, parse_log, read_log, write_log
from spoofdet.logsim import simulate_log


def test_csv_round_trip_keeps_every_column(tmp_path):
    log = simulate_log(duration_s=20.0, seed=4)
    path = write_log(tmp_path / "run.csv", log)
    back = read_log(path)
    assert back.n_samples == log.n_samples
    assert back.dt_s == pytest.approx(log.dt_s)
    np.testing.assert_allclose(back.t_s, log.t_s, atol=1e-9)
    np.testing.assert_allclose(back.fixes_m, log.fixes_m, rtol=1e-8)
    np.testing.assert_allclose(back.speed_mps, log.speed_mps, rtol=1e-8)
    np.testing.assert_allclose(back.yaw_rate_radps, log.yaw_rate_radps, rtol=1e-8)
    np.testing.assert_allclose(back.mag_ut, log.mag_ut, rtol=1e-8)


def test_jsonl_round_trip(tmp_path):
    log = simulate_log(duration_s=5.0, seed=1, with_mag=False)
    path = write_log(tmp_path / "run.jsonl", log)
    assert path.read_text().splitlines()[0].startswith('{"t"')
    back = read_log(path)
    assert not back.has_mag
    np.testing.assert_allclose(back.fixes_m, log.fixes_m, rtol=1e-8)


def test_csv_is_written_with_lf_only(tmp_path):
    # the repo is LF throughout; csv.writer defaults to CRLF
    path = write_log(tmp_path / "run.csv", simulate_log(duration_s=2.0))
    assert "\r" not in path.read_text()


def test_only_the_columns_the_log_carries_are_written(tmp_path):
    log = simulate_log(duration_s=2.0, with_mag=False, with_gyro=False)
    header = write_log(tmp_path / "run.csv", log).read_text().splitlines()[0]
    assert header == "t,east,north,speed"


def test_headers_are_matched_by_name_ignoring_case_and_spaces():
    text = "Time , East , NORTH \n0,1.0,2.0\n0.1,1.5,2.5\n"
    log = parse_log(text)
    np.testing.assert_allclose(log.fixes_m, [[1.0, 2.0], [1.5, 2.5]])
    assert log.dt_s == pytest.approx(0.1)


def test_an_empty_cell_is_a_dropout_not_an_error():
    text = "t,east,north\n0,1,2\n0.1,,\n0.2,1.4,2.4\n"
    log = parse_log(text)
    assert np.isnan(log.fixes_m[1]).all()
    assert log.telemetry()["gnss_dropouts"] == 1


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "empty"),
        ("t,north\n0,1\n1,2\n", "no 'east' column"),
        ("east,north\n1,2\n3,4\n", "sample interval is unknown"),
        ("t,east,north\n0,1,2\n", "the log holds 1 sample"),
        ("t,east,north\n0,1,2\n0.1,1.5,2.5\n0.05,2,3\n", "goes back"),
        ("t,east,north\n0,1,2\n0.1,1.5,2.5\n0.2,oops,3\n", "not a number"),
        ("t,east,east_m,north\n0,1,2,3\n", "both name the east column"),
    ],
)
def test_unusable_csv_is_refused_with_the_reason(text, message):
    with pytest.raises(LogError, match=message):
        parse_log(text, source="run.csv")


def test_a_jittery_log_is_refused_and_can_be_forced():
    times = ["0.0", "0.1", "0.2", "0.35"] + [f"{0.1 * k:.1f}" for k in range(4, 10)]
    text = "t,east,north\n" + "".join(f"{t},0,0\n" for t in times)
    with pytest.raises(LogError, match="not 0.1000 s"):
        parse_log(text)
    log = parse_log(text, jitter_tolerance=None)
    assert log.n_samples == 10


def test_a_missing_dt_is_required_without_a_time_column():
    text = "east,north\n0,0\n1,1\n"
    with pytest.raises(LogError, match="sample interval is unknown"):
        parse_log(text)
    log = parse_log(text, dt_s=0.5)
    np.testing.assert_allclose(log.t_s, [0.0, 0.5])


def test_jsonl_reports_the_line_it_could_not_read():
    text = '{"t": 0, "east": 0, "north": 0}\n{"t": 0.1, "east": 1, "north": 1\n'
    with pytest.raises(LogError, match="run.jsonl:2"):
        parse_log(text, source="run.jsonl")


def test_jsonl_records_need_not_carry_every_column():
    text = (
        '{"t": 0, "east": 0, "north": 0, "speed": 2.0}\n'
        '{"t": 0.1, "east": 0.2, "north": 0, "speed": 2.0}\n'
    )
    log = parse_log(text)
    assert log.has_speed and not log.has_gyro and not log.has_mag


def test_check_refuses_arrays_of_different_lengths():
    log = Log(t_s=np.arange(4) * 0.1, fixes_m=np.zeros((4, 2)), speed_mps=np.zeros(3))
    with pytest.raises(LogError, match="speed_mps holds 3 values for 4 timestamps"):
        log.check()


def test_format_log_is_the_same_text_write_log_puts_on_disk(tmp_path):
    log = simulate_log(duration_s=2.0, with_gyro=False)
    path = write_log(tmp_path / "run.csv", log)
    assert format_log(log) == path.read_text()
