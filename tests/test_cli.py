import numpy as np
import pytest

from spoofdet.cli import alarm_intervals, main, read_log

HEADER = "t_s,east_m,north_m,speed_mps,yaw_rate_radps"


def write(path, header, rows):
    path.write_text(header + "\n" + "\n".join(",".join(map(str, r)) for r in rows))
    return path


def test_read_log_rejects_missing_column(tmp_path):
    p = write(tmp_path / "a.csv", "t_s,east_m,north_m,speed_mps", [[0, 0, 0, 1]] * 3)
    with pytest.raises(ValueError, match="missing column.*yaw_rate_radps"):
        read_log(p)


def test_read_log_rejects_one_magnetometer_column(tmp_path):
    rows = [[0.1 * k, 0, 0, 1, 0, 20] for k in range(3)]
    p = write(tmp_path / "a.csv", HEADER + ",mag_fwd_ut", rows)
    with pytest.raises(ValueError, match="need both"):
        read_log(p)


def test_read_log_rejects_uneven_steps(tmp_path):
    # one sample dropped at t = 0.3 s
    rows = [[t, 0, 0, 1, 0] for t in (0.0, 0.1, 0.2, 0.4, 0.5)]
    p = write(tmp_path / "a.csv", HEADER, rows)
    with pytest.raises(ValueError, match="sample step 0.2 s at t = 0.2 s"):
        read_log(p)


def test_read_log_rejects_empty_cell(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text(HEADER + "\n0,0,0,1,0\n0.1,0,,1,0\n0.2,0,0,1,0\n")
    with pytest.raises(ValueError, match="line 3"):
        read_log(p)


def test_alarm_intervals_merges_alarms_closer_than_gap():
    t_s = np.arange(100) * 1.0
    alarms = np.zeros(100, dtype=bool)
    alarms[[10, 30, 45, 80]] = True
    # 10 -> 30 -> 45 are 20 and 15 s apart, 45 -> 80 is 35 s
    assert alarm_intervals(t_s, alarms, 30.0) == [(10.0, 45.0, 3), (80.0, 80.0, 1)]
    assert alarm_intervals(t_s, alarms, 15.0) == [
        (10.0, 10.0, 1),
        (30.0, 30.0, 1),
        (45.0, 45.0, 1),
        (80.0, 80.0, 1),
    ]
    assert alarm_intervals(t_s, np.zeros(100, dtype=bool), 30.0) == []


def test_detect_finds_sideways_drag_in_simulated_log(tmp_path, capsys):
    log = str(tmp_path / "log.csv")
    common = ["--duration-s", "300", "--seed", "1"]
    attack = ["--attack", "drag-across", "--size", "0.2"]
    main(["simulate", log, *common])
    assert main(["detect", log]) == 0
    main(["simulate", log, *common, *attack])
    capsys.readouterr()
    assert main(["detect", log]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert "magnetometer" in lines[0]
    # the attack starts at 60 s; CUSUM needs about 10 s at 0.2 m/s
    first_s = float(lines[1].split()[1])
    assert 60.0 < first_s < 90.0
    assert "across" in lines[1]


def test_detect_out_has_one_row_per_sample(tmp_path):
    log, out = tmp_path / "log.csv", tmp_path / "steps.csv"
    main(["simulate", str(log), "--duration-s", "30", "--no-mag"])
    main(["detect", str(log), "--out", str(out)])
    header = out.read_text().splitlines()[0]
    assert header == "t_s,nis,nu_along,nu_across,nu_speed,alarm"
    steps = np.loadtxt(out, delimiter=",", skiprows=1)
    assert steps.shape == (300, 6)
    # filter starts at the end of the 5 s heading fit
    assert np.isnan(steps[:50, 1]).all() and np.isfinite(steps[50:, 1]).all()


def test_detect_reports_bad_log_with_status_2(tmp_path, capsys):
    p = write(tmp_path / "a.csv", "t_s,east_m", [[0, 0]])
    assert main(["detect", str(p)]) == 2
    assert "missing column" in capsys.readouterr().err
