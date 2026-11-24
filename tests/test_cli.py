import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from spoofdet.cli import main
from spoofdet.logio import read_log

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_simulate_writes_a_log_that_detect_reads_back(tmp_path, capsys):
    path = tmp_path / "run.csv"
    assert main(["simulate", "--out", str(path), "--duration", "30", "--seed", "1"]) == 0
    assert "300 samples" in capsys.readouterr().out
    log = read_log(path)
    assert log.n_samples == 300
    assert log.has_mag and log.has_gyro and log.has_speed


def test_detect_on_a_clean_log_exits_zero(tmp_path, capsys):
    path = tmp_path / "clean.csv"
    main(["simulate", "--out", str(path), "--duration", "60", "--seed", "1"])
    assert main(["detect", str(path)]) == 0
    out = capsys.readouterr().out
    assert "no alarm" in out and "filter gyro" in out


def test_detect_on_an_attacked_log_exits_one(tmp_path, capsys):
    path = tmp_path / "jump.csv"
    main(
        [
            "simulate", "--out", str(path), "--duration", "60", "--seed", "2",
            "--attack", "jump", "--jump-m", "15", "--start", "30",
        ]
    )
    capsys.readouterr()
    assert main(["detect", str(path)]) == 1
    out = capsys.readouterr().out
    assert "alarm steps" in out
    assert "first at" in out


def test_json_output_is_one_parseable_object(tmp_path, capsys):
    path = tmp_path / "clean.csv"
    main(["simulate", "--out", str(path), "--duration", "30", "--seed", "1"])
    capsys.readouterr()
    assert main(["detect", str(path), "--json"]) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["filter"] == "gyro"
    assert report["log"]["samples"] == 300
    assert "note:" in captured.err


def test_a_log_on_stdin_can_be_used_as_report(tmp_path, capsys, monkeypatch):
    source = tmp_path / "clean.csv"
    main(["simulate", "--out", str(source), "--duration", "30", "--seed", "1"])
    capsys.readouterr()
    monkeypatch.setattr(sys, "stdin", io.StringIO(source.read_text()))
    assert main(["detect", "-"]) == 0
    assert "filter gyro" in capsys.readouterr().out


def test_simulate_to_stdout_prints_the_csv_and_the_note_goes_to_stderr(capsys):
    assert main(["simulate", "--duration", "5", "--seed", "1"]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("t,east,north,speed,yaw_rate,mag_forward,mag_left")
    assert "50 samples" in captured.err


def test_the_log_can_be_trimmed_to_the_sensors_of_the_filter(tmp_path, capsys):
    path = tmp_path / "bare.csv"
    main(
        [
            "simulate", "--out", str(path), "--duration", "30", "--seed", "1",
            "--no-gyro", "--no-speed", "--no-mag",
        ]
    )
    capsys.readouterr()
    assert main(["detect", str(path)]) == 0
    out = capsys.readouterr().out
    assert "filter gnss" in out and "nis gate" in out


def test_the_track_file_has_one_row_per_sample(tmp_path, capsys):
    path = tmp_path / "clean.csv"
    track = tmp_path / "track.csv"
    main(["simulate", "--out", str(path), "--duration", "30", "--seed", "1"])
    capsys.readouterr()
    assert main(["detect", str(path), "--track", str(track)]) == 0
    lines = track.read_text().splitlines()
    assert lines[0] == "t,east,north,psi,speed,nis,stat,alarm"
    assert len(lines) == 301
    assert lines[1].split(",")[4] == "nan"  # no state before the filter starts
    last = lines[-1].split(",")
    assert last[-1] == "0"
    assert np.isfinite(float(last[1])) and np.isfinite(float(last[2]))


@pytest.mark.parametrize(
    "argv, message",
    [
        (["detect", "missing.csv"], "cannot read"),
        (["detect", "-", "--drift", "0"], "drift must be positive"),
        (["detect", "-", "--threshold", "-1"], "threshold must be positive"),
        (["simulate", "--duration", "0"], "duration_s must be positive"),
    ],
)
def test_bad_input_exits_two_with_the_reason(tmp_path, capsys, monkeypatch, argv, message):
    if argv[0] == "detect":
        source = tmp_path / "clean.csv"
        main(["simulate", "--out", str(source), "--duration", "10", "--seed", "1"])
        capsys.readouterr()
        monkeypatch.setattr(sys, "stdin", io.StringIO(source.read_text()))
    assert main(argv) == 2
    assert message in capsys.readouterr().err


def test_a_filter_the_log_cannot_feed_says_which_column_is_missing(tmp_path, capsys):
    path = tmp_path / "gnss.csv"
    main(
        [
            "simulate", "--out", str(path), "--duration", "20", "--seed", "1",
            "--no-gyro", "--no-speed",
        ]
    )
    capsys.readouterr()
    assert main(["detect", str(path), "--filter", "gyro"]) == 2
    assert "no 'yaw_rate' column" in capsys.readouterr().err


def test_an_unknown_choice_is_an_argument_error(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["simulate", "--attack", "nudge"])
    assert raised.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_the_module_entry_point_runs_the_same_command_line(tmp_path):
    path = tmp_path / "clean.csv"
    assert main(["simulate", "--out", str(path), "--duration", "20", "--seed", "1"]) == 0
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    done = subprocess.run(
        [sys.executable, "-m", "spoofdet", "detect", str(path)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert done.returncode == 0
    assert "filter gyro" in done.stdout
    assert done.stderr == ""


def test_an_unwritable_track_path_does_not_take_the_report_with_it(tmp_path, capsys):
    path = tmp_path / "clean.csv"
    main(["simulate", "--out", str(path), "--duration", "20", "--seed", "1"])
    capsys.readouterr()
    code = main(["detect", str(path), "--track", str(tmp_path / "no" / "dir" / "t.csv")])
    assert code == 2
    assert "cannot write" in capsys.readouterr().err
