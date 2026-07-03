import json
import os
import subprocess
import sys
from pathlib import Path


def test_process_worker_inherits_visible_devices_before_import():
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root
        / "tests"
        / "support"
        / "process"
        / "run_process_worker_env_probe.py"
    )

    env = os.environ.copy()
    pythonpath = [str(repo_root / "src"), str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath)
    env["CUDA_VISIBLE_DEVICES"] = "9"

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    reports = json.loads(result.stdout)

    explicit_reports = reports["explicit"]
    assert [report["import_visible_devices"] for report in explicit_reports] == [
        "0",
        "1",
        "2",
        "3",
    ]
    assert [report["runtime_visible_devices"] for report in explicit_reports] == [
        "0",
        "1",
        "2",
        "3",
    ]

    cpu_report = reports["cpu"][0]
    assert cpu_report["import_visible_devices"] == ""
    assert cpu_report["runtime_visible_devices"] == ""

    inherit_report = reports["inherit"][0]
    assert inherit_report["import_visible_devices"] == "9"
    assert inherit_report["runtime_visible_devices"] == "9"
