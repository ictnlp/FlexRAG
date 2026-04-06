import json
import os
import subprocess
import sys
from pathlib import Path


def test_process_worker_inherits_visible_devices_before_import():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tests" / "support" / "run_process_worker_env_probe.py"

    env = os.environ.copy()
    pythonpath = [str(repo_root / "src"), str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath)
    env.pop("CUDA_VISIBLE_DEVICES", None)

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    reports = json.loads(result.stdout)

    assert [report["import_visible_devices"] for report in reports] == [
        "0",
        "1",
        "2",
        "3",
    ]
    assert [report["runtime_visible_devices"] for report in reports] == [
        "0",
        "1",
        "2",
        "3",
    ]
    assert all(report["config_device_id"] == [0] for report in reports)
