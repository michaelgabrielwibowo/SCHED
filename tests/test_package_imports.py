import subprocess
import sys
from pathlib import Path


def test_root_package_importable():
    repo_parent = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        "-c",
        (
            "from SCHEDULE import GreedyScheduler, BenchmarkGenerator, SFJSSPEnv; "
            "print(GreedyScheduler.__name__, BenchmarkGenerator.__name__, SFJSSPEnv.__name__)"
        ),
    ]
    result = subprocess.run(
        cmd,
        cwd=repo_parent,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
