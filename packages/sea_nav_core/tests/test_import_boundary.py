import os
from pathlib import Path
import subprocess
import sys


def test_standalone_namespace_does_not_import_robotics_or_vendored_trainer():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1")
    code = "import sys; import sea_nav_core; assert not any(n.split('.')[0] in {'rsl_rl','isaacgym','isaaclab','rclpy','legged_gym'} for n in sys.modules); assert sea_nav_core.__version__ == '0.1.0'"
    result = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_pyproject_distribution_and_source_namespace():
    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text()
    assert 'name = "sea-nav-core"' in text
    assert 'dependencies = ["torch>=2.1"]' in text
    packages = [p.name for p in (root / "src").iterdir() if (p / "__init__.py").is_file()]
    assert packages == ["sea_nav_core"]
