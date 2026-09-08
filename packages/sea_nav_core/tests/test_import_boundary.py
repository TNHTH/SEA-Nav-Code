import os
from pathlib import Path
import subprocess
import sys


def test_standalone_namespace_does_not_import_robotics_or_vendored_trainer():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1")
    code = "import sys; import sea_nav_core; assert not any(n.split('.')[0] in {'rsl_rl','isaacgym','isaaclab','rclpy','legged_gym'} for n in sys.modules); assert sea_nav_core.__version__ == '0.2.0'"
    result = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_pyproject_distribution_and_source_namespace():
    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text()
    assert 'requires = ["setuptools>=77"]' in text
    assert 'name = "sea-nav-core"' in text
    assert 'version = "0.2.0"' in text
    assert 'dependencies = ["torch>=2.1"]' in text
    assert 'license = "MIT"' in text
    assert 'license-files = ["LICENSE"]' in text
    assert '174231229+TNHTH@users.noreply.github.com' in text
    packages = [p.name for p in (root / "src").iterdir() if (p / "__init__.py").is_file()]
    assert packages == ["sea_nav_core"]


def test_scoped_mit_license_is_present_and_declared_for_both_build_artifacts():
    root = Path(__file__).resolve().parents[1]
    license_text = (root / "LICENSE").read_text()
    assert license_text.startswith("MIT License\n\nCopyright (c) 2026 TNHTH\n")
    assert "Permission is hereby granted, free of charge" in license_text
    assert "THE SOFTWARE IS PROVIDED \"AS IS\"" in license_text
    manifest = (root / "MANIFEST.in").read_text().splitlines()
    assert "include LICENSE" in manifest
    readme = (root / "README.md").read_text()
    assert "The included MIT license" in readme
    assert "`packages/sea_nav_core` distribution" in readme
    assert "does not relicense the containing" in readme
