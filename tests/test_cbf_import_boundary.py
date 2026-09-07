"""Run only real package bootstrap statements; no simulator import or stand-in."""
import ast
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("full_method_runtime_smoke.py", "train_full_method_ppo.py",
           "train_full_method_acsi_replay_ppo.py")


def bootstrap(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    start = next(i for i, node in enumerate(main.body)
                 if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "adapter_root"
                                                         for t in node.targets))
    end = next(i for i, node in enumerate(main.body)
               if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("rsl_rl."))
    return tree, main, main.body[start:end + 1]


@pytest.mark.parametrize("script", SCRIPTS)
def test_bundled_selection_precedes_dependent_adapter_import(script):
    path = ROOT / "sea_nav_current_isaaclab_full_method" / script
    tree, main, nodes = bootstrap(path)
    adapter_import = next(n for n in nodes if isinstance(n, ast.ImportFrom) and n.module == "adapters.cbf_shield")
    purge = [n for n in nodes if isinstance(n, ast.For) and
             any(isinstance(child, ast.Delete) for child in ast.walk(n))]
    assert len(purge) == 1 and purge[0].lineno < adapter_import.lineno
    select = [n for n in nodes if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
              and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "insert"]
    assert len(select) == 1 and select[0].lineno < adapter_import.lineno
    # Main stays after the real application launcher, which these CPU tests do not run.
    launchers = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "AppLauncher"]
    assert len(launchers) == 1 and launchers[0].lineno < main.lineno


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("preload_real_core", [False, True])
def test_fresh_process_bootstrap_preserves_real_core_class_identity(script, preload_real_core, tmp_path):
    path = ROOT / "sea_nav_current_isaaclab_full_method" / script
    _, _, nodes = bootstrap(path)
    source = ast.unparse(ast.Module(body=nodes, type_ignores=[]))
    prelude = "from pathlib import Path\nimport sys\n__file__ = " + repr(str(path)) + "\n"
    if preload_real_core:
        prelude += ("sys.path.insert(0, " + repr(str(ROOT / "training/rsl_rl")) + ")\n"
                    "import rsl_rl.modules.cbf_lse_layer\nsys.path.pop(0)\n")
    checks = """
from adapters.cbf_shield import ExactLSECBFLayer as adapter_core, FootprintAwareLSECBFLayer
from rsl_rl.modules.cbf_lse_layer import ExactLSECBFLayer as training_core
import rsl_rl
assert adapter_core is training_core
assert isinstance(FootprintAwareLSECBFLayer(), training_core)
assert Path(rsl_rl.__file__).resolve().parent == sea_root / "training/rsl_rl/rsl_rl"
assert not any(name.startswith(("isaacgym", "isaaclab", "omni")) for name in sys.modules)
print("real bundled CBF class identity passed")
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", prelude + source + "\n" + checks],
                            cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "real bundled CBF class identity passed" in result.stdout
