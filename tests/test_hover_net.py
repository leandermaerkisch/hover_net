import pytest
import subprocess
import hover_net
from hover_net import dataloader, infer, metrics, misc, models, run_utils

def test_version():
    assert hasattr(hover_net, '__version__'), "hover_net should have a __version__ attribute"
    assert isinstance(hover_net.__version__, str), "__version__ should be a string"
    print(f"Hover-Net version: {hover_net.__version__}")

def test_submodule_imports():
    submodules = [dataloader, infer, metrics, misc, models, run_utils]
    for submodule in submodules:
        assert submodule is not None, f"{submodule.__name__} should be importable"

def test_infer_module():
    assert hasattr(infer, 'tile'), "infer module should have 'tile' attribute"

def test_models_module():
    assert hasattr(models, 'hovernet'), "models module should have 'hovernet' attribute"

@pytest.mark.parametrize("script", ["hover-net-infer", "hover-net-train"])
def test_cli_scripts(script):
    result = subprocess.run([script, "--help"], capture_output=True, text=True)
    assert result.returncode == 0, f"{script} --help should run without errors"
    assert "usage:" in result.stdout, f"{script} --help should display usage information"

# Add more specific tests for your package's functionality here