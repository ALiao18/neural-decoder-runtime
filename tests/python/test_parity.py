import json
import subprocess
import numpy as np
import torch
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/python'))

REPO_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
BINARY        = os.path.join(REPO_ROOT, "build/neural_decoder_runtime")
CONFIG        = os.path.join(REPO_ROOT, "artifacts/model_config.json")
CONSTANTS     = os.path.join(REPO_ROOT, "artifacts/constants.json")
FIXTURE       = os.path.join(REPO_ROOT, "tests/python/fixtures/parity_fixture.json")

def run_python_pipeline(fixture_path: str) -> torch.Tensor:
    """
    Run z-score, forward in python, returns [25, 40] logits.
    """
    import json as json_
    with open(fixture_path) as f:
        fixture = json_.load(f)
    
    block_id   = fixture["block_id"]
    input_vec  = fixture["input"]
    x          = torch.tensor(input_vec, dtype=torch.float32).reshape(25, 256)

    # load constants and zscore
    with open(CONSTANTS) as f:
        stats = json_.load(f)
    means  = torch.tensor(stats[block_id]["means"], dtype=torch.float32)
    stds   = torch.tensor(stats[block_id]["stds"], dtype=torch.float32)
    normed = (x - means) / stds

    # Load TorchScript model and run forward
    import torch as torch_
    model = torch_.jit.load(os.path.join(REPO_ROOT, 'artifacts/model.pt'))
    model.eval()
    with torch_.no_grad():
        output = model.forward(normed.unsqueeze(0)).squeeze(0) # [25, 40]

    return output

def run_cpp_pipeline(fixture_path: str) -> torch.Tensor:
    """
    shell out to C++ binary in parity mode, returns [25, 40] logits
    """
    result = subprocess.run(
        [BINARY, CONFIG, '--parity', fixture_path],
        capture_output = True,
        text = True,
        cwd = REPO_ROOT
    ) 

    if result.returncode != 0:
        pytest.fail(f"C++ binary failed:\n{result.stderr}")
    
    flat = json.loads(result.stdout.strip())
    return torch.tensor(flat, dtype=torch.float32).reshape(25, 40)

def test_parity():
    assert os.path.exists(BINARY),   f"Binary not found: {BINARY}"
    assert os.path.exists(CONFIG),   f"Config not found: {CONFIG}"

    py_output  = run_python_pipeline(FIXTURE)
    cpp_output = run_cpp_pipeline(FIXTURE)

    assert py_output.shape == (25, 40),  f"Python output shape wrong: {py_output.shape}, should be (25, 40)"
    assert cpp_output.shape == (25, 40), f"C++ output shape wrong: {cpp_output.shape}, should be (25, 40)"

    max_diff = (py_output - cpp_output).abs().max().item()
    print(f"\nMax absolute difference: {max_diff:.2e}")

    assert torch.allclose(py_output, cpp_output, atol=1e-5, rtol=1e-5), \
        f"Parity failed. Max absolute difference: {max_diff:.2e}"


