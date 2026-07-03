import json
import torch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/python'))

torch.manual_seed(0)
x = torch.randn(25, 256) # fixed input

fixture = {
    "block_id": "1", 
    "input"   : x.reshape(-1).tolist() # flat, C++ reshapes to [25, 256]
}

os.makedirs("tests/python/fixtures", exist_ok = True)
with open("tests/python/fixtures/parity_fixture.json", "w") as f:
    json.dump(fixture, f)

print("Fixutre written.")
print("input[:3]:", x.reshape(-1)[:3].tolist())
