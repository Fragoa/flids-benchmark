"""Verify the readable historical engines against their distributed Kaggle notebooks."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "notebooks/legacy"
SOURCES = [
    ("FLIDS_Master_Experiments01_to07_ExpandedBaselines_Kaggle.ipynb", "EMBEDDED", "early"),
    ("FLIDS_Experiment08_09_Full_Kaggle_AutoRun.ipynb", "PAYLOADS", "retuning"),
]

def verify(notebook, variable, directory):
    cells = json.loads((LEGACY / notebook).read_text())["cells"]
    payloads = []
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == variable:
                payloads.append(ast.literal_eval(node.value))
    if len(payloads) != 1:
        raise ValueError(f"Expected one {variable} dictionary in {notebook}")
    for name, record in payloads[0].items():
        packed = record["packed"] if isinstance(record, dict) else record
        data = zlib.decompress(base64.b64decode(packed, validate=True))
        if isinstance(record, dict) and hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"Embedded digest mismatch: {name}")
        path = ROOT / "experiments" / directory / name
        if path.read_bytes() != data:
            raise ValueError(f"Source differs from notebook: {path}")
        compile(data, str(path), "exec")
    return len(payloads[0])

if __name__ == "__main__":
    print(f"Verified {sum(verify(*entry) for entry in SOURCES)} embedded modules.")
