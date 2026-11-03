"""Fetch only the EPT nodes of the OpenTopography Haiti 2010 LiDAR that cover the crop.

Run: uv run python -I scripts/fetch_ept.py <out_dir>
Downloads go to <out_dir> (untrusted data); a manifest of URL + sha256 is written to
data/manifest.json in this repo.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

BASE = "https://opentopography.s3.sdsc.edu/pc-bulk/Haiti/ept"
REPO = Path(__file__).resolve().parent.parent
CROP = json.loads((REPO / "data" / "crop.json").read_text())


def get(url: str, dest: Path) -> str:
    if not dest.exists():
        subprocess.run(["timeout", "120", "curl", "-fsSL", "-o", str(dest), url], check=True)
    return hashlib.sha256(dest.read_bytes()).hexdigest()


def main(out: Path) -> None:
    manifest = []
    ept_path = out / "ept.json"
    manifest.append({"url": f"{BASE}/ept.json", "file": "ept.json", "sha256": get(f"{BASE}/ept.json", ept_path)})
    ept = json.loads(ept_path.read_text())
    x0, y0, _, x1, y1, _ = ept["bounds"]
    cx0, cy0, cx1, cy1 = CROP["bbox_utm"]

    def node_box(d, x, y):
        w = (x1 - x0) / 2**d
        h = (y1 - y0) / 2**d
        return x0 + x * w, y0 + y * h, x0 + (x + 1) * w, y0 + (y + 1) * h

    def hits(key):
        d, x, y, _ = map(int, key.split("-"))
        bx0, by0, bx1, by1 = node_box(d, x, y)
        return bx0 < cx1 and bx1 > cx0 and by0 < cy1 and by1 > cy0

    nodes = {}
    todo = ["0-0-0-0"]
    while todo:
        hk = todo.pop()
        url = f"{BASE}/ept-hierarchy/{hk}.json"
        f = out / f"hier-{hk}.json"
        manifest.append({"url": url, "file": f.name, "sha256": get(url, f)})
        for key, count in json.loads(f.read_text()).items():
            if not hits(key):
                continue
            if count == -1:
                todo.append(key)
            elif count > 0:
                nodes[key] = count

    print(f"{len(nodes)} nodes, {sum(nodes.values())} points (before crop)")
    for key in sorted(nodes, key=lambda k: int(k.split("-")[0])):
        url = f"{BASE}/ept-data/{key}.laz"
        f = out / f"{key}.laz"
        manifest.append({"url": url, "file": f.name, "sha256": get(url, f), "points": nodes[key]})
    (REPO / "data" / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
