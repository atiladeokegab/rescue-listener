"""Crop the downloaded EPT nodes to the scene and write the web assets.

Run: uv run --no-project --with 'laspy[lazrs]' --with numpy --with scipy --with pyproj --with pillow \
       python -I scripts/build.py <laz_dir> [preview.png]

Writes data/ground.bin (heights, Uint16 cm over min_h, CELL m grid, row 0 = north),
data/ground_i.bin (Uint8 intensity grid) and data/meta.json. Scene frame: metres, origin at the crop
centre at median ground height, x east, y up, z south.
"""
import glob
import json
import sys
from pathlib import Path

import laspy
import numpy as np
from pyproj import Transformer
from PIL import Image
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
CELL = 0.5
VEG_SPLIT = 0.2  # share of multi-return pulses that marks a cell as canopy
VEG_SPLIT_GREEN = 0.06  # a lower share is enough where the 2010 photo is green too
EXG = 0.05  # excess green (2g - r - b, 0..1 scale) that reads as foliage in the photo
TREE_M2, TREE_LUM = 150, 0.3  # a lone dark blob smaller than this is a tree crown


def main(laz_dir: str, preview: str | None) -> None:
    crop = json.loads((REPO / "data" / "crop.json").read_text())
    bx0, by0, bx1, by1 = crop["bbox_utm"]
    cols = []
    for f in sorted(glob.glob(f"{laz_dir}/*.laz")):
        las = laspy.read(f)
        x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
        m = (x >= bx0) & (x < bx1) & (y >= by0) & (y < by1) & (np.asarray(las.classification) != 7)
        last = np.asarray(las.return_number) == np.asarray(las.number_of_returns)
        multi = np.asarray(las.number_of_returns) > 1
        cols.append(np.c_[x[m], y[m], z[m], np.asarray(las.intensity)[m], last[m], multi[m]])
    P = np.vstack(cols)
    raw = len(P)
    ground_med = float(np.median(P[:, 2]))
    # drop isolated high/low returns: birds, multipath (class 7 already gone)
    lo, hi = np.percentile(P[:, 2], 0.05) - 1, ground_med + 45
    P = P[(P[:, 2] > lo) & (P[:, 2] < hi)]
    assert len(P) > 300_000, len(P)

    n = int(round((bx1 - bx0) / CELL))
    ci = np.clip(((P[:, 0] - bx0) / CELL).astype(int), 0, n - 1)
    ri = np.clip(((by1 - P[:, 1]) / CELL).astype(int), 0, n - 1)  # row 0 = north
    H = np.full((n, n), -np.inf)
    L = P[:, 4] > 0  # last returns only: the pulse got through the canopy, so trees mostly drop out
    np.maximum.at(H, (ri[L], ci[L]), P[L, 2])
    I = np.zeros((n, n)); C = np.zeros((n, n))
    np.add.at(I, (ri, ci), P[:, 3]); np.add.at(C, (ri, ci), 1)
    empty = ~np.isfinite(H)
    idx = ndimage.distance_transform_edt(empty, return_distances=False, return_indices=True)
    H = H[tuple(idx)]
    I = np.where(C > 0, I / np.maximum(C, 1), 0)
    I = I[tuple(idx)]
    H = ndimage.median_filter(H, size=5)
    # grey opening with a 4.5 m square: trees and lamp-post spikes are narrower than any wall run, so
    # they drop to the ground under them while roofs and rubble keep their hard edges
    H = ndimage.grey_opening(H, size=(9, 9))
    # trees wider than that stay as square boxes. Canopy splits the pulse into several returns, roofs and
    # rubble give one (the photo's colour can't tell trees from the green roofs), so where most pulses
    # split, drop to a 15 m opening and the photo's trees lie on the ground they shade
    M = np.zeros((n, n)); np.add.at(M, (ri, ci), P[:, 5])
    share = ndimage.uniform_filter(M / np.maximum(C, 1), 5)
    # thinner canopy splits fewer pulses; there the photo's green decides. Roofs, green ones included, split
    # almost none, so neither test alone flattens them
    rgb = np.asarray(Image.open(REPO / "data" / "aerial.jpg").convert("RGB").resize((n, n), Image.BILINEAR), float) / 255
    green = ndimage.uniform_filter(2 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2], 5) > EXG
    veg = (share > VEG_SPLIT) | ((share > VEG_SPLIT_GREEN) & green)
    print("canopy cells:", int((share > VEG_SPLIT).sum()), "->", int(veg.sum()))
    veg = ndimage.binary_dilation(veg, iterations=2)
    O = ndimage.grey_opening(H, size=(31, 31))
    # what still stands alone: a blob over 2.5 m above the 15 m opening, under 150 m2 and dark in the photo is
    # a crown in shadow (crowns here read ~0.15-0.2; roofs and rubble ~0.4 and come in bigger blobs)
    lab, k = ndimage.label(H - O > 2.5)
    ids = np.arange(1, k + 1)
    dark = (ndimage.sum(np.ones((n, n)), lab, ids) * CELL ** 2 < TREE_M2) & (ndimage.mean(rgb.mean(-1), lab, ids) < TREE_LUM)
    crowns = ndimage.binary_dilation(np.isin(lab, ids[dark]), iterations=3)
    print("dark crowns:", int(dark.sum()), "blobs,", int(crowns.sum()), "cells")
    H = np.where(veg | crowns, np.minimum(H, O), H)
    I = ndimage.uniform_filter(I, size=3)

    min_h = float(H.min())
    (REPO / "data" / "ground.bin").write_bytes(np.round((H - min_h) * 100).astype("<u2").tobytes())
    i_hi = np.percentile(I, 99.5)
    (REPO / "data" / "ground_i.bin").write_bytes(np.clip(I / i_hi * 255, 0, 255).astype(np.uint8).tobytes())

    to_ll = Transformer.from_crs(32618, 4326, always_xy=True)
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    lon, lat = to_ll.transform(cx, cy)
    # affine scene (x, z) -> (lat, lon) for the on-screen readout; fit on a grid, checked
    gx, gz = np.meshgrid(np.linspace(-150, 150, 7), np.linspace(-150, 150, 7))
    glon, glat = to_ll.transform(cx + gx.ravel(), cy - gz.ravel())
    A = np.c_[np.ones(gx.size), gx.ravel(), gz.ravel()]
    k_lat = np.linalg.lstsq(A, glat, rcond=None)[0]
    k_lon = np.linalg.lstsq(A, glon, rcond=None)[0]
    assert np.abs(A @ k_lat - glat).max() < 2e-6 and np.abs(A @ k_lon - glon).max() < 2e-6
    meta = {
        "to_wgs84": {"lat": k_lat.tolist(), "lon": k_lon.tolist(), "form": "c0 + c1*x + c2*z"},
        "crs": "EPSG:32618",
        "bbox_utm": crop["bbox_utm"],
        "bbox_wgs84": [list(to_ll.transform(bx0, by0))[::-1], list(to_ll.transform(bx1, by1))[::-1]],
        "origin_utm": [cx, cy, ground_med],
        "origin_wgs84": [lat, lon],
        "cell_m": CELL, "size_px": n, "min_h": min_h, "max_h": float(H.max()),
        "ground_med": ground_med, "points": int(len(P)), "points_raw_in_crop": raw,
        "frame": "metres; origin at crop centre, median return height; x east, y up, z south",
        "source": crop["source"],
    }
    (REPO / "data" / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))

    if preview:  # hillshade for picking the heap by eye; not shipped
        gy, gx = np.gradient(H, CELL)
        shade = np.clip(0.5 + (-gx - gy) * 0.35, 0, 1)
        img = (shade * 0.6 + (H - min_h) / (H.max() - min_h) * 0.4) * 255
        hdr = f"P5 {n} {n} 255\n".encode()
        Path(preview).write_bytes(hdr + img.astype(np.uint8).tobytes())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
