"""
generate_falcon_tube.py
=======================
Procedurally build a watertight mesh of a standard unskirted 50 ml conical
centrifuge ("Falcon") tube with its screw cap, and write it as a binary STL.

The tube is a surface of revolution around +Z (bottom tip at z = 0, millimetres),
built from a radial profile r(z):

    z  0.0 - 21.0  conical bottom tapering to a small rounded tip
    z 21.0 - 101.5 cylindrical body, outer diameter 29 mm (slight 0.4 mm draft)
    z 101.5 - 114.5 cap, outer diameter 34 mm, with vertical grip ribs and a
                    chamfered top edge (the cap is wider than the body)

Dimensions follow the common 30 x 115 mm 50 ml tube format. The STL has no units;
the Blender generator imports it as millimetres, like the other models.

    python scripts/generate_falcon_tube.py                      # -> stl_models/Falcon_Tube.stl
    python scripts/generate_falcon_tube.py --segments 128 --out other.stl
"""

import argparse
import math
import struct
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BODY_R_BOTTOM = 14.5     # body outer radius at the cone junction (mm)
BODY_R_TOP = 14.7        # body outer radius under the cap (moulding draft)
CONE_HEIGHT = 21.0       # cone length from tip to body
TIP_R = 1.6              # radius of the rounded tip
BODY_TOP = 101.5         # z where the cap starts
CAP_R = 17.0             # cap outer radius between ribs
CAP_RIB_DEPTH = 0.35     # ribs stand this far out of the cap wall
CAP_RIBS = 48
CAP_TOP = 114.5          # overall length
CAP_CHAMFER = 0.8


def profile():
    """(z, r, ribbed) rings from tip to top; first and last rings collapse to the axis."""
    rings = [(0.0, 0.0, False)]
    for a in np.linspace(0.15, 1.0, 5) * (math.pi / 2):        # rounded tip, quarter arc
        rings.append((TIP_R * (1 - math.cos(a)) * 0.6, TIP_R * math.sin(a), False))
    z0 = rings[-1][0]
    for t in np.linspace(0.0, 1.0, 13)[1:]:                    # cone, slightly convex
        z = z0 + t * (CONE_HEIGHT - z0)
        r = TIP_R + (BODY_R_BOTTOM - TIP_R) * (t ** 0.92)
        rings.append((z, r, False))
    for t in np.linspace(0.0, 1.0, 9)[1:]:                     # body with draft
        rings.append((CONE_HEIGHT + t * (BODY_TOP - CONE_HEIGHT),
                      BODY_R_BOTTOM + t * (BODY_R_TOP - BODY_R_BOTTOM), False))
    rings.append((BODY_TOP, CAP_R, True))                       # cap underside step
    for z in np.linspace(BODY_TOP, CAP_TOP - CAP_CHAMFER, 6)[1:]:
        rings.append((float(z), CAP_R, True))                   # ribbed cap wall
    rings.append((CAP_TOP, CAP_R - CAP_CHAMFER, False))         # chamfered top edge
    rings.append((CAP_TOP, 0.0, False))
    return rings


def lathe(rings, segments):
    """Revolve the profile into an indexed, closed triangle mesh."""
    theta = np.linspace(0.0, 2 * math.pi, segments, endpoint=False)
    rib = (np.cos(theta * CAP_RIBS) > 0).astype(float) * CAP_RIB_DEPTH
    verts, ring_index = [], []
    for z, r, ribbed in rings:
        if r == 0.0:
            ring_index.append(len(verts))
            verts.append((0.0, 0.0, z))
            continue
        rr = r + (rib if ribbed else 0.0)
        start = len(verts)
        verts.extend(zip(rr * np.cos(theta), rr * np.sin(theta), np.full(segments, z)))
        ring_index.append(list(range(start, start + segments)))
    faces = []
    for a, b in zip(ring_index[:-1], ring_index[1:]):
        for k in range(segments):
            k1 = (k + 1) % segments
            if isinstance(a, int):                              # bottom fan (outward = down)
                faces.append((a, b[k1], b[k]))
            elif isinstance(b, int):                            # top fan
                faces.append((a[k], a[k1], b))
            else:
                faces.append((a[k], a[k1], b[k1]))
                faces.append((a[k], b[k1], b[k]))
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def write_binary_stl(path, verts, faces):
    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    with path.open("wb") as fh:
        fh.write(b"Falcon_Tube 50ml procedural (scripts/generate_falcon_tube.py)".ljust(80, b" "))
        fh.write(struct.pack("<I", len(faces)))
        rec = np.zeros(len(faces), dtype=[("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")])
        rec["n"] = normals
        rec["v"] = tri.reshape(-1, 9)
        fh.write(rec.tobytes())


def check_closed(faces):
    """Every edge must be shared by exactly two faces with opposite direction."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    directed = {tuple(e) for e in edges.tolist()}
    return len(directed) == len(edges) and all((b, a) in directed for a, b in directed)


def signed_volume(verts, faces):
    tri = verts[faces]
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def main():
    ap = argparse.ArgumentParser(description="Generate a procedural 50 ml Falcon tube STL")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "stl_models" / "Falcon_Tube.stl")
    ap.add_argument("--segments", type=int, default=96, help="vertices per ring (multiple of 2*ribs looks best)")
    args = ap.parse_args()
    verts, faces = lathe(profile(), args.segments)
    if not check_closed(faces):
        raise RuntimeError("mesh is not closed / consistently oriented")
    volume = signed_volume(verts, faces)
    if volume <= 0:
        raise RuntimeError("faces point inwards")
    write_binary_stl(args.out, verts, faces)
    lo, hi = verts.min(0), verts.max(0)
    print(f"wrote {args.out}  {len(verts)} verts, {len(faces)} tris, watertight")
    print(f"bbox {hi[0]-lo[0]:.1f} x {hi[1]-lo[1]:.1f} x {hi[2]-lo[2]:.1f} mm, "
          f"enclosed volume {volume/1000:.1f} ml (solid)")


if __name__ == "__main__":
    main()
