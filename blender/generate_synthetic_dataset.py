"""
generate_synthetic_dataset.py
=============================
Headless Blender (bpy) pipeline that turns a handful of labware STL files into a
domain-randomised, YOLOv8-OBB-labelled synthetic dataset for Sim2Real training
of the UR16e labware detector.

Pipeline (one run)
------------------
1. Scene setup   : wipe the default scene, import every STL, scale mm -> m, fix the
                   up-axis, re-centre each mesh so it stands on z = 0.
2. Assembly      : find the tube holes in the TubeHolder by ray-casting a grid down
                   onto it (trying every face as "up"), keep the holes the tube body
                   actually fits, drop one tube instance into each, and put the loose
                   objects on the bench around the holder.
3. Materials     : Principled BSDF glass (transmission 1.0, low roughness) for the
                   transparent objects, matte plastic for the holder, a configurable
                   bench plane and world background.
4. Randomisation : per frame -> camera on a dome around the holder (distance, lens,
                   roll), 1-3 point lights sharing a bounded power budget (pose / colour
                   / softness), a procedurally textured bench (wood grain, brushed metal,
                   noise, tiles, fine grain - or your own images via --textures-dir),
                   a mottled or HDRI world, plastic colour, glass tint, camera exposure,
                   which holes are filled, where the loose objects lie. Frames that still
                   come out with > 10 % blown-out pixels are re-rendered 1 EV darker.
5. Label export  : every rendered object is projected through the camera, the convex
                   hull of the projected points is clipped to the frame and wrapped in
                   its minimum-area rotated rectangle. Written as
                   `cls x1 y1 x2 y2 x3 y3 x4 y4` with coordinates normalised to [0, 1].
                   --label-mode visible (default) boxes only the part of an object the
                   camera can actually see (glass counts as see-through); amodal boxes
                   the whole object even when it is buried inside an opaque rack.

Output layout
-------------
<out_dir>/
    images/train/000000.png ...
    labels/train/000000.txt ...
    data.yaml                      (class names, ready for `yolo obb train`)

Running (headless / background mode)
------------------------------------
Everything after the bare `--` belongs to this script, everything before it to Blender.

Windows (PowerShell or cmd, from the project root):
    & "C:\\Program Files\\Blender Foundation\\Blender 4.3\\blender.exe" -b -P blender\\generate_synthetic_dataset.py -- --num-images 1000

Linux / macOS:
    blender -b -P blender/generate_synthetic_dataset.py -- --num-images 1000

Quick smoke test (5 fast frames into a scratch folder), then eyeball the labels:
    blender -b -P blender/generate_synthetic_dataset.py -- --num-images 5 --samples 16 --out-dir dataset_smoke
    python scripts/visualize_obb_labels.py --dataset dataset_smoke --all

Flags: --stl-dir --out-dir --num-images --start-index --resolution --samples
       --engine {cycles,eevee} --label-mode {visible,amodal} --seed
       --textures-dir --hdri-dir --overwrite
Frames are seeded individually, so re-running with the same --seed reproduces the same
frames and a crashed run can be resumed (existing frames are skipped unless --overwrite).
"""

import argparse
import colorsys
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import bmesh
import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import convex_hull_2d

# =============================================================================== #
# 0. CONFIGURATION                                                                #
# =============================================================================== #

# This file lives in <project>/blender/, so the project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()

# Which STL plays which role.
#   role     : 'holder' -> the rack whose holes are detected and filled
#              'tube'   -> instanced once per fitting hole, stands upright inside it
#              'loose'  -> stands on the bench next to the holder
#   material : 'glass' (transparent) or 'plastic' (matte, randomly coloured)
#   class_id : YOLO class index written to the label files
#   pre_rotate_deg : optional extra rotation (X, Y, Z degrees) applied after the global
#                    up-axis fix, for files exported with an unusual orientation.
# Geometry of the shipped files (measured): TubeHolder.stl is a 3-way rack (4 x 30 mm,
# 12 x 17 mm and 32 x 12 mm holes on different faces), Tube.stl is a capped 50 ml conical
# tube (30 mm body, 35 mm cap), eppendorf_tarto.stl is a 96-well PCR rack lying flat and
# Slide_holder.stl is a small stand. The last two are rack-shaped, so they cannot sit in a
# tube hole and are placed on the bench; they render as glass as requested - switch
# material to 'plastic' if the real parts are opaque.
OBJECT_SPECS = [
    dict(file="TubeHolder.stl",      class_id=0, name="Tube_Holder",    role="holder", material="plastic"),
    dict(file="Tube.stl",            class_id=1, name="Glass_Tube",     role="tube",   material="glass"),
    dict(file="Slide_holder.stl",    class_id=2, name="Slide_Holder",   role="loose",  material="glass"),
    dict(file="eppendorf_tarto.stl", class_id=3, name="Eppendorf_Rack", role="loose",  material="glass"),
]

# STL carries no unit or axis metadata. These files are in millimetres with +Y as "up"
# (CAD convention). The holder's up-face and the tube's up-end are verified geometrically
# below, so a wrong guess here is corrected automatically for those two; loose objects
# rely on this setting (plus pre_rotate_deg).
STL_UNIT_SCALE = 0.001            # mm -> m (Blender works in metres)
SOURCE_UP_AXIS = "Y"              # 'Y' or 'Z'

# Camera dome around the holder (metres / degrees)
CAM_DIST_RANGE = (0.50, 0.95)
CAM_ELEV_RANGE_DEG = (12.0, 80.0)
CAM_ROLL_MAX_DEG = 8.0
CAM_LENS_RANGE_MM = (35.0, 55.0)
CAM_TARGET_JITTER = 0.04          # look-at point jitter so the holder is not always centred

# Lighting. Power is a budget for the WHOLE frame: it is split across the 1-3 lights, so
# three lights never add up to three times the exposure of one.
NUM_LIGHTS_RANGE = (1, 3)
LIGHT_DIST_RANGE = (0.5, 1.3)
LIGHT_ELEV_RANGE_DEG = (20.0, 85.0)
LIGHT_TOTAL_POWER_RANGE_W = (20.0, 75.0)     # summed over all lights, normalised to 0.8 m distance
LIGHT_RADIUS_RANGE = (0.01, 0.25)            # point-light radius -> hard vs soft shadows
WORLD_STRENGTH_RANGE = (0.04, 0.20)          # ambient fill from the world background
EXPOSURE_RANGE_EV = (-1.5, 0.0)              # camera auto-exposure variation, biased dark
MAX_ALBEDO = 0.75                            # no near-white bench/world: glass silhouettes stay readable
# Auto-exposure guard, evaluated on the saved PNG (display space, 0-1): a frame is accepted
# when its mean brightness is inside TARGET_MEAN_RANGE and few pixels are blown out;
# otherwise the exposure is stepped and the frame re-rendered (labels are unaffected).
TARGET_MEAN_RANGE = (0.28, 0.60)
MAX_CLIPPED_FRACTION = 0.10
MAX_EXPOSURE_RETRIES = 3

# Scene composition
HOLE_FILL_PROB = 0.75             # chance each hole holds a tube (at least one is always filled)
LOOSE_PRESENT_PROB = 0.85         # chance each loose object is in the scene at all
LOOSE_MAX_DIST = 0.08             # metres beyond touching distance a loose object may lie
HOLE_GRID_STEP = 0.001            # ray-cast grid resolution for hole detection (1 mm)
HOLE_MIN_DEPTH = 0.004            # a depression must be >= 4 mm deep to count as a hole
HOLE_FIT_TOLERANCE = 0.001        # tube body may exceed the hole radius by this much (1 mm)
PLASTIC_PALETTE = [               # typical lab-rack colours (linear RGB)
    (0.90, 0.90, 0.90), (0.05, 0.15, 0.70), (0.70, 0.05, 0.05), (0.05, 0.45, 0.12),
    (0.90, 0.70, 0.05), (0.02, 0.02, 0.02), (0.85, 0.30, 0.03), (0.35, 0.05, 0.60), (0.40, 0.40, 0.42),
]

# Labelling
MIN_VISIBLE_FRACTION = 0.25       # drop labels for objects mostly outside the frame
MIN_BOX_AREA = 2e-5               # normalised area (~8 px^2 at 640x640); drops edge-on slivers
VISIBILITY_SAMPLES = 800          # surface points per object tested for occlusion (visible mode)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
HDRI_EXTS = IMAGE_EXTS | {".hdr", ".exr"}


# =============================================================================== #
# 1. GENERIC HELPERS                                                              #
# =============================================================================== #

def log(msg):
    print(f"[synth] {msg}", flush=True)


def parse_args():
    """Blender consumes its own CLI flags; everything after a bare `--` is ours."""
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(
        prog="blender -b -P blender/generate_synthetic_dataset.py --",
        description="Render a domain-randomised YOLOv8-OBB dataset from labware STL files.")
    p.add_argument("--stl-dir", default=str(PROJECT_ROOT / "stl_models"),
                   help="folder containing the .stl files listed in OBJECT_SPECS")
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "dataset"),
                   help="dataset root; images/train and labels/train are created inside")
    p.add_argument("--num-images", type=int, default=1000)
    p.add_argument("--start-index", type=int, default=0, help="first frame index (resume / shard runs)")
    p.add_argument("--resolution", type=int, default=640, help="square render size in pixels")
    p.add_argument("--samples", type=int, default=64, help="render samples per pixel (denoised)")
    p.add_argument("--engine", choices=("cycles", "eevee"), default="cycles",
                   help="cycles gives physically correct glass; eevee is a fast preview")
    p.add_argument("--label-mode", choices=("visible", "amodal"), default="visible",
                   help="visible: box only what the camera sees; amodal: box the whole object")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--textures-dir", default=None, help="optional folder of images used as bench textures")
    p.add_argument("--hdri-dir", default=None, help="optional folder of HDRIs/images used as world backgrounds")
    p.add_argument("--overwrite", action="store_true", help="re-render frames whose outputs already exist")
    return p.parse_args(argv)


def set_input(node, candidate_names, value):
    """Set a shader socket by trying several names (Blender 4.x renamed Principled inputs)."""
    for name in candidate_names:
        sock = node.inputs.get(name)
        if sock is not None:
            sock.default_value = value
            return True
    return False


def list_images(folder, exts):
    if not folder:
        return []
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in exts)


def load_image(S, path):
    """Load an image datablock once and cache it for the rest of the run."""
    key = str(path)
    if key not in S.image_cache:
        S.image_cache[key] = bpy.data.images.load(key, check_existing=True)
    return S.image_cache[key]


# =============================================================================== #
# 2. SCENE SETUP & STL IMPORT                                                     #
# =============================================================================== #

def reset_scene():
    """Start from a truly empty file: no default cube, camera, light or world."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for ob in list(bpy.data.objects):                      # belt and braces
        bpy.data.objects.remove(ob, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights,
                 bpy.data.cameras, bpy.data.images, bpy.data.worlds):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)


def import_stl(path, pre_rotate_deg=(0.0, 0.0, 0.0)):
    """Import one STL, bake its transform into the mesh (local == world), fix the up axis."""
    before = {o.name for o in bpy.data.objects}
    if hasattr(bpy.ops.wm, "stl_import"):                  # Blender >= 4.0 (C++ importer)
        bpy.ops.wm.stl_import(filepath=str(path), global_scale=STL_UNIT_SCALE)
    else:                                                   # Blender 3.x legacy add-on
        bpy.ops.import_mesh.stl(filepath=str(path), global_scale=STL_UNIT_SCALE)
    new = [o for o in bpy.data.objects if o.name not in before]
    if len(new) != 1:
        raise RuntimeError(f"expected exactly one object from {path.name}, got {len(new)}")
    obj = new[0]
    obj.data.transform(obj.matrix_world)                    # bake any object-level transform
    obj.matrix_world = Matrix.Identity(4)
    if SOURCE_UP_AXIS.upper() == "Y":                       # Y-up (CAD) -> Z-up (Blender)
        obj.data.transform(Matrix.Rotation(math.radians(90.0), 4, "X"))
    rx, ry, rz = (math.radians(a) for a in pre_rotate_deg)
    if rx or ry or rz:
        obj.data.transform(Matrix.Rotation(rz, 4, "Z") @ Matrix.Rotation(ry, 4, "Y") @ Matrix.Rotation(rx, 4, "X"))
    clean_mesh(obj)
    return obj


def clean_mesh(obj, sharp_angle_deg=30.0):
    """Weld the STL triangle soup, fix normals, smooth-shade with sharp creases preserved."""
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    limit = math.radians(sharp_angle_deg)
    for f in bm.faces:
        f.smooth = True
    for e in bm.edges:
        angle = e.calc_face_angle(-1.0)                     # -1 when the edge is not 2-manifold
        e.smooth = 0.0 <= angle <= limit                    # smooth=False marks the edge sharp
    bm.to_mesh(me)
    bm.free()
    if hasattr(me, "use_auto_smooth"):                      # Blender <= 4.0 needs this flag
        me.use_auto_smooth = True
        me.auto_smooth_angle = math.pi
    me.update()


def mesh_arrays(obj):
    """Vertex positions (N, 3) and polygon index tuples of a mesh, in local coordinates."""
    me = obj.data
    co = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    return co.reshape(-1, 3), [tuple(p.vertices) for p in me.polygons]


def mesh_bounds(obj):
    co, _ = mesh_arrays(obj)
    return co.min(axis=0), co.max(axis=0)


def footprint_radius(obj):
    """Radius of the circle enclosing the object's XY footprint (used for collision checks)."""
    lo, hi = mesh_bounds(obj)
    return 0.5 * math.hypot(hi[0] - lo[0], hi[1] - lo[1])


def recenter_on_ground(obj):
    """Translate the mesh so its bbox centre is on the Z axis and its lowest point is at z = 0."""
    lo, hi = mesh_bounds(obj)
    obj.data.transform(Matrix.Translation(Vector((-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, -lo[2]))))


def convex_hull_vertices(obj):
    """3-D convex-hull vertices of a mesh (local coords).

    Under perspective projection the 2-D silhouette hull is spanned by images of 3-D hull
    vertices, so projecting only these (tens to hundreds of points instead of tens of
    thousands) gives exactly the same oriented bounding box, much faster.
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    try:
        res = bmesh.ops.convex_hull(bm, input=bm.verts)
        pts = np.array([v.co[:] for v in res["geom"] if isinstance(v, bmesh.types.BMVert)], dtype=np.float64)
    except Exception:                                       # degenerate mesh -> fall back to all verts
        pts = mesh_arrays(obj)[0]
    bm.free()
    return pts


def surface_samples(obj, hull, n, seed):
    """Hull vertices plus up to `n` random mesh vertices: the point set tested for occlusion."""
    co = mesh_arrays(obj)[0]
    if len(co) > n:
        co = co[np.random.default_rng(seed).choice(len(co), n, replace=False)]
    return np.vstack([hull, co])


# =============================================================================== #
# 3. ASSEMBLY: HOLE DETECTION & ORIENTATION                                       #
# =============================================================================== #

def insertion_radius(tube_co, depth):
    """Widest radial extent of the tube over the part that goes into a hole `depth` deep.

    Ignoring everything above the hole means a wide screw cap on a slim tube body does
    not stop the body from fitting.
    """
    part = tube_co[tube_co[:, 2] <= depth + 1e-6]
    return float(np.hypot(part[:, 0], part[:, 1]).max()) if len(part) else 0.0


def probe_hole_radius(bvh, cx, cy, z_levels, n_dirs=24):
    """Inscribed radius of a hole measured with horizontal rays from its centre.

    Exact against the mesh (unlike the grid estimate). Directions where the ray escapes
    through a side opening are ignored. Returns None if nothing was hit.
    """
    best = None
    for z in z_levels:
        for k in range(n_dirs):
            a = 2.0 * math.pi * k / n_dirs
            hit = bvh.ray_cast(Vector((cx, cy, z)), Vector((math.cos(a), math.sin(a), 0.0)), 0.5)
            if hit[0] is not None and (best is None or hit[3] < best):
                best = hit[3]
    return best


def detect_holes(co, polys, tube_co):
    """Find holes the tube body fits in a holder given its vertices (Z-up, standing on z = 0).

    A regular XY grid of rays is cast straight down onto the mesh. The 'deck' is the highest
    1 mm height band that covers >= 10 % of the hits (so small raised features are ignored).
    Cells whose ray misses, or lands >= HOLE_MIN_DEPTH below the deck, are hole cells; they
    are grouped by 4-connectivity. Groups touching the grid border are the outside world /
    side cut-outs and are discarded. Each remaining group's radius is refined with
    horizontal probe rays and compared with the tube's insertion radius for that depth.
    Returns a list of dicts: center, floor_z, radius, tube_r, fits.
    """
    lo, hi = co.min(axis=0), co.max(axis=0)
    bvh = BVHTree.FromPolygons(co.tolist(), polys, all_triangles=True)
    xs = np.arange(lo[0] + HOLE_GRID_STEP / 2, hi[0], HOLE_GRID_STEP)
    ys = np.arange(lo[1] + HOLE_GRID_STEP / 2, hi[1], HOLE_GRID_STEP)
    nx, ny = len(xs), len(ys)
    height = np.full((nx, ny), np.nan)
    down = Vector((0.0, 0.0, -1.0))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            loc = bvh.ray_cast(Vector((x, y, hi[2] + 0.01)), down)[0]
            if loc is not None:
                height[i, j] = loc.z

    hits = height[~np.isnan(height)]
    if hits.size == 0:
        return []
    counts, edges = np.histogram(hits, bins=np.arange(lo[2], hi[2] + 0.002, 0.001))
    big = np.nonzero(counts >= 0.10 * hits.size)[0]
    deck_z = float(edges[big[-1] + 1]) if big.size else float(hi[2])   # top edge of the deck band
    is_hole = np.isnan(height) | (height < deck_z - HOLE_MIN_DEPTH)

    # Flood-fill connected hole cells.
    label = np.zeros((nx, ny), dtype=np.int32)
    n_regions = 0
    for i in range(nx):
        for j in range(ny):
            if is_hole[i, j] and label[i, j] == 0:
                n_regions += 1
                label[i, j] = n_regions
                stack = [(i, j)]
                while stack:
                    a, b = stack.pop()
                    for c, d in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                        if 0 <= c < nx and 0 <= d < ny and is_hole[c, d] and label[c, d] == 0:
                            label[c, d] = n_regions
                            stack.append((c, d))

    si, sj = np.nonzero(~is_hole)
    solid_xy = np.c_[xs[si], ys[sj]]
    holes = []
    for k in range(1, n_regions + 1):
        ii, jj = np.nonzero(label == k)
        if ii.min() == 0 or jj.min() == 0 or ii.max() == nx - 1 or jj.max() == ny - 1:
            continue                                        # touches the border -> not a hole
        cx, cy = float(xs[ii].mean()), float(ys[jj].mean())
        r_grid = float(np.hypot(solid_xy[:, 0] - cx, solid_xy[:, 1] - cy).min()) if len(solid_xy) else 0.0
        r_equiv = math.sqrt(len(ii) * HOLE_GRID_STEP ** 2 / math.pi)
        hz = height[ii, jj]
        floor_z = float(lo[2]) if np.isnan(hz).mean() > 0.5 else float(np.nanmedian(hz))
        depth = deck_z - floor_z
        r_probe = probe_hole_radius(bvh, cx, cy, (deck_z - 0.006, deck_z - 0.5 * depth))
        radius = r_probe if r_probe is not None else r_grid
        tube_r = insertion_radius(tube_co, depth)
        roundish = r_equiv <= 1.3 * radius                  # rejects slots / open shell ends
        holes.append(dict(center=(cx, cy), floor_z=floor_z, radius=radius, tube_r=tube_r,
                          fits=roundish and tube_r <= radius + HOLE_FIT_TOLERANCE))
    return holes


def orient_holder(holder, tube_co):
    """Try all six axis-aligned 'which way is up' options and keep the one exposing the most
    holes the tube body fits (ties -> the configured up axis, which is tried first). Applies
    the winning rotation to the mesh, stands it on z = 0 and returns the detected holes."""
    candidates = [
        ("+Z", Matrix.Identity(4)),
        ("-Z", Matrix.Rotation(math.pi, 4, "X")),
        ("+Y", Matrix.Rotation(math.radians(90.0), 4, "X")),
        ("-Y", Matrix.Rotation(math.radians(-90.0), 4, "X")),
        ("+X", Matrix.Rotation(math.radians(-90.0), 4, "Y")),
        ("-X", Matrix.Rotation(math.radians(90.0), 4, "Y")),
    ]
    co0, polys = mesh_arrays(holder)
    best = None
    for name, R in candidates:
        co = co0 @ np.array(R.to_3x3(), dtype=np.float64).T
        lo, hi = co.min(axis=0), co.max(axis=0)
        co -= np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, lo[2]])
        holes = detect_holes(co, polys, tube_co)
        score = sum(h["fits"] for h in holes)
        radii = sorted({round(h["radius"] * 1e3, 1) for h in holes}, reverse=True)
        log(f"holder orientation {name:>2} up -> {len(holes):2d} holes (radii {radii} mm), {score} fit the tube")
        if best is None or score > best[0]:
            best = (score, name, R, holes)
    score, name, R, holes = best
    holder.data.transform(R)
    recenter_on_ground(holder)
    log(f"holder: using {name} as up ({score} tube holes)")
    for h in holes:
        if h["fits"]:
            log("  hole at ({:+.1f}, {:+.1f}) mm  r={:.1f} mm  floor z={:.1f} mm  tube body r={:.1f} mm".format(
                h["center"][0] * 1e3, h["center"][1] * 1e3, h["radius"] * 1e3, h["floor_z"] * 1e3, h["tube_r"] * 1e3))
    return holes


def orient_tube(tube):
    """Stand the tube upright (longest bbox axis -> Z) with the right end down.

    Rule 1: the narrower end goes down (conical / rounded bottoms, wide screw caps up).
    Rule 2 (ends equally wide): an open end goes up - a ray cast down the axis from above
            falls into an open tube and hits the inner floor near z ~ 0, but hits a closed
            top near z ~ height.
    Returns the oriented vertex array (bottom at z = 0, axis on Z) and the bbox radius.
    """
    lo, hi = mesh_bounds(tube)
    dims = hi - lo
    longest = int(np.argmax(dims))
    to_z = {0: Matrix.Rotation(math.radians(-90.0), 4, "Y"),
            1: Matrix.Rotation(math.radians(90.0), 4, "X"),
            2: Matrix.Identity(4)}[longest]
    tube.data.transform(to_z)
    recenter_on_ground(tube)
    radius = 0.5 * max(dims[(longest + 1) % 3], dims[(longest + 2) % 3])

    co, polys = mesh_arrays(tube)
    height = co[:, 2].max()
    band = 0.12 * height
    radial = np.hypot(co[:, 0], co[:, 1])
    r_bottom, r_top = radial[co[:, 2] <= band].mean(), radial[co[:, 2] >= height - band].mean()
    flip, why = False, ""
    if abs(r_bottom - r_top) > 0.05 * max(r_bottom, r_top):
        flip, why = r_bottom > r_top, "narrower end goes down"
    else:
        bvh = BVHTree.FromPolygons(co.tolist(), polys, all_triangles=True)
        hit = bvh.ray_cast(Vector((0.25 * radius, 0.0, height + 0.01)), Vector((0.0, 0.0, -1.0)))[0]
        flip, why = hit is not None and hit.z > 0.5 * height, "open end goes up"
    if flip:
        tube.data.transform(Matrix.Rotation(math.pi, 4, "X"))
        recenter_on_ground(tube)
        co = mesh_arrays(tube)[0]
        log(f"tube: flipped ({why})")
    log(f"tube: bbox radius {radius * 1e3:.1f} mm, height {height * 1e3:.1f} mm, "
        f"end radii bottom {min(r_bottom, r_top) * 1e3:.1f} / top {max(r_bottom, r_top) * 1e3:.1f} mm")
    return co, radius


# =============================================================================== #
# 4. MATERIALS, BENCH, WORLD, CAMERA, RENDER SETTINGS                             #
# =============================================================================== #

def make_material(name, kind):
    """'glass': clear Principled BSDF (transmission 1, low roughness). 'plastic': matte."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    if kind == "glass":
        set_input(bsdf, ["Base Color"], (1.0, 1.0, 1.0, 1.0))
        set_input(bsdf, ["Roughness"], 0.03)
        set_input(bsdf, ["IOR"], 1.5)
        set_input(bsdf, ["Transmission Weight", "Transmission"], 1.0)   # 4.x name, then 3.x
        set_input(bsdf, ["Specular IOR Level", "Specular"], 0.5)
    else:
        set_input(bsdf, ["Base Color"], (0.8, 0.1, 0.1, 1.0))
        set_input(bsdf, ["Roughness"], 0.55)
        set_input(bsdf, ["Specular IOR Level", "Specular"], 0.4)
    return mat, bsdf


def make_bench(S):
    """Large ground plane with a procedural material: one shared Mapping node feeds a Wave
    (wood grain), Noise, Voronoi and Brick texture plus an optional Image texture. Each
    frame one of them is wired into a 3-stop ColorRamp -> Base Color and into a Bump node,
    and its parameters are randomised, so no two floors look alike."""
    me = bpy.data.meshes.new("Bench")
    s = 2.0
    me.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
    me.update()
    plane = bpy.data.objects.new("Bench", me)
    plane.location.z = -0.0002                              # avoid z-fighting with object bases
    S.scene.collection.objects.link(plane)
    mat = bpy.data.materials.new("Bench_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    coords = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    nt.links.new(coords.outputs["Generated"], mapping.inputs["Vector"])   # no UVs needed
    wave, noise, voronoi, brick, image = (nt.nodes.new(t) for t in (
        "ShaderNodeTexWave", "ShaderNodeTexNoise", "ShaderNodeTexVoronoi", "ShaderNodeTexBrick", "ShaderNodeTexImage"))
    for node in (wave, noise, voronoi, brick, image):
        nt.links.new(mapping.outputs["Vector"], node.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements.new(0.5)                       # dark / mid / light stops
    bump = nt.nodes.new("ShaderNodeBump")
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    me.materials.append(mat)
    S.bench = SimpleNamespace(obj=plane, bsdf=bsdf, mapping=mapping, wave=wave, noise=noise, voronoi=voronoi,
                              brick=brick, image=image, ramp=ramp, bump=bump, links=nt.links)


def make_world(S):
    """World background: a mottled two-colour noise gradient by default (never a flat sterile
    colour), or an HDRI/image when --hdri-dir is given."""
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    S.scene.world = world
    nt = world.node_tree
    bg = nt.nodes["Background"]
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements.new(0.5)
    mapping = nt.nodes.new("ShaderNodeMapping")
    coords = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(coords.outputs["Generated"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    S.world = SimpleNamespace(bg=bg, env=env, noise=noise, ramp=ramp, mapping=mapping, links=nt.links)


def make_camera(S):
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.sensor_fit = "AUTO"
    cam_data.clip_start = 0.01
    cam_data.clip_end = 50.0
    cam = bpy.data.objects.new("Camera", cam_data)
    S.scene.collection.objects.link(cam)
    S.scene.camera = cam
    S.cam = cam


def enable_gpu(scene):
    """Cycles only enables GPUs through the preferences, which must be done explicitly in -b mode."""
    try:
        cprefs = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        scene.cycles.device = "CPU"
        return "CPU"
    for backend in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
        try:
            cprefs.compute_device_type = backend
        except TypeError:                                   # backend not compiled in / unsupported
            continue
        (cprefs.refresh_devices if hasattr(cprefs, "refresh_devices") else cprefs.get_devices)()
        gpus = [d for d in cprefs.devices if d.type == backend]
        if gpus:
            for d in cprefs.devices:
                d.use = d.type == backend
            scene.cycles.device = "GPU"
            return f"GPU/{backend}: " + ", ".join(d.name for d in gpus)
    cprefs.compute_device_type = "NONE"
    scene.cycles.device = "CPU"
    return "CPU"


def setup_render(S, args):
    r = S.scene.render
    r.resolution_x = r.resolution_y = args.resolution
    r.resolution_percentage = 100
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.image_settings.compression = 40
    r.film_transparent = False
    r.use_persistent_data = True                            # keep BVH between frames (geometry is static)
    try:
        S.scene.view_settings.view_transform = "AgX"        # Blender 4.x default filmic curve
    except TypeError:
        S.scene.view_settings.view_transform = "Filmic"
    if args.engine == "cycles":
        r.engine = "CYCLES"
        cy = S.scene.cycles
        cy.samples = args.samples
        cy.use_adaptive_sampling = True
        cy.use_denoising = True
        cy.caustics_reflective = False                      # noisy with glass + point lights
        cy.caustics_refractive = False
        cy.blur_glossy = 1.0
        cy.max_bounces = 12
        cy.transmission_bounces = 12
        cy.transparent_max_bounces = 8
        log(f"render device: {enable_gpu(S.scene)}")
    else:
        for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
            try:
                r.engine = engine
                break
            except TypeError:
                continue
        S.scene.eevee.taa_render_samples = args.samples
        if hasattr(S.scene.eevee, "use_raytracing"):
            S.scene.eevee.use_raytracing = True             # needed for refraction in EEVEE Next


# =============================================================================== #
# 5. SCENE BUILD                                                                  #
# =============================================================================== #

def build_scene(args, stl_dir):
    S = SimpleNamespace(lights=[], image_cache={}, bsdf={}, label_mode=args.label_mode)
    reset_scene()
    S.scene = bpy.context.scene

    # ---- import -----------------------------------------------------------------
    holder, tube, loose = None, None, []
    for spec in OBJECT_SPECS:
        path = stl_dir / spec["file"]
        if not path.exists():
            raise FileNotFoundError(path)
        obj = import_stl(path, spec.get("pre_rotate_deg", (0.0, 0.0, 0.0)))
        obj.name = spec["name"]
        obj["class_id"] = spec["class_id"]
        obj["material_kind"] = spec["material"]
        mat, bsdf = make_material(f"{spec['name']}_mat", spec["material"])
        obj.data.materials.append(mat)
        S.bsdf[obj.data.name] = bsdf                        # keyed by mesh so instances share it
        log(f"imported {spec['file']:<22} -> class {spec['class_id']} '{spec['name']}' "
            f"({len(obj.data.vertices)} verts, {spec['material']})")
        if spec["role"] == "holder":
            holder = obj
        elif spec["role"] == "tube":
            tube = obj
        else:
            loose.append(obj)
    if holder is None or tube is None:
        raise RuntimeError("OBJECT_SPECS needs exactly one 'holder' and one 'tube' role")

    # ---- orientation + assembly ---------------------------------------------------
    tube_co, S.tube_radius = orient_tube(tube)
    holes = orient_holder(holder, tube_co)
    for obj in loose:
        recenter_on_ground(obj)
    S.holes = [h for h in holes if h["fits"]]
    if S.holes:
        S.tube_instances = [tube]
        for i in range(1, len(S.holes)):                    # linked duplicates share the mesh/material
            inst = tube.copy()
            inst.data = tube.data
            inst.name = f"{tube.name}.{i:03d}"
            S.scene.collection.objects.link(inst)
            S.tube_instances.append(inst)
    else:
        log("WARNING: no hole fits the tube body - the tube will stand loose on the bench instead")
        S.tube_instances = []
        loose.append(tube)
    S.holder, S.loose = holder, loose
    S.holder_radius = footprint_radius(holder)
    S.loose_radius = {o.name: footprint_radius(o) for o in loose}
    S.labelled = [holder] + S.tube_instances + loose
    S.hulls = {o.data.name: convex_hull_vertices(o) for o in S.labelled}
    S.samples = {o.data.name: surface_samples(o, S.hulls[o.data.name], VISIBILITY_SAMPLES, args.seed)
                 for o in S.labelled}

    # ---- environment --------------------------------------------------------------
    make_bench(S)
    make_world(S)
    make_camera(S)
    setup_render(S, args)
    S.textures = list_images(args.textures_dir, IMAGE_EXTS)
    S.hdris = list_images(args.hdri_dir, HDRI_EXTS)
    log(f"bench textures: {len(S.textures)}, HDRIs: {len(S.hdris)} (procedural backgrounds are used when none)")
    S.projector = Projector(S.scene, S.cam)
    return S


# =============================================================================== #
# 6. PER-FRAME DOMAIN RANDOMISATION                                               #
# =============================================================================== #

def random_light_color(rng):
    """Mostly near-white with random colour temperature; sometimes a strongly tinted lamp."""
    sat = rng.uniform(0.0, 0.12) if rng.random() < 0.8 else rng.uniform(0.25, 0.6)
    return colorsys.hsv_to_rgb(rng.random(), sat, 1.0)


def hsv(h, s, v):
    """HSV -> linear RGB with the value capped so nothing in the background is near-white."""
    return colorsys.hsv_to_rgb(h % 1.0, min(max(s, 0.0), 1.0), min(max(v, 0.0), MAX_ALBEDO))


def random_surface_color(rng):
    sat = rng.uniform(0.0, 0.5) if rng.random() < 0.7 else rng.uniform(0.5, 1.0)
    return hsv(rng.random(), sat, rng.uniform(0.15, MAX_ALBEDO))


def random_plastic_color(rng):
    return rng.choice(PLASTIC_PALETTE) if rng.random() < 0.8 else random_surface_color(rng)


def relink(links, socket_in, socket_out):
    """Make `socket_out` the only thing driving `socket_in`."""
    for link in list(socket_in.links):
        links.remove(link)
    links.new(socket_out, socket_in)


def unlink(links, socket_in):
    for link in list(socket_in.links):
        links.remove(link)


def set_ramp(ramp, colors, positions=(0.0, 0.5, 1.0), interpolation="LINEAR"):
    """Program the 3-stop ColorRamp (colours are RGB tuples)."""
    ramp.color_ramp.interpolation = interpolation
    for el, pos, col in zip(ramp.color_ramp.elements, positions, colors):
        el.position = pos
        el.color = (*col, 1.0)


def set_hidden(obj, hidden):
    """Hide from the render AND from ray casts (hide_render alone still occludes rays)."""
    obj.hide_render = hidden
    obj.hide_viewport = hidden


def clear_lights(S):
    for ob in S.lights:
        data = ob.data
        bpy.data.objects.remove(ob, do_unlink=True)
        bpy.data.lights.remove(data)
    S.lights = []


def spawn_lights(S, rng):
    """1-3 point lights on a dome above the bench.

    One power budget is drawn per frame and split between the lights with random weights,
    then each light is scaled by distance^2 so its contribution at the bench matches the
    budget regardless of where it sits. The total illumination is therefore bounded by
    LIGHT_TOTAL_POWER_RANGE_W no matter how many lights there are.
    """
    clear_lights(S)
    n = rng.randint(*NUM_LIGHTS_RANGE)
    budget = rng.uniform(*LIGHT_TOTAL_POWER_RANGE_W)
    weights = [rng.uniform(0.3, 1.0) for _ in range(n)]
    for i, w in enumerate(weights):
        elev = math.radians(rng.uniform(*LIGHT_ELEV_RANGE_DEG))
        azim = rng.uniform(0.0, 2.0 * math.pi)
        dist = rng.uniform(*LIGHT_DIST_RANGE)
        ld = bpy.data.lights.new(f"RandLight{i}", "POINT")
        ld.energy = budget * (w / sum(weights)) * (dist / 0.8) ** 2
        ld.color = random_light_color(rng)
        ld.shadow_soft_size = rng.uniform(*LIGHT_RADIUS_RANGE)
        ob = bpy.data.objects.new(ld.name, ld)
        ob.location = (dist * math.cos(elev) * math.cos(azim),
                       dist * math.cos(elev) * math.sin(azim),
                       dist * math.sin(elev))
        S.scene.collection.objects.link(ob)
        S.lights.append(ob)


def randomize_bench(S, rng):
    """Give the bench a fresh procedural (or image) texture. Never a flat colour.

    Texture coordinates are 'Generated' (0..1 across the 4 m plane), so a texture scale of
    40 means ~40 repetitions over the plane, i.e. a ~10 cm feature size.
    """
    b = S.bench
    families = ["wood", "metal", "noise", "tiles", "grain"] + (["image"] if S.textures else [])
    weights = [2.0, 1.5, 2.0, 1.5, 1.5] + ([3.0] if S.textures else [])
    family = rng.choices(families, weights)[0]
    scale, metallic, rough = (1.0, 1.0, 1.0), 0.0, rng.uniform(0.3, 0.8)
    color_out, height_out = b.ramp.outputs["Color"], None

    if family == "wood":                                    # planks: distorted bands in two tones
        b.wave.wave_type, b.wave.bands_direction = "BANDS", "X"
        b.wave.wave_profile = rng.choice(["SIN", "SAW"])
        set_input(b.wave, ["Scale"], rng.uniform(15.0, 60.0))
        set_input(b.wave, ["Distortion"], rng.uniform(3.0, 12.0))
        set_input(b.wave, ["Detail"], rng.uniform(2.0, 8.0))
        set_input(b.wave, ["Detail Scale"], rng.uniform(1.0, 3.0))
        set_input(b.wave, ["Phase Offset"], rng.uniform(0.0, 100.0))
        hue = rng.uniform(0.04, 0.11)
        sat = rng.uniform(0.3, 0.75) if rng.random() < 0.8 else rng.uniform(0.0, 0.15)   # bleached/grey wood
        dark, light = rng.uniform(0.05, 0.3), rng.uniform(0.3, 0.75)
        set_ramp(b.ramp, [hsv(hue, sat, dark), hsv(hue + rng.uniform(-0.02, 0.02), sat * 0.8, (dark + light) / 2),
                          hsv(hue, sat, light)])
        height_out, scale, rough = b.wave.outputs["Fac"], (1.0, rng.uniform(1.0, 3.0), 1.0), rng.uniform(0.3, 0.65)
    elif family == "metal":                                 # brushed metal: noise stretched along one axis
        set_input(b.noise, ["Scale"], rng.uniform(5.0, 30.0))
        set_input(b.noise, ["Detail"], rng.uniform(10.0, 16.0))
        set_input(b.noise, ["Roughness"], rng.uniform(0.6, 0.9))
        set_input(b.noise, ["Distortion"], 0.0)
        hue, tint, grey = rng.random(), rng.uniform(0.0, 0.08), rng.uniform(0.3, 0.7)
        set_ramp(b.ramp, [hsv(hue, tint, grey - 0.12), hsv(hue, tint, grey), hsv(hue, tint, grey + 0.12)])
        height_out, scale = b.noise.outputs["Fac"], (1.0, rng.uniform(30.0, 80.0), 1.0)
        metallic, rough = rng.uniform(0.7, 1.0), rng.uniform(0.15, 0.45)
    elif family == "noise":                                 # speckle / terrazzo / mottled paint
        if rng.random() < 0.5:
            set_input(b.noise, ["Scale"], rng.uniform(10.0, 120.0))
            set_input(b.noise, ["Detail"], rng.uniform(5.0, 15.0))
            set_input(b.noise, ["Roughness"], rng.uniform(0.4, 0.9))
            set_input(b.noise, ["Distortion"], rng.uniform(0.0, 3.0))
            height_out = b.noise.outputs["Fac"]
        else:
            b.voronoi.feature = rng.choice(["F1", "SMOOTH_F1", "DISTANCE_TO_EDGE"])
            set_input(b.voronoi, ["Scale"], rng.uniform(20.0, 150.0))
            set_input(b.voronoi, ["Randomness"], rng.uniform(0.6, 1.0))
            height_out = b.voronoi.outputs["Distance"]
        set_ramp(b.ramp, [random_surface_color(rng) for _ in range(3)],
                 interpolation=rng.choice(["LINEAR", "CONSTANT", "EASE"]))
    elif family == "tiles":                                 # floor tiles / bricks with mortar lines
        set_input(b.brick, ["Scale"], rng.uniform(5.0, 25.0))
        set_input(b.brick, ["Mortar Size"], rng.uniform(0.01, 0.05))
        set_input(b.brick, ["Mortar Smooth"], rng.uniform(0.0, 0.5))
        set_input(b.brick, ["Bias"], rng.uniform(-0.3, 0.3))
        set_input(b.brick, ["Brick Width"], rng.uniform(0.3, 0.8))
        set_input(b.brick, ["Row Height"], rng.uniform(0.15, 0.5))
        b.brick.offset, b.brick.squash = rng.choice([0.0, 0.5]), rng.choice([1.0, 1.0, 2.0])
        c = random_surface_color(rng)
        set_input(b.brick, ["Color1"], (*c, 1.0))
        set_input(b.brick, ["Color2"], (*[min(v * rng.uniform(0.7, 1.3), MAX_ALBEDO) for v in c], 1.0))
        set_input(b.brick, ["Mortar"], (*hsv(0.0, 0.0, rng.uniform(0.05, 0.3)), 1.0))
        color_out, height_out, rough = b.brick.outputs["Color"], b.brick.outputs["Fac"], rng.uniform(0.2, 0.6)
    elif family == "grain":                                 # near-plain bench with fine grain
        set_input(b.noise, ["Scale"], rng.uniform(60.0, 400.0))
        set_input(b.noise, ["Detail"], rng.uniform(3.0, 8.0))
        set_input(b.noise, ["Roughness"], rng.uniform(0.4, 0.8))
        set_input(b.noise, ["Distortion"], 0.0)
        c, d = random_surface_color(rng), rng.uniform(0.05, 0.2)
        set_ramp(b.ramp, [[v * (1 - d) for v in c], c, [min(v * (1 + d), MAX_ALBEDO) for v in c]])
        height_out = b.noise.outputs["Fac"]
    else:                                                   # photo texture from --textures-dir
        b.image.image = load_image(S, rng.choice(S.textures))
        s = rng.uniform(1.0, 6.0)
        set_input(b.noise, ["Scale"], rng.uniform(100.0, 300.0))
        color_out, height_out, scale = b.image.outputs["Color"], b.noise.outputs["Fac"], (s, s, 1.0)

    b.mapping.inputs["Scale"].default_value = scale
    b.mapping.inputs["Rotation"].default_value = (0.0, 0.0, rng.uniform(0.0, 2.0 * math.pi))
    relink(b.links, b.ramp.inputs["Fac"], height_out)
    relink(b.links, b.bump.inputs["Height"], height_out)
    relink(b.links, b.bsdf.inputs["Base Color"], color_out)
    set_input(b.bump, ["Strength"], rng.uniform(0.05, 0.35))
    set_input(b.bump, ["Distance"], rng.uniform(0.01, 0.1))
    set_input(b.bsdf, ["Roughness"], rough)
    set_input(b.bsdf, ["Metallic"], metallic)


def randomize_world(S, rng):
    w = S.world
    unlink(w.links, w.bg.inputs["Color"])
    if S.hdris and rng.random() < 0.6:
        w.env.image = load_image(S, rng.choice(S.hdris))
        w.mapping.inputs["Rotation"].default_value = (0.0, 0.0, rng.uniform(0.0, 2.0 * math.pi))
        w.links.new(w.env.outputs["Color"], w.bg.inputs["Color"])
        w.bg.inputs["Strength"].default_value = rng.uniform(0.3, 1.0)
    else:                                                   # mottled two-colour gradient, never flat
        set_input(w.noise, ["Scale"], rng.uniform(1.0, 6.0))
        set_input(w.noise, ["Detail"], rng.uniform(2.0, 6.0))
        set_input(w.noise, ["Distortion"], rng.uniform(0.0, 1.0))
        w.mapping.inputs["Rotation"].default_value = (0.0, 0.0, rng.uniform(0.0, 2.0 * math.pi))
        c1, c2 = random_surface_color(rng), random_surface_color(rng)
        set_ramp(w.ramp, [c1, [(a + b) / 2 for a, b in zip(c1, c2)], c2])
        w.links.new(w.ramp.outputs["Color"], w.bg.inputs["Color"])
        w.bg.inputs["Strength"].default_value = rng.uniform(*WORLD_STRENGTH_RANGE)


def randomize_materials(S, rng):
    seen = set()
    for obj in S.labelled:
        key = obj.data.name
        if key in seen:
            continue
        seen.add(key)
        bsdf = S.bsdf[key]
        if obj["material_kind"] == "plastic":
            set_input(bsdf, ["Base Color"], (*random_plastic_color(rng), 1.0))
            set_input(bsdf, ["Roughness"], rng.uniform(0.35, 0.75))
        else:                                               # glass: clear or a faint tint
            tint = [rng.uniform(0.9, 1.0) for _ in range(3)] if rng.random() < 0.3 else [1.0, 1.0, 1.0]
            set_input(bsdf, ["Base Color"], (*tint, 1.0))
            set_input(bsdf, ["Roughness"], rng.uniform(0.0, 0.1))


def place_objects(S, rng):
    """Yaw the holder, fill a random subset of its holes with tubes, scatter loose objects."""
    yaw = rng.uniform(0.0, 2.0 * math.pi)
    S.holder.rotation_euler = (0.0, 0.0, yaw)
    S.holder.location = (0.0, 0.0, 0.0)
    yaw_m = Matrix.Rotation(yaw, 4, "Z")

    if S.tube_instances:
        chosen = [h for h in S.holes if rng.random() < HOLE_FILL_PROB] or [rng.choice(S.holes)]
        for inst in S.tube_instances:
            set_hidden(inst, True)
        for inst, hole in zip(S.tube_instances, chosen):
            clearance = max(hole["radius"] - hole["tube_r"], 0.0) * 0.7   # keep the tube off the wall
            ang, d = rng.uniform(0.0, 2.0 * math.pi), rng.uniform(0.0, clearance)
            local = Vector((hole["center"][0] + d * math.cos(ang),
                            hole["center"][1] + d * math.sin(ang),
                            hole["floor_z"]))                            # rests on the hole floor
            inst.location = yaw_m @ local
            inst.rotation_euler = (0.0, 0.0, rng.uniform(0.0, 2.0 * math.pi))
            set_hidden(inst, False)

    placed = [(0.0, 0.0, S.holder_radius)]
    for obj in S.loose:
        r = S.loose_radius[obj.name]
        if rng.random() > LOOSE_PRESENT_PROB:
            set_hidden(obj, True)
            continue
        x = y = 0.0
        for _ in range(60):                                 # rejection sampling against overlaps
            ang = rng.uniform(0.0, 2.0 * math.pi)
            d = rng.uniform(S.holder_radius + r + 0.01, S.holder_radius + r + LOOSE_MAX_DIST)
            x, y = d * math.cos(ang), d * math.sin(ang)
            if all(math.hypot(x - px, y - py) > r + pr + 0.01 for px, py, pr in placed):
                break
        placed.append((x, y, r))
        obj.location = (x, y, 0.0)
        obj.rotation_euler = (0.0, 0.0, rng.uniform(0.0, 2.0 * math.pi))
        set_hidden(obj, False)


def place_camera(S, rng):
    """Random point on a dome around the holder, looking at (a jittered) holder centre."""
    j = CAM_TARGET_JITTER
    target = Vector((rng.uniform(-j, j), rng.uniform(-j, j), rng.uniform(0.0, 0.06)))
    elev = math.radians(rng.uniform(*CAM_ELEV_RANGE_DEG))
    azim = rng.uniform(0.0, 2.0 * math.pi)
    dist = rng.uniform(*CAM_DIST_RANGE)
    S.cam.location = target + Vector((dist * math.cos(elev) * math.cos(azim),
                                      dist * math.cos(elev) * math.sin(azim),
                                      dist * math.sin(elev)))
    look = (target - S.cam.location).normalized()
    roll = Quaternion((0.0, 0.0, 1.0), math.radians(rng.uniform(-CAM_ROLL_MAX_DEG, CAM_ROLL_MAX_DEG)))
    S.cam.rotation_euler = (look.to_track_quat("-Z", "Y") @ roll).to_euler()   # roll about the view axis
    S.cam.data.lens = rng.uniform(*CAM_LENS_RANGE_MM)
    S.scene.view_settings.exposure = rng.uniform(*EXPOSURE_RANGE_EV)


def randomize_frame(S, rng):
    place_objects(S, rng)
    randomize_materials(S, rng)
    randomize_bench(S, rng)
    randomize_world(S, rng)
    spawn_lights(S, rng)
    place_camera(S, rng)
    bpy.context.view_layer.update()                         # make every matrix_world current


# =============================================================================== #
# 7. YOLOv8-OBB LABELS                                                            #
# =============================================================================== #

class Projector:
    """Vectorised twin of bpy_extras.object_utils.world_to_camera_view for perspective cameras.

    Returns normalised image coordinates with x to the right and y DOWN (image convention,
    which is what YOLO expects) plus the depth of every point along the view axis.
    """

    def __init__(self, scene, cam):
        self.scene, self.cam = scene, cam

    def refresh(self):
        f = self.cam.data.view_frame(scene=self.scene)      # sensor frame corners in camera space
        self.depth = -f[0].z
        self.x0, self.x1, self.y0, self.y1 = f[2].x, f[1].x, f[1].y, f[0].y
        self.view = np.array(self.cam.matrix_world.normalized().inverted(), dtype=np.float64)

    def project(self, pts_world):
        local = np.c_[pts_world, np.ones(len(pts_world))] @ self.view.T
        depth = -local[:, 2]                                # Blender cameras look down -Z
        with np.errstate(divide="ignore", invalid="ignore"):
            s = depth / self.depth                          # frame grows linearly with depth
            u = (local[:, 0] - self.x0 * s) / ((self.x1 - self.x0) * s)
            v = (local[:, 1] - self.y0 * s) / ((self.y1 - self.y0) * s)
        return np.c_[u, 1.0 - v], depth

    def self_test(self, pts_world):
        """Max deviation from Blender's reference implementation (should be ~1e-7)."""
        uv, _ = self.project(pts_world)
        ref = np.array([world_to_camera_view(self.scene, self.cam, Vector(p))[:2] for p in pts_world])
        ref[:, 1] = 1.0 - ref[:, 1]
        return float(np.abs(uv - ref).max())


def visible_points(S, obj, pts_world, depsgraph):
    """Keep the points of `obj` whose line of sight to the camera is not blocked by an opaque
    object. Glass objects are stepped through (up to 8 surfaces); hitting `obj` itself counts
    as visible because that ray still lands inside the object's visible silhouette."""
    cam_pos = S.cam.matrix_world.translation
    keep = []
    for p in pts_world:
        target = Vector(p)
        origin = cam_pos.copy()
        direction = target - origin
        remaining = direction.length - 1e-4                 # stop just short of the point itself
        direction.normalize()
        ok = False
        for _ in range(8):
            hit, loc, _n, _i, hit_obj, _m = S.scene.ray_cast(depsgraph, origin, direction, distance=remaining)
            if not hit or hit_obj.name == obj.name:
                ok = True
                break
            if hit_obj.get("material_kind") == "glass":    # see-through: continue behind it
                step = (loc - origin).length + 1e-4
                origin, remaining = loc + direction * 1e-4, remaining - step
                continue
            break                                           # opaque occluder
        if ok:
            keep.append(p)
    return np.array(keep) if keep else np.empty((0, 3))


def polygon_area(poly):
    a = 0.0
    for i in range(len(poly)):
        x0, y0 = poly[i - 1]
        x1, y1 = poly[i]
        a += x0 * y1 - x1 * y0
    return abs(a) * 0.5


def clip_to_unit_square(poly):
    """Sutherland-Hodgman clip of a convex polygon against the image rectangle [0,1]x[0,1]."""
    for axis in (0, 1):
        for bound, keep_greater in ((0.0, True), (1.0, False)):
            if not poly:
                return []
            out = []
            for i in range(len(poly)):
                p, q = poly[i - 1], poly[i]
                p_in = p[axis] >= bound if keep_greater else p[axis] <= bound
                q_in = q[axis] >= bound if keep_greater else q[axis] <= bound
                if p_in != q_in:                            # edge crosses the boundary
                    t = (bound - p[axis]) / (q[axis] - p[axis])
                    out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
                if q_in:
                    out.append(q)
            poly = out
    return poly


def min_area_rect(poly):
    """Smallest-area rectangle enclosing a convex polygon.

    One side of the optimal rectangle is always collinear with a polygon edge, so each edge
    direction is tried in turn. Returns (4 corners in order, area).
    """
    pts = np.asarray(poly, dtype=np.float64)
    best = None
    for i in range(len(pts)):
        edge = pts[(i + 1) % len(pts)] - pts[i]
        n = math.hypot(*edge)
        if n < 1e-12:
            continue
        u = edge / n
        v = np.array([-u[1], u[0]])
        pu, pv = pts @ u, pts @ v
        area = (pu.max() - pu.min()) * (pv.max() - pv.min())
        if best is None or area < best[0]:
            best = (area, u, v, pu.min(), pu.max(), pv.min(), pv.max())
    if best is None:
        return None, 0.0
    area, u, v, u0, u1, v0, v1 = best
    corners = np.array([u0 * u + v0 * v, u1 * u + v0 * v, u1 * u + v1 * v, u0 * u + v1 * v])
    return corners, area


def obb_for_object(S, obj, depsgraph):
    """Tight oriented box of `obj` in normalised image coordinates, or None if (almost) unseen."""
    M = np.array(obj.matrix_world, dtype=np.float64)
    if S.label_mode == "visible":
        pts = S.samples[obj.data.name] @ M[:3, :3].T + M[:3, 3]
        pts = visible_points(S, obj, pts, depsgraph)
    else:
        pts = S.hulls[obj.data.name] @ M[:3, :3].T + M[:3, 3]
    if len(pts) < 3:
        return None
    uv, depth = S.projector.project(pts)
    uv = uv[depth > 1e-6]                                   # drop anything behind the camera
    if len(uv) < 3:
        return None
    hull = [tuple(uv[i]) for i in convex_hull_2d(uv.tolist())]
    if len(hull) < 3:
        return None
    full_area = polygon_area(hull)
    inside = clip_to_unit_square(hull)                      # only the part inside the frame
    if len(inside) < 3 or full_area <= 0.0 or polygon_area(inside) / full_area < MIN_VISIBLE_FRACTION:
        return None
    corners, area = min_area_rect(inside)
    if corners is None or area < MIN_BOX_AREA:
        return None
    return np.clip(corners, 0.0, 1.0)                       # Ultralytics rejects coords outside [0,1]


def compute_labels(S):
    S.projector.refresh()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    lines = []
    for obj in S.labelled:
        if obj.hide_render:
            continue
        corners = obb_for_object(S, obj, depsgraph)
        if corners is not None:
            lines.append(f"{int(obj['class_id'])} " + " ".join(f"{c:.6f}" for c in corners.ravel()))
    return lines


def frame_stats(path):
    """(mean brightness, blown-out fraction) of a saved render, in display space 0-1.
    Image.pixels of an 8-bit PNG are the sRGB-encoded values, i.e. what a viewer shows."""
    img = bpy.data.images.load(str(path), check_existing=False)
    try:
        px = np.empty(img.size[0] * img.size[1] * img.channels, dtype=np.float32)
        img.pixels.foreach_get(px)
        rgb = px.reshape(-1, img.channels)[:, :3]
        return float(rgb.mean()), float((rgb.min(axis=1) > 0.95).mean())
    finally:
        bpy.data.images.remove(img)


def render_with_exposure_guard(S, img_path, stem):
    """Render, then act like a camera's auto-exposure: if the frame is too bright, too dark
    or has too many blown-out pixels, step the exposure and re-render (a few times at most).
    Geometry is untouched, so the labels computed beforehand stay valid."""
    S.scene.render.filepath = str(img_path)
    bpy.ops.render.render(write_still=True)
    lo, hi = TARGET_MEAN_RANGE
    for _ in range(MAX_EXPOSURE_RETRIES):
        mean, clipped = frame_stats(img_path)
        if lo <= mean <= hi and clipped <= MAX_CLIPPED_FRACTION:
            break
        # Display values follow roughly a gamma-2.2 curve, so a brightness ratio maps to
        # about 2.2x as many EV stops; clamp to sensible steps.
        step = max(0.5, min(2.0, abs(math.log2(max(mean, 0.02) / ((lo + hi) / 2))) * 2.2))
        if mean > hi or clipped > MAX_CLIPPED_FRACTION:
            step = -step
        S.scene.view_settings.exposure += step
        log(f"frame {stem}: mean brightness {mean:.2f}, {clipped:.0%} blown out -> re-rendering at "
            f"{S.scene.view_settings.exposure:+.1f} EV")
        bpy.ops.render.render(write_still=True)


def write_data_yaml(out_dir):
    names = {s["class_id"]: s["name"] for s in OBJECT_SPECS}
    text = (f"path: {out_dir.as_posix()}\n"
            "train: images/train\n"
            "val: images/train   # TODO: carve out a validation split (e.g. scripts/yolo_dataset_splitter.py)\n"
            "\nnames:\n" + "".join(f"  {k}: {v}\n" for k, v in sorted(names.items())))
    (out_dir / "data.yaml").write_text(text)


# =============================================================================== #
# 8. MAIN LOOP                                                                    #
# =============================================================================== #

def main():
    args = parse_args()
    # Absolute paths: Blender resolves relative render paths against its own base, not the cwd.
    stl_dir, out_dir = Path(args.stl_dir).resolve(), Path(args.out_dir).resolve()
    if (out_dir / "split_manifest.json").exists():
        raise RuntimeError("Output already contains a frozen train/val/test split; choose a new --out-dir.")
    images_dir, labels_dir = out_dir / "images" / "train", out_dir / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    log(f"Blender {bpy.app.version_string} | STLs: {stl_dir} | output: {out_dir} | labels: {args.label_mode}")

    S = build_scene(args, stl_dir)
    write_data_yaml(out_dir)

    # Sanity-check the vectorised projector against Blender's reference once per run.
    randomize_frame(S, random.Random(args.seed))
    S.projector.refresh()
    M = np.array(S.holder.matrix_world, dtype=np.float64)
    err = S.projector.self_test(S.hulls[S.holder.data.name][:16] @ M[:3, :3].T + M[:3, 3])
    log(f"projection self-test: max deviation {err:.2e}")
    if err > 1e-5:
        raise RuntimeError("projection mismatch vs. world_to_camera_view - aborting before writing labels")

    t0, done = time.time(), 0
    first, last = args.start_index, args.start_index + args.num_images
    for idx in range(first, last):
        stem = f"{idx:06d}"
        img_path, lbl_path = images_dir / f"{stem}.png", labels_dir / f"{stem}.txt"
        if img_path.exists() and lbl_path.exists() and not args.overwrite:
            continue
        rng = random.Random(args.seed * 1_000_003 + idx)   # per-frame seed -> reproducible & resumable
        randomize_frame(S, rng)
        lines = compute_labels(S)
        render_with_exposure_guard(S, img_path, stem)
        lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        done += 1
        per = (time.time() - t0) / done
        eta = per * (last - idx - 1)
        log(f"frame {stem} | {len(lines):2d} labels | {per:5.1f} s/frame | ETA {int(eta // 3600):02d}:"
            f"{int(eta % 3600 // 60):02d}:{int(eta % 60):02d}")

    log(f"done: {done} frames rendered in {time.time() - t0:.0f} s -> {out_dir}")


if __name__ == "__main__":
    main()
