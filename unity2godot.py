#!/usr/bin/env python3
"""
unity2godot.py - Convert a .unitypackage into Godot 4 ready assets.

What it does
------------
* Unpacks the .unitypackage (a gzipped tar of GUID folders) and restores the
  original folder/file names (Assets/Foo/bar.png -> <out>/Foo/bar.png).
* Copies textures, models (fbx/obj/dae/gltf/blend) and audio.
* Converts formats Godot can't import (.tif/.tiff/.psd/.gif) to .png.
* Converts Unity materials (.mat) to Godot StandardMaterial3D (.tres):
    - Built-in "Standard", URP Lit and HDRP Lit style property names
    - albedo / normal / metallic / occlusion / emission / alpha-clip / cull
    - Unity smoothness (alpha channel) is baked to a Godot roughness texture
      (requires Pillow)
* Converts Unity Terrain Layers (.terrainlayer) to tiling, triplanar
  StandardMaterial3D (.tres).
* Tries to export Terrain heightmaps (text-serialised TerrainData) to a 16-bit
  .r16 file (loadable by Terrain3D / HTerrain) plus an info .json.

Usage
-----
GUI:  python unity2godot.py
CLI:  python unity2godot.py pack.unitypackage -o /path/to/godot_project/unity_import

Dependencies:  pip install pyyaml pillow      (Pillow is optional but recommended)
GUI uses tkinter (bundled with most Python installs). Optional drag & drop:
pip install tkinterdnd2
"""

import argparse
import array
import json
import re
import shutil
import sys
import tarfile
import tempfile
import threading
import queue
from pathlib import Path

try:
    import yaml
except ImportError:  # handled at runtime with a friendly message
    yaml = None

try:
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = None
except ImportError:
    Image = None
    ImageOps = None

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
GUID_RE = re.compile(r"^[0-9a-fA-F]{32}$")
ZERO_GUID = "0" * 32

IMAGE_NATIVE = {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".webp", ".exr", ".hdr", ".svg", ".dds"}
IMAGE_CONVERT = {".tif", ".tiff", ".psd", ".gif"}
MODELS = {".fbx", ".obj", ".dae", ".blend", ".gltf", ".glb", ".bin", ".mtl"}
AUDIO = {".wav", ".ogg", ".mp3"}
DOCS = {".txt", ".md", ".pdf"}
COPY_THROUGH = IMAGE_NATIVE | MODELS | AUDIO | DOCS

ALBEDO = ["_MainTex", "_BaseMap", "_BaseColorMap", "_BaseTex", "_Albedo", "_Diffuse"]
NORMAL = ["_BumpMap", "_NormalMap", "_NormalTex"]
METALGLOSS = ["_MetallicGlossMap", "_MetallicMap", "_MetallicTex"]
MASK = ["_MaskMap"]
OCCLUSION = ["_OcclusionMap"]
EMISSION = ["_EmissionMap", "_EmissiveColorMap"]
COLORS = ["_Color", "_BaseColor"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def safe_rel_path(pathname: str) -> Path:
    """Turn a Unity pathname into a safe relative path (drops leading Assets/)."""
    first = pathname.splitlines()[0].strip() if pathname.strip() else ""
    parts = [p for p in first.replace("\\", "/").split("/") if p not in ("", ".")]
    if parts and parts[0].lower() == "assets":
        parts = parts[1:]
    cleaned = []
    for p in parts:
        if p == "..":
            p = "_"
        p = re.sub(r'[<>:"|?*\x00-\x1f]', "_", p)
        cleaned.append(p)
    return Path(*cleaned) if cleaned else Path()


def find_project_root(start: Path):
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "project.godot").exists():
            return p
    return None


def fnum(x) -> str:
    return repr(round(float(x), 6))


def load_unity_yaml(path: Path):
    """Load a text-serialised Unity YAML file. Returns list of docs or None."""
    if yaml is None:
        raise RuntimeError("PyYAML is missing. Run: pip install pyyaml")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not text.startswith("%YAML"):
        return None  # binary-serialised
    text = re.sub(r"^%.*\n", "", text, flags=re.M)
    text = re.sub(r"^--- !u!\d+ &-?\d+.*$", "---", text, flags=re.M)
    # GUIDs made only of digits / containing 'e' would otherwise parse as numbers
    text = re.sub(r"guid: ([0-9a-fA-F]{32})", r'guid: "\1"', text)
    try:
        return [d for d in yaml.safe_load_all(text) if isinstance(d, dict)]
    except yaml.YAMLError:
        return None


def get_props(mat: dict):
    """Return (textures, floats, colors) dicts from a Unity Material body."""
    saved = mat.get("m_SavedProperties") or {}
    texs, floats, cols = {}, {}, {}

    def entries(key):
        for e in saved.get(key) or []:
            if not isinstance(e, dict):
                continue
            if "first" in e:  # old format
                first = e["first"]
                name = first.get("name") if isinstance(first, dict) else first
                yield name, e.get("second")
            else:
                for k, v in e.items():
                    yield k, v

    for name, val in entries("m_TexEnvs"):
        if not isinstance(val, dict):
            continue
        t = val.get("m_Texture") or {}
        guid = str(t.get("guid", "")) if isinstance(t, dict) else ""
        if guid and guid != ZERO_GUID:
            sc = val.get("m_Scale") or {}
            of = val.get("m_Offset") or {}
            texs[name] = {
                "guid": guid,
                "scale": (float(sc.get("x", 1)), float(sc.get("y", 1))),
                "offset": (float(of.get("x", 0)), float(of.get("y", 0))),
            }
    for name, val in entries("m_Floats"):
        try:
            floats[name] = float(val)
        except (TypeError, ValueError):
            pass
    for name, val in entries("m_Colors"):
        if isinstance(val, dict):
            cols[name] = (float(val.get("r", 1)), float(val.get("g", 1)),
                          float(val.get("b", 1)), float(val.get("a", 1)))
    return texs, floats, cols


def first_of(d: dict, names, default=None):
    for n in names:
        if n in d:
            return d[n]
    return default


class Baker:
    """Bakes Unity smoothness (alpha) into Godot roughness textures."""

    def __init__(self, enabled, log):
        self.enabled = enabled and Image is not None
        self.cache = {}
        self.log = log

    def roughness_from_alpha(self, src: Path):
        if not self.enabled:
            return None
        if src in self.cache:
            return self.cache[src]
        out = src.with_name(src.stem + "_roughness.png")
        try:
            with Image.open(src) as im:
                if "A" not in im.getbands():
                    self.cache[src] = None
                    return None
                alpha = im.getchannel("A")
                if alpha.mode != "L":
                    alpha = alpha.convert("L")
                ImageOps.invert(alpha).save(out)
            self.cache[src] = out
            return out
        except Exception as e:  # noqa
            self.log(f"  ! could not bake roughness from {src.name}: {e}")
            self.cache[src] = None
            return None


# --------------------------------------------------------------------------
# Spec builders (Unity -> neutral material spec)
# --------------------------------------------------------------------------
def build_spec_from_material(mat, gmap, baker):
    texs, floats, cols = get_props(mat)

    def T(names):
        for n in names:
            t = texs.get(n)
            if t and t["guid"] in gmap:
                return t, gmap[t["guid"]]
        return None

    spec = {}
    col = first_of(cols, COLORS, (1, 1, 1, 1))
    spec["albedo_color"] = col

    alb = T(ALBEDO)
    if alb:
        spec["albedo_tex"] = alb[1]
        spec["uv_scale"], spec["uv_offset"] = alb[0]["scale"], alb[0]["offset"]
    else:
        any_t = next((t for t in texs.values()), None)
        if any_t:
            spec["uv_scale"], spec["uv_offset"] = any_t["scale"], any_t["offset"]

    # normal
    nrm = T(NORMAL)
    if nrm:
        spec["normal_tex"] = nrm[1]
        spec["normal_scale"] = first_of(floats, ["_BumpScale", "_NormalScale"], 1.0)

    # metallic / smoothness
    gloss = first_of(floats, ["_Glossiness", "_Smoothness", "_GlossMapScale"], 0.5)
    metal = floats.get("_Metallic", 0.0)
    mask, mg = T(MASK), T(METALGLOSS)
    rough_src = None
    if mask:
        spec["metal_tex"] = mask[1]          # R
        spec["ao_tex"], spec["ao_channel"] = mask[1], 1  # G
        rough_src = mask[1]
    elif mg:
        spec["metal_tex"] = mg[1]
        rough_src = mg[1]
    elif floats.get("_SmoothnessTextureChannel", 0) == 1 and alb:
        rough_src = alb[1]
    spec["metal"] = 1.0 if "metal_tex" in spec else metal

    baked = baker.roughness_from_alpha(rough_src) if rough_src else None
    if baked:
        spec["rough_tex"] = baked
        spec["rough"] = 1.0
    else:
        spec["rough"] = 1.0 - gloss

    occ = T(OCCLUSION)
    if occ and "ao_tex" not in spec:
        spec["ao_tex"], spec["ao_channel"] = occ[1], 0

    # emission
    emis = T(EMISSION)
    ecol = first_of(cols, ["_EmissionColor", "_EmissiveColor"], (0, 0, 0, 1))
    if emis or max(ecol[:3]) > 0.001:
        mx = max(ecol[:3])
        energy = 1.0
        if mx > 1.0:
            ecol = (ecol[0] / mx, ecol[1] / mx, ecol[2] / mx, 1.0)
            energy = mx
        if emis and mx <= 0.001:
            ecol = (1, 1, 1, 1)
        spec["emission"] = (ecol, energy, emis[1] if emis else None)

    # transparency
    mode = int(floats.get("_Mode", 0))
    surface = int(floats.get("_Surface", 0))
    clip = int(floats.get("_AlphaClip", 0))
    if mode == 1 or clip:
        spec["transparency"] = 2
        spec["cutoff"] = floats.get("_Cutoff", 0.5)
    elif mode in (2, 3) or surface == 1:
        spec["transparency"] = 1

    # culling (Unity 0=off 1=front 2=back  ->  Godot 2=disabled 1=front 0=back)
    cull = first_of(floats, ["_Cull", "_CullMode"])
    if cull is not None and int(cull) in (0, 1):
        spec["cull"] = 2 if int(cull) == 0 else 1
    return spec


def build_spec_from_layer(layer, gmap, baker):
    def G(key):
        t = layer.get(key) or {}
        g = str(t.get("guid", "")) if isinstance(t, dict) else ""
        return gmap.get(g)

    spec = {"albedo_color": (1, 1, 1, 1), "triplanar": True}
    diff, nrm, mask = G("m_DiffuseTexture"), G("m_NormalMapTexture"), G("m_MaskMapTexture")
    ts = layer.get("m_TileSize") or {"x": 1, "y": 1}
    sx, sy = float(ts.get("x", 1)) or 1.0, float(ts.get("y", 1)) or 1.0
    spec["uv_scale"] = (1.0 / sx, 1.0 / sy)
    spec["uv_offset"] = (0.0, 0.0)
    if diff:
        spec["albedo_tex"] = diff
    if nrm:
        spec["normal_tex"] = nrm
        spec["normal_scale"] = float(layer.get("m_NormalScale", 1.0))
    if mask:
        spec["metal_tex"] = mask
        spec["ao_tex"], spec["ao_channel"] = mask, 1
        spec["metal"] = 1.0
        baked = baker.roughness_from_alpha(mask)
        if baked:
            spec["rough_tex"], spec["rough"] = baked, 1.0
        else:
            spec["rough"] = 1.0 - float(layer.get("m_Smoothness", 0.5))
    else:
        spec["metal"] = float(layer.get("m_Metallic", 0.0))
        spec["rough"] = 1.0 - float(layer.get("m_Smoothness", 0.0))
    return spec


# --------------------------------------------------------------------------
# Godot .tres writer
# --------------------------------------------------------------------------
def write_tres(path: Path, spec: dict, root: Path):
    ext = {}

    def tex(p: Path) -> str:
        res = "res://" + p.resolve().relative_to(root.resolve()).as_posix()
        if res not in ext:
            ext[res] = str(len(ext) + 1)
        return f'ExtResource("{ext[res]}")'

    def color(c):
        return f"Color({fnum(c[0])}, {fnum(c[1])}, {fnum(c[2])}, {fnum(c[3])})"

    props = []
    if spec.get("transparency"):
        props.append(f"transparency = {spec['transparency']}")
        if spec["transparency"] == 2:
            props.append(f"alpha_scissor_threshold = {fnum(spec.get('cutoff', 0.5))}")
    if "cull" in spec:
        props.append(f"cull_mode = {spec['cull']}")
    props.append(f"albedo_color = {color(spec.get('albedo_color', (1, 1, 1, 1)))}")
    if "albedo_tex" in spec:
        props.append(f"albedo_texture = {tex(spec['albedo_tex'])}")
    props.append(f"metallic = {fnum(spec.get('metal', 0.0))}")
    if "metal_tex" in spec:
        props.append(f"metallic_texture = {tex(spec['metal_tex'])}")
    props.append(f"roughness = {fnum(min(max(spec.get('rough', 0.5), 0), 1))}")
    if "rough_tex" in spec:
        props.append(f"roughness_texture = {tex(spec['rough_tex'])}")
    if "emission" in spec:
        ecol, energy, etex = spec["emission"]
        props.append("emission_enabled = true")
        props.append(f"emission = {color(ecol)}")
        props.append(f"emission_energy_multiplier = {fnum(energy)}")
        if etex:
            props.append(f"emission_texture = {tex(etex)}")
    if "normal_tex" in spec:
        props.append("normal_enabled = true")
        props.append(f"normal_scale = {fnum(spec.get('normal_scale', 1.0))}")
        props.append(f"normal_texture = {tex(spec['normal_tex'])}")
    if "ao_tex" in spec:
        props.append("ao_enabled = true")
        if spec.get("ao_channel", 0) != 0:
            props.append(f"ao_texture_channel = {spec['ao_channel']}")
        props.append(f"ao_texture = {tex(spec['ao_tex'])}")
    sc = spec.get("uv_scale", (1, 1))
    of = spec.get("uv_offset", (0, 0))
    if tuple(sc) != (1, 1):
        props.append(f"uv1_scale = Vector3({fnum(sc[0])}, {fnum(sc[1])}, 1.0)")
    if tuple(of) != (0, 0):
        props.append(f"uv1_offset = Vector3({fnum(of[0])}, {fnum(of[1])}, 0.0)")
    if spec.get("triplanar"):
        props.append("uv1_triplanar = true")
        props.append("uv1_world_triplanar = true")

    lines = [f'[gd_resource type="StandardMaterial3D" load_steps={len(ext) + 1} format=3]', ""]
    for res, i in ext.items():
        lines.append(f'[ext_resource type="Texture2D" path="{res}" id="{i}"]')
    if ext:
        lines.append("")
    lines.append("[resource]")
    lines.extend(props)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Terrain height export
# --------------------------------------------------------------------------
def export_terrain(asset_path: Path, dest_base: Path, layer_paths: dict, log):
    try:
        text = asset_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    if not text.startswith("%YAML") or "TerrainData:" not in text:
        return False
    hm = text.find("m_Heightmap:")
    if hm < 0:
        log(f"  ! {dest_base.name}: no heightmap section found")
        return True
    seg = text[hm:]
    mres = re.search(r"m_Resolution:\s*(\d+)", seg)
    mh = re.search(r"m_Heights:\s*([0-9a-fA-F]+)", seg)
    if not (mres and mh):
        log(f"  ! {dest_base.name}: heightmap data not readable")
        return True
    res = int(mres.group(1))
    raw = bytes.fromhex(mh.group(1))
    if len(raw) != res * res * 2:
        log(f"  ! {dest_base.name}: unexpected heightmap length, skipped")
        return True
    arr = array.array("H")
    arr.frombytes(raw)
    if sys.byteorder == "big":
        arr.byteswap()
    # Unity stores 0..32766 (int16). Stretch to full 0..65535.
    out = array.array("H", (min(65535, v * 65535 // 32766) for v in arr))
    if sys.byteorder == "big":
        out.byteswap()
    r16 = dest_base.with_name(dest_base.name + "_height.r16")
    r16.parent.mkdir(parents=True, exist_ok=True)
    r16.write_bytes(out.tobytes())

    msc = re.search(r"m_Scale:\s*\{x:\s*([-\d.eE+]+),\s*y:\s*([-\d.eE+]+),\s*z:\s*([-\d.eE+]+)\}", seg)
    scale = [float(msc.group(i)) for i in (1, 2, 3)] if msc else None
    layers = []
    ml = text.find("m_TerrainLayers:")
    if ml >= 0:
        for line in text[ml:].splitlines()[1:]:
            if not line.lstrip().startswith("-"):
                break
            g = re.search(r"guid:\s*\"?([0-9a-fA-F]{32})", line)
            if g:
                layers.append(layer_paths.get(g.group(1), g.group(1)))
    info = {
        "resolution": res,
        "heightmap_file": r16.name,
        "format": "16-bit unsigned little-endian RAW (.r16), full 0..65535 range",
        "unity_heightmap_scale": scale,
        "approx_height_range_m": round(scale[1] * 32766, 3) if scale else None,
        "approx_height_note": "Verify against the Terrain Size Y you see in Unity/the asset's docs.",
        "terrain_layers_in_order": layers,
    }
    r16.with_name(dest_base.name + "_terrain_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log(f"  terrain heightmap {res}x{res} -> {r16.name}")
    return True


# --------------------------------------------------------------------------
# Main conversion
# --------------------------------------------------------------------------
def convert(package: Path, out_dir: Path, project_root: Path = None,
            bake_roughness=True, keep_other=False, log=print):
    if yaml is None:
        raise RuntimeError("PyYAML is missing. Run: pip install pyyaml")
    package, out_dir = Path(package), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(project_root) if project_root else find_project_root(out_dir)
    if root is None:
        root = out_dir
        log("! No project.godot found above the output folder. Material paths will be "
            "relative to the output folder - put the output inside your Godot project "
            "(or pass --project-root) so res:// paths resolve.")
    log(f"Godot project root: {root}")
    if Image is None and bake_roughness:
        log("! Pillow not installed: roughness maps can't be baked and .tif/.psd can't be converted.")

    stats = {"textures": 0, "models": 0, "audio": 0, "materials": 0,
             "terrain_layers": 0, "terrains": 0, "skipped": 0}
    skipped_types = {}
    gmap = {}   # guid -> final absolute Path (textures only)
    mats, layers, assets = [], [], []

    with tempfile.TemporaryDirectory(prefix="u2g_") as tmpname:
        tmp = Path(tmpname)
        log(f"Reading {package.name} ...")
        with tarfile.open(package, "r:*") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                name = m.name.replace("\\", "/")
                while name.startswith("./"):
                    name = name[2:]
                parts = name.split("/")
                if len(parts) != 2 or not GUID_RE.match(parts[0]) or parts[1] not in ("asset", "pathname"):
                    continue
                d = tmp / parts[0]
                d.mkdir(exist_ok=True)
                with tar.extractfile(m) as src, open(d / parts[1], "wb") as dst:
                    shutil.copyfileobj(src, dst)

        entries = []
        for d in tmp.iterdir():
            a, p = d / "asset", d / "pathname"
            if a.exists() and p.exists():
                rel = safe_rel_path(p.read_text(encoding="utf-8", errors="replace"))
                if rel.name:
                    entries.append((d.name, a, rel))
        log(f"Found {len(entries)} files in package")

        # pass 1: copy/convert plain files, collect special ones
        for guid, asset, rel in entries:
            ext = rel.suffix.lower()
            dest = out_dir / rel
            if ext == ".mat":
                mats.append((guid, asset, rel))
            elif ext == ".terrainlayer":
                layers.append((guid, asset, rel))
            elif ext == ".asset":
                assets.append((guid, asset, rel))
            elif ext in IMAGE_CONVERT:
                if Image is None:
                    stats["skipped"] += 1
                    skipped_types[ext] = skipped_types.get(ext, 0) + 1
                    continue
                try:
                    dest = dest.with_suffix(".png")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(asset) as im:
                        if im.mode not in ("L", "LA", "RGB", "RGBA", "I;16"):
                            im = im.convert("RGBA" if "A" in im.getbands() or im.mode == "P" else "RGB")
                        im.save(dest)
                    gmap[guid] = dest
                    stats["textures"] += 1
                except Exception as e:  # noqa
                    log(f"  ! could not convert {rel}: {e}")
            elif ext in COPY_THROUGH or keep_other:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(asset, dest)
                if ext in IMAGE_NATIVE:
                    gmap[guid] = dest
                    stats["textures"] += 1
                elif ext in MODELS:
                    stats["models"] += 1
                elif ext in AUDIO:
                    stats["audio"] += 1
            else:
                stats["skipped"] += 1
                skipped_types[ext or "(none)"] = skipped_types.get(ext or "(none)", 0) + 1

        baker = Baker(bake_roughness, log)
        layer_paths = {}

        # pass 2: materials
        for guid, asset, rel in mats:
            docs = load_unity_yaml(asset)
            body = next((d["Material"] for d in docs or [] if "Material" in d), None)
            if body is None:
                log(f"  ! {rel}: binary-serialised or unreadable material, skipped")
                continue
            try:
                spec = build_spec_from_material(body, gmap, baker)
                target = (out_dir / rel).with_suffix(".tres")
                write_tres(target, spec, root)
                stats["materials"] += 1
            except Exception as e:  # noqa
                log(f"  ! {rel}: {e}")

        # pass 3: terrain layers
        for guid, asset, rel in layers:
            docs = load_unity_yaml(asset)
            body = next((d["TerrainLayer"] for d in docs or [] if "TerrainLayer" in d), None)
            if body is None:
                log(f"  ! {rel}: unreadable terrain layer, skipped")
                continue
            try:
                spec = build_spec_from_layer(body, gmap, baker)
                target = (out_dir / rel).with_suffix(".tres")
                write_tres(target, spec, root)
                layer_paths[guid] = target.relative_to(out_dir).as_posix()
                stats["terrain_layers"] += 1
            except Exception as e:  # noqa
                log(f"  ! {rel}: {e}")

        # pass 4: terrain data
        for guid, asset, rel in assets:
            base = (out_dir / rel).with_suffix("")
            if export_terrain(asset, base, layer_paths, log):
                stats["terrains"] += 1
            else:
                stats["skipped"] += 1
                skipped_types[".asset"] = skipped_types.get(".asset", 0) + 1

    log("")
    log("Done.")
    log(f"  textures: {stats['textures']}   models: {stats['models']}   audio: {stats['audio']}")
    log(f"  materials converted: {stats['materials']}   terrain layers: {stats['terrain_layers']}"
        f"   terrain heightmaps: {stats['terrains']}")
    if skipped_types:
        listing = ", ".join(f"{k} x{v}" for k, v in sorted(skipped_types.items()))
        log(f"  not converted (Unity-only): {listing}")
    log("")
    log("Next: open the Godot project and let it import. Textures used as normal maps may")
    log("prompt a one-time 'Fix' / reimport as Normal Map in the material inspector.")
    return stats


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
def run_gui():
    import tkinter as tk
    from tkinter import filedialog, ttk, scrolledtext

    dnd = False
    try:
        from tkinterdnd2 import TkinterDnD, DND_FILES
        root = TkinterDnD.Tk()
        dnd = True
    except Exception:
        root = tk.Tk()
    root.title("Unity Package -> Godot")
    root.geometry("720x560")

    pkg_var, proj_var, sub_var = tk.StringVar(), tk.StringVar(), tk.StringVar(value="unity_import")
    bake_var, other_var = tk.BooleanVar(value=True), tk.BooleanVar(value=False)
    q = queue.Queue()

    frm = ttk.Frame(root, padding=12)
    frm.pack(fill="both", expand=True)
    frm.columnconfigure(1, weight=1)

    ttk.Label(frm, text=".unitypackage:").grid(row=0, column=0, sticky="w", pady=4)
    pkg_entry = ttk.Entry(frm, textvariable=pkg_var)
    pkg_entry.grid(row=0, column=1, sticky="ew", padx=6)
    ttk.Button(frm, text="Browse", command=lambda: pkg_var.set(
        filedialog.askopenfilename(filetypes=[("Unity package", "*.unitypackage"), ("All", "*.*")])
        or pkg_var.get())).grid(row=0, column=2)

    ttk.Label(frm, text="Godot project folder:").grid(row=1, column=0, sticky="w", pady=4)
    ttk.Entry(frm, textvariable=proj_var).grid(row=1, column=1, sticky="ew", padx=6)
    ttk.Button(frm, text="Browse", command=lambda: proj_var.set(
        filedialog.askdirectory() or proj_var.get())).grid(row=1, column=2)

    ttk.Label(frm, text="Subfolder inside project:").grid(row=2, column=0, sticky="w", pady=4)
    ttk.Entry(frm, textvariable=sub_var).grid(row=2, column=1, sticky="ew", padx=6)

    ttk.Checkbutton(frm, text="Bake Unity smoothness into Godot roughness textures",
                    variable=bake_var).grid(row=3, column=1, sticky="w")
    ttk.Checkbutton(frm, text="Also copy other file types (scripts, prefabs, etc.)",
                    variable=other_var).grid(row=4, column=1, sticky="w")

    btn = ttk.Button(frm, text="Convert")
    btn.grid(row=5, column=1, pady=10)
    logbox = scrolledtext.ScrolledText(frm, height=18, state="disabled", font=("Consolas", 9))
    logbox.grid(row=6, column=0, columnspan=3, sticky="nsew")
    frm.rowconfigure(6, weight=1)

    if dnd:
        pkg_entry.drop_target_register(DND_FILES)
        pkg_entry.dnd_bind("<<Drop>>", lambda e: pkg_var.set(e.data.strip("{}")))
        ttk.Label(frm, text="(You can drag a .unitypackage onto the first field)").grid(
            row=7, column=0, columnspan=3, sticky="w")

    def log(msg):
        q.put(msg)

    def pump():
        try:
            while True:
                msg = q.get_nowait()
                logbox.config(state="normal")
                logbox.insert("end", msg + "\n")
                logbox.see("end")
                logbox.config(state="disabled")
        except queue.Empty:
            pass
        root.after(100, pump)

    def work(pkg, proj, out):
        try:
            convert(pkg, out, project_root=proj, bake_roughness=bake_var.get(),
                    keep_other=other_var.get(), log=log)
        except Exception as e:  # noqa
            log(f"ERROR: {e}")
        finally:
            root.after(0, lambda: btn.config(state="normal"))

    def start():
        if not pkg_var.get() or not proj_var.get():
            log("Please choose a .unitypackage and your Godot project folder.")
            return
        proj = Path(proj_var.get())
        out = proj / sub_var.get().strip() if sub_var.get().strip() else proj
        if not (proj / "project.godot").exists():
            log("Note: no project.godot in that folder - is it the project root?")
        btn.config(state="disabled")
        threading.Thread(target=work, args=(Path(pkg_var.get()), proj, out), daemon=True).start()

    btn.config(command=start)
    pump()
    root.mainloop()


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Convert a .unitypackage into Godot 4 assets.")
    ap.add_argument("package", nargs="?", help=".unitypackage file (omit to open the GUI)")
    ap.add_argument("-o", "--out", help="output folder (inside your Godot project)")
    ap.add_argument("--project-root", help="Godot project folder (auto-detected from --out if omitted)")
    ap.add_argument("--no-bake", action="store_true", help="don't bake roughness textures")
    ap.add_argument("--keep-other", action="store_true", help="also copy scripts/prefabs/etc.")
    args = ap.parse_args()

    if not args.package:
        run_gui()
        return
    if not args.out:
        ap.error("--out is required in CLI mode")
    try:
        convert(Path(args.package), Path(args.out),
                Path(args.project_root) if args.project_root else None,
                bake_roughness=not args.no_bake, keep_other=args.keep_other)
    except Exception as e:  # noqa
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
