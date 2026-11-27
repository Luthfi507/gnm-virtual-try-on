"""Exports GNM head weights, semantic character presets, and sample .obj accessories for the web app.

Usage (from repo root with conda env `gnm`):
    python web/tools/export_assets.py
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any

from absl import app, flags, logging
from gnm.shape import gnm_numpy, semantic_sampler
import numpy as np
import numpy.typing as npt
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "web" / "public" / "assets"
DEFAULT_CORR = REPO_ROOT / "assets" / "correspondence.npz"

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    str(DEFAULT_OUT),
    "Directory to write gnm_head.bin, gnm_head.json, character_presets.json, and objects/.",
)
_IDENTITY_COMPONENTS = flags.DEFINE_integer(
    "identity_components",
    64,
    "Number of identity components to keep.",
)
_CORRESPONDENCE_PATH = flags.DEFINE_string(
    "correspondence_path",
    str(DEFAULT_CORR),
    "MediaPipe landmark to GNM vertex correspondence .npz.",
)
_EXPRESSION_BUDGET = flags.DEFINE_string(
    "expression_budget",
    "left_eye_region=24,right_eye_region=24,lower_face_region=32,tongue=8,pupils=1",
    "Per-region expression component budget as name=count pairs.",
)

_ALIGNMENT = 4
_DTYPE_NAMES = {
    np.dtype(np.float32): "float32",
    np.dtype(np.float16): "float16",
    np.dtype(np.uint16): "uint16",
    np.dtype(np.uint8): "uint8",
}


def parse_budget(spec: str) -> dict[str, int]:
    """Parses comma-separated region=count budget pairs."""
    budget = {}
    for pair in spec.split(","):
        if not pair.strip():
            continue
        name, sep, count = pair.partition("=")
        if not sep:
            raise ValueError(f"Malformed budget entry {pair!r}, expected name=count.")
        val = int(count)
        if val <= 0:
            raise ValueError(f"Budget for {name!r} must be positive, got {val}.")
        budget[name.strip()] = val
    return budget


def region_of(name: str) -> str:
    return name.rsplit("_", 1)[0]


def select_expression_components(
    expression_names: Sequence[str],
    budget: Mapping[str, int],
) -> npt.NDArray[np.int32]:
    regions: dict[str, list[int]] = {}
    for idx, name in enumerate(expression_names):
        regions.setdefault(region_of(name), []).append(idx)

    unknown = set(budget) - set(regions)
    if unknown:
        raise ValueError(f"Unknown regions {sorted(unknown)}. Available: {sorted(regions)}.")

    kept: list[int] = []
    for region, indices in regions.items():
        count = budget.get(region, 0)
        kept.extend(indices[:count])
    return np.asarray(sorted(kept), dtype=np.int32)


def component_ids(gnm: gnm_numpy.GNM) -> npt.NDArray[np.uint8]:
    weights = np.stack([np.asarray(gnm.vertex_group(name)) for name in gnm.mesh_component_names])
    if not np.all(weights.max(axis=0) > 0):
        raise ValueError("Some vertices belong to no mesh component.")
    return weights.argmax(axis=0).astype(np.uint8)


def quad_edges(quads: npt.NDArray[np.integer]) -> npt.NDArray[np.uint16]:
    tail = quads.ravel()
    head = np.roll(quads, -1, axis=1).ravel()
    pairs = np.sort(np.stack([tail, head], axis=1), axis=1)
    return np.unique(pairs, axis=0).astype(np.uint16)


def normal_adjacency(
    triangles: npt.NDArray[np.integer],
    vertex_count: int,
) -> tuple[npt.NDArray[np.uint16], npt.NDArray[np.uint8], int]:
    incident: list[list[tuple[int, int]]] = [[] for _ in range(vertex_count)]
    for tri in triangles:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        incident[a].append((b, c))
        incident[b].append((c, a))
        incident[c].append((a, b))

    max_corners = max(len(entry) for entry in incident)
    pairs = np.zeros((vertex_count, max_corners, 2), dtype=np.uint16)
    counts = np.zeros(vertex_count, dtype=np.uint8)
    for v, entry in enumerate(incident):
        counts[v] = len(entry)
        for slot, (p, q) in enumerate(entry):
            pairs[v, slot] = (p, q)
    return pairs, counts, max_corners


def _vertex_normals(vertices: npt.NDArray[np.floating], triangles: npt.NDArray[np.integer]) -> npt.NDArray[np.float64]:
    a, b, c = (vertices[triangles[:, k]] for k in range(3))
    face = np.cross(b - a, c - a)
    normals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(normals, triangles[:, k], face)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-12)


def cheek_puff_corrective(gnm: gnm_numpy.GNM, amplitude: float = 0.012, radius: float = 0.035) -> npt.NDArray[np.float32]:
    vertices = np.asarray(gnm.template_vertex_positions, dtype=np.float64)
    skin = np.asarray(gnm.vertex_group_indices("skin"))
    normals = _vertex_normals(vertices, np.asarray(gnm.triangles))

    eyes = np.concatenate([
        np.asarray(gnm.vertex_group_indices(name))
        for name in ("left_eye", "right_eye")
    ])
    lower_teeth = np.asarray(gnm.vertex_group_indices("lower_teeth_and_gums"))
    low = vertices[lower_teeth, 1].min()
    high = vertices[eyes, 1].min()

    sideways = np.abs(normals[:, 0]) > 0.5
    band = skin[
        sideways[skin]
        & (vertices[skin, 1] > low)
        & (vertices[skin, 1] < high)
        & (vertices[skin, 2] > vertices[:, 2].mean())
    ]
    weight = np.zeros(len(vertices))
    for side in (band[vertices[band, 0] > 0], band[vertices[band, 0] < 0]):
        centre = vertices[side].mean(axis=0)
        distance = np.linalg.norm(vertices[skin] - centre, axis=1)
        falloff = np.clip(1.0 - (distance / radius) ** 2, 0.0, 1.0) ** 2
        weight[skin] = np.maximum(weight[skin], falloff)

    return (normals * (weight[:, None] * amplitude)).astype(np.float32)


def _hsl_to_rgb(hue: npt.NDArray[np.floating], saturation: float, lightness: npt.NDArray[np.floating]) -> npt.NDArray[np.uint8]:
    chroma = (1.0 - np.abs(2.0 * lightness - 1.0)) * saturation
    sector = hue * 6.0
    second = chroma * (1.0 - np.abs(np.mod(sector, 2.0) - 1.0))
    base = lightness - chroma / 2.0
    zeros = np.zeros_like(hue)
    table = np.stack([
        np.stack([chroma, second, zeros], axis=-1),
        np.stack([second, chroma, zeros], axis=-1),
        np.stack([zeros, chroma, second], axis=-1),
        np.stack([zeros, second, chroma], axis=-1),
        np.stack([second, zeros, chroma], axis=-1),
        np.stack([chroma, zeros, second], axis=-1),
    ])
    rgb = table[np.clip(sector.astype(np.int32), 0, 5), np.arange(hue.size)]
    return np.clip((rgb + base[:, None]) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def correspondence_colors(positions: npt.NDArray[np.floating]) -> npt.NDArray[np.uint8]:
    def normalize(values):
        low, high = values.min(), values.max()
        return (values - low) / max(high - low, 1e-9)

    hue = normalize(positions[:, 0]) * 0.82
    lightness = 0.34 + normalize(positions[:, 1]) * 0.24
    return _hsl_to_rgb(hue, 0.95, lightness)


def load_correspondence(path: str, vertex_count: int) -> dict[str, npt.NDArray[Any]] | None:
    if not os.path.exists(path):
        logging.warning("No correspondence at %s; identity fit will be unavailable.", path)
        return None
    data = np.load(path, allow_pickle=True)
    landmark_indices = np.asarray(data["landmark_indices"], dtype=np.int32)
    vertex_indices = np.asarray(data["vertex_indices"], dtype=np.int32)
    if vertex_indices.max() >= vertex_count:
        raise ValueError("Correspondence vertex index out of bounds.")
    reference = np.asarray(data["reference_landmarks"], dtype=np.float32)
    return {
        "landmark_indices": landmark_indices.astype(np.uint16),
        "vertex_indices": vertex_indices.astype(np.uint16),
        "rigid": np.asarray(data["rigid"]).astype(np.uint8),
        "reference": reference,
    }


def _as_uint16_bits(values: npt.NDArray[np.floating]) -> npt.NDArray[np.uint16]:
    return values.astype(np.float16).view(np.uint16)


def serialize(arrays: Mapping[str, npt.NDArray[Any]]) -> tuple[bytes, dict[str, Any]]:
    declared = {"identity_basis", "expression_basis", "corrective_basis"}
    blob = bytearray()
    views: dict[str, Any] = {}
    for name, array in arrays.items():
        padding = -len(blob) % _ALIGNMENT
        blob.extend(b"\0" * padding)
        contiguous = np.ascontiguousarray(array)
        payload = contiguous.tobytes()
        dtype = "float16" if name in declared else _DTYPE_NAMES[contiguous.dtype]
        views[name] = {
            "dtype": dtype,
            "shape": list(contiguous.shape),
            "byteOffset": len(blob),
            "byteLength": len(payload),
        }
        blob.extend(payload)
    return bytes(blob), views


def export_gnm(output_dir: Path, identity_components: int, expression_budget: Mapping[str, int]) -> None:
    gnm = gnm_numpy.GNM.from_remote(
        version=gnm_numpy.GNMMajorVersion.V3,
        variant=gnm_numpy.GNMVariant.HEAD,
    )
    expression_names = [str(name) for name in gnm.expression_names]
    identity_names = [str(name) for name in gnm.identity_names]
    identity_indices = np.arange(min(identity_components, len(identity_names)), dtype=np.int32)
    expression_indices = select_expression_components(expression_names, expression_budget)
    correspondence = load_correspondence(_CORRESPONDENCE_PATH.value, int(gnm.num_vertices))

    triangles = np.asarray(gnm.triangles)
    adjacency_pairs, adjacency_counts, _ = normal_adjacency(triangles, int(gnm.num_vertices))
    arrays: dict[str, npt.NDArray[Any]] = {
        "template_vertex_positions": np.asarray(gnm.template_vertex_positions, dtype=np.float32),
        "identity_basis": _as_uint16_bits(np.asarray(gnm.vertex_identity_basis)[identity_indices]),
        "expression_basis": _as_uint16_bits(np.asarray(gnm.expression_basis)[expression_indices]),
        "template_joint_positions": np.asarray(gnm.template_joint_positions, dtype=np.float32),
        "joint_identity_basis": np.asarray(gnm.joint_identity_basis)[identity_indices].astype(np.float32),
        "skinning_weights": np.asarray(gnm.skinning_weights, dtype=np.float32),
        "triangles": triangles.astype(np.uint16),
        "quad_edges": quad_edges(np.asarray(gnm.quads)),
        "component_ids": component_ids(gnm),
        "normal_adjacency": adjacency_pairs,
        "normal_adjacency_count": adjacency_counts,
        "corrective_basis": _as_uint16_bits(np.stack([cheek_puff_corrective(gnm)])),
    }
    if correspondence is not None:
        template = np.asarray(gnm.template_vertex_positions, dtype=np.float32)
        arrays["correspondence_landmarks"] = correspondence["landmark_indices"]
        arrays["correspondence_vertices"] = correspondence["vertex_indices"]
        arrays["correspondence_rigid"] = correspondence["rigid"]
        arrays["correspondence_reference"] = correspondence["reference"]
        arrays["correspondence_colors"] = correspondence_colors(
            template[correspondence["vertex_indices"].astype(np.int32)]
        )

    blob, views = serialize(arrays)
    kept_expression = [expression_names[i] for i in expression_indices]
    regions: dict[str, list[int]] = {}
    for pos, name in enumerate(kept_expression):
        regions.setdefault(region_of(name), []).append(pos)

    manifest = {
        "model": {
            "version": str(gnm.version),
            "variant": str(gnm.variant),
            "vertexCount": int(gnm.num_vertices),
            "jointCount": int(gnm.num_joints),
            "jointNames": [str(n) for n in gnm.joint_names],
            "jointParentIndices": [int(i) for i in gnm.joint_parent_indices],
            "componentNames": [str(n) for n in gnm.mesh_component_names],
            "maxNormalCorners": views["normal_adjacency"]["shape"][1],
        },
        "correspondence": {"count": 0 if correspondence is None else len(correspondence["landmark_indices"])},
        "correctives": {"names": ["cheek_puff"], "blendshapes": ["cheekPuff"]},
        "identity": {
            "count": len(identity_indices),
            "sourceCount": len(identity_names),
            "indices": [int(i) for i in identity_indices],
            "names": [identity_names[i] for i in identity_indices],
        },
        "expression": {
            "count": len(expression_indices),
            "sourceCount": len(expression_names),
            "indices": [int(i) for i in expression_indices],
            "names": kept_expression,
            "regions": {name: {"start": pos[0], "count": len(pos)} for name, pos in regions.items()},
        },
        "arrays": views,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gnm_head.bin").write_bytes(blob)
    (output_dir / "gnm_head.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.info("Exported GNM binary (%.1f MB) and manifest.", len(blob) / 1e6)


def export_character_presets(output_dir: Path, identity_components: int) -> None:
    """Exports curated semantic identity presets sampled from GNM IdentitySampler."""
    rng = np.random.default_rng(42)
    presets: list[dict[str, Any]] = [
        {"id": "neutral", "name": "Neutral Template", "identity": [0.0] * identity_components},
    ]
    try:
        sampler = semantic_sampler.IdentitySampler()
        specs = [
            ("female_asian", "Asian Female", semantic_sampler.Gender.FEMALE, semantic_sampler.Ethnicity.ASIAN),
            ("male_asian", "Asian Male", semantic_sampler.Gender.MALE, semantic_sampler.Ethnicity.ASIAN),
            ("female_white", "European Female", semantic_sampler.Gender.FEMALE, semantic_sampler.Ethnicity.WHITE),
            ("male_white", "European Male", semantic_sampler.Gender.MALE, semantic_sampler.Ethnicity.WHITE),
            ("female_black", "African Female", semantic_sampler.Gender.FEMALE, semantic_sampler.Ethnicity.BLACK),
            ("male_me", "Middle Eastern Male", semantic_sampler.Gender.MALE, semantic_sampler.Ethnicity.MIDDLE_EASTERN),
        ]
        for pid, label, gender, eth in specs:
            vec = sampler.sample_identity(gender, eth, 1, rng=rng)[0, :identity_components]
            # Clamp gently for clean visual proportions
            vec = np.clip(vec, -2.5, 2.5)
            presets.append({"id": pid, "name": label, "identity": [round(float(v), 4) for v in vec]})
    except Exception as exc:
        logging.warning("IdentitySampler unavailable (%s); using deterministic presets.", exc)

    # Add two expressive stylized character archetypes
    sculpted = np.zeros(identity_components, dtype=np.float32)
    sculpted[:6] = [1.8, -1.2, 1.4, -0.9, 1.1, -0.7]
    presets.append({"id": "sculpted_hero", "name": "Sculpted Hero", "identity": [round(float(v), 4) for v in sculpted]})

    chibi = np.zeros(identity_components, dtype=np.float32)
    chibi[:6] = [-1.6, 1.5, -1.1, 1.3, -0.8, 0.9]
    presets.append({"id": "soft_stylized", "name": "Soft Stylized", "identity": [round(float(v), 4) for v in chibi]})

    (output_dir / "character_presets.json").write_text(
        json.dumps({"presets": presets}, indent=2),
        encoding="utf-8",
    )


def _save_obj_with_mtl(obj_dir: Path, name: str, parts: list[tuple[str, trimesh.Trimesh, tuple[float, float, float], float]]) -> None:
    """Writes a multi-material Wavefront .obj + .mtl pair using trimesh scenes/export."""
    mtl_name = f"{name}.mtl"
    mtl_lines = []
    obj_lines = [f"mtllib {mtl_name}"]
    v_offset = 1
    vn_offset = 1

    for mat_name, mesh, kd, ks in parts:
        mtl_lines.extend([
            f"newmtl {mat_name}",
            f"Ka {kd[0]*0.2:.4f} {kd[1]*0.2:.4f} {kd[2]*0.2:.4f}",
            f"Kd {kd[0]:.4f} {kd[1]:.4f} {kd[2]:.4f}",
            f"Ks {ks:.4f} {ks:.4f} {ks:.4f}",
            "Ns 96.0",
            "d 1.0",
            "",
        ])
        mesh.fix_normals()
        obj_lines.append(f"o {mat_name}")
        obj_lines.append(f"usemtl {mat_name}")
        for vx, vy, vz in mesh.vertices:
            obj_lines.append(f"v {vx:.6f} {vy:.6f} {vz:.6f}")
        for nx, ny, nz in mesh.vertex_normals:
            obj_lines.append(f"vn {nx:.6f} {ny:.6f} {nz:.6f}")
        for f0, f1, f2 in mesh.faces:
            a, b, c = f0 + v_offset, f1 + v_offset, f2 + v_offset
            na, nb, nc = f0 + vn_offset, f1 + vn_offset, f2 + vn_offset
            obj_lines.append(f"f {a}//{na} {b}//{nb} {c}//{nc}")
        v_offset += len(mesh.vertices)
        vn_offset += len(mesh.vertex_normals)

    (obj_dir / mtl_name).write_text("\n".join(mtl_lines) + "\n", encoding="utf-8")
    (obj_dir / f"{name}.obj").write_text("\n".join(obj_lines) + "\n", encoding="utf-8")


def export_sample_objects(output_dir: Path) -> None:
    """Generates clean procedural sample .obj + .mtl accessories in face-centered meter coordinates."""
    obj_dir = output_dir / "objects"
    obj_dir.mkdir(parents=True, exist_ok=True)

    # 1. Round Optical Glasses (centered around origin; placed at nose bridge via default offset)
    glasses_frame: list[trimesh.Trimesh] = []
    glasses_lenses: list[trimesh.Trimesh] = []
    lens_r = 0.024
    bridge_w = 0.020
    temple_len = 0.115
    for sign in (-1.0, 1.0):
        cx = sign * (lens_r + bridge_w / 2)
        rim = trimesh.creation.torus(major_radius=lens_r, minor_radius=0.0026)
        rim.apply_translation([cx, 0.0, 0.0])
        glasses_frame.append(rim)

        lens = trimesh.creation.cylinder(radius=lens_r * 0.94, height=0.0014, sections=32)
        lens.apply_translation([cx, 0.0, 0.0])
        glasses_lenses.append(lens)

        temple = trimesh.creation.box(extents=[0.0032, 0.0045, temple_len])
        temple.apply_translation([sign * (2 * lens_r + bridge_w / 2 + 0.002), 0.003, -temple_len / 2])
        glasses_frame.append(temple)

    bridge = trimesh.creation.box(extents=[bridge_w + 0.004, 0.0032, 0.0032])
    bridge.apply_translation([0.0, 0.004, 0.0])
    glasses_frame.append(bridge)
    _save_obj_with_mtl(
        obj_dir,
        "round_glasses",
        [
            ("frame_matte", trimesh.util.concatenate(glasses_frame), (0.12, 0.14, 0.18), 0.6),
            ("lens_tint", trimesh.util.concatenate(glasses_lenses), (0.28, 0.68, 0.65), 0.9),
        ],
    )

    # 2. Cyber Visor
    visor_band = trimesh.creation.box(extents=[0.142, 0.036, 0.010])
    visor_band.apply_translation([0.0, 0.0, 0.008])
    visor_arms = []
    for sign in (-1.0, 1.0):
        arm = trimesh.creation.box(extents=[0.005, 0.012, 0.11])
        arm.apply_translation([sign * 0.070, 0.0, -0.048])
        visor_arms.append(arm)
    _save_obj_with_mtl(
        obj_dir,
        "cyber_visor",
        [
            ("visor_shield", visor_band, (0.15, 0.85, 0.72), 0.95),
            ("visor_frame", trimesh.util.concatenate(visor_arms), (0.12, 0.13, 0.22), 0.5),
        ],
    )

    # 3. Royal Crown
    crown_parts: list[trimesh.Trimesh] = []
    band = trimesh.creation.cylinder(radius=0.088, height=0.022, sections=36)
    band.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    crown_parts.append(band)
    for k in range(8):
        ang = k * (2 * np.pi / 8)
        spike = trimesh.creation.cone(radius=0.014, height=0.040, sections=16)
        spike.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        spike.apply_translation([0.082 * np.cos(ang), 0.026, 0.082 * np.sin(ang)])
        crown_parts.append(spike)
    _save_obj_with_mtl(
        obj_dir,
        "royal_crown",
        [("gold_crown", trimesh.util.concatenate(crown_parts), (0.92, 0.72, 0.22), 0.95)],
    )

    # 4. Top Hat
    brim = trimesh.creation.cylinder(radius=0.145, height=0.006, sections=40)
    brim.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    body = trimesh.creation.cylinder(radius=0.095, height=0.125, sections=40)
    body.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    body.apply_translation([0.0, 0.064, 0.0])
    ribbon = trimesh.creation.cylinder(radius=0.097, height=0.020, sections=40)
    ribbon.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    ribbon.apply_translation([0.0, 0.014, 0.0])
    _save_obj_with_mtl(
        obj_dir,
        "top_hat",
        [
            ("hat_felt", trimesh.util.concatenate([brim, body]), (0.14, 0.14, 0.18), 0.15),
            ("hat_ribbon", ribbon, (0.62, 0.22, 0.56), 0.4),
        ],
    )

    # 5. Cat Ears Headband
    band_parts: list[trimesh.Trimesh] = []
    arc = trimesh.creation.torus(major_radius=0.092, minor_radius=0.006, major_sections=36, minor_sections=12)
    band_parts.append(arc)
    ear_inners: list[trimesh.Trimesh] = []
    for sign in (-1.0, 1.0):
        ear = trimesh.creation.cone(radius=0.030, height=0.055, sections=16)
        ear.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        ear.apply_transform(trimesh.transformations.rotation_matrix(sign * -0.28, [0, 0, 1]))
        ear.apply_translation([sign * 0.065, 0.088, 0.0])
        band_parts.append(ear)

        inner = trimesh.creation.cone(radius=0.017, height=0.034, sections=16)
        inner.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        inner.apply_transform(trimesh.transformations.rotation_matrix(sign * -0.28, [0, 0, 1]))
        inner.apply_translation([sign * 0.064, 0.085, 0.008])
        ear_inners.append(inner)
    _save_obj_with_mtl(
        obj_dir,
        "cat_ears",
        [
            ("ear_outer", trimesh.util.concatenate(band_parts), (0.16, 0.18, 0.26), 0.35),
            ("ear_inner", trimesh.util.concatenate(ear_inners), (0.88, 0.42, 0.65), 0.2),
        ],
    )


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError(f"Unexpected arguments: {argv[1:]}")
    out = Path(_OUTPUT_DIR.value)
    export_gnm(out, _IDENTITY_COMPONENTS.value, parse_budget(_EXPRESSION_BUDGET.value))
    export_character_presets(out, _IDENTITY_COMPONENTS.value)
    export_sample_objects(out)
    print(f"[OK] Exported web assets to {out}")


if __name__ == "__main__":
    app.run(main)
