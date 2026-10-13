"""Reproduce the survey demo from mathematical geometry, never from a real scan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WIDTH_M = 8.0
DEPTH_M = 5.0
EAVE_M = 3.0
RIDGE_M = 3.8
THICKNESS_M = 0.18
SPACING_M = 0.04
NOISE_SIGMA_M = 0.0015
YAW_DEG = 17.0
SEED = 2026
DEMO_STAMP = "20000101T000000Z"


def _grid(start: float, stop: float) -> np.ndarray:
    return np.linspace(start, stop, int(np.ceil((stop - start) / SPACING_M)) + 1)


def mathematical_room() -> tuple[np.ndarray, np.ndarray, dict]:
    """Return XYZ/RGB for a fictional two-bay room with a gable roof."""
    blocks, colours = [], []
    primitive_counts = {}

    def add(name, points, colour):
        points = np.asarray(points, dtype=np.float64)
        blocks.append(points)
        colours.append(np.tile(colour, (len(points), 1)).astype(np.uint8))
        primitive_counts[name] = len(points)

    def horizontal(name, z, colour):
        x, y = np.meshgrid(_grid(-THICKNESS_M, WIDTH_M + THICKNESS_M),
                           _grid(-THICKNESS_M, DEPTH_M + THICKNESS_M))
        add(name, np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z))), colour)

    def wall_x(name, y, colour, window=False):
        x, z = np.meshgrid(_grid(-THICKNESS_M, WIDTH_M + THICKNESS_M), _grid(0, EAVE_M))
        keep = np.ones(x.shape, dtype=bool)
        if window:
            keep &= ~((x > 1.2) & (x < 2.4) & (z > 0.9) & (z < 2.1))
        add(name, np.column_stack((x[keep], np.full(keep.sum(), y), z[keep])), colour)

    def wall_y(name, x, colour, doorway=False):
        y, z = np.meshgrid(_grid(-THICKNESS_M, DEPTH_M + THICKNESS_M), _grid(0, EAVE_M))
        keep = np.ones(y.shape, dtype=bool)
        if doorway:
            keep &= ~((y > 2.0) & (y < 3.0) & (z < 2.1))
        add(name, np.column_stack((np.full(keep.sum(), x), y[keep], z[keep])), colour)

    horizontal("floor", 0, (164, 174, 182))
    for y, name in [(-THICKNESS_M, "front_outer"), (0, "front_inner"),
                    (DEPTH_M, "rear_inner"), (DEPTH_M + THICKNESS_M, "rear_outer")]:
        wall_x(name, y, (224, 221, 209), window="front" in name)
    for x, name in [(-THICKNESS_M, "left_outer"), (0, "left_inner"),
                    (WIDTH_M, "right_inner"), (WIDTH_M + THICKNESS_M, "right_outer")]:
        wall_y(name, x, (200, 206, 210))
    for x, name in [(WIDTH_M / 2 - THICKNESS_M / 2, "partition_left"),
                    (WIDTH_M / 2 + THICKNESS_M / 2, "partition_right")]:
        wall_y(name, x, (200, 16, 46), doorway=True)
    for y0, y1, name in [(0, DEPTH_M / 2, "roof_front"),
                         (DEPTH_M / 2, DEPTH_M, "roof_rear")]:
        x, y = np.meshgrid(_grid(-THICKNESS_M, WIDTH_M + THICKNESS_M), _grid(y0, y1))
        z = EAVE_M + (RIDGE_M - EAVE_M) * (1 - np.abs(y - DEPTH_M / 2) / (DEPTH_M / 2))
        add(name, np.column_stack((x.ravel(), y.ravel(), z.ravel())), (110, 139, 159))

    xyz = np.vstack(blocks)
    rgb = np.vstack(colours)
    rng = np.random.default_rng(SEED)
    noise = rng.normal(0.0, NOISE_SIGMA_M, xyz.shape)
    theta = np.deg2rad(YAW_DEG)
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    xyz = (xyz + noise) @ rotation.T
    known = {
        "synthetic": True, "generator": "mathematical-two-bay-room-v1", "seed": SEED,
        "interior_width_m": WIDTH_M, "interior_depth_m": DEPTH_M,
        "wall_thickness_m": THICKNESS_M, "eave_m": EAVE_M, "ridge_m": RIDGE_M,
        "partition_centre_x_m": WIDTH_M / 2,
        "doorway": {"y_m": [2.0, 3.0], "height_m": 2.1},
        "window": {"x_m": [1.2, 2.4], "z_m": [0.9, 2.1]},
        "sampling_spacing_m": SPACING_M, "gaussian_noise_sigma_m": NOISE_SIGMA_M,
        "source_yaw_deg": YAW_DEG, "point_count": len(xyz),
        "primitive_point_counts": primitive_counts,
        "added_noise_xyz_rms_m": round(float(np.sqrt(np.mean(np.sum(noise ** 2, axis=1)))), 8),
    }
    return xyz, rgb, known


def generate_inputs(output: str | Path) -> tuple[Path, Path, dict]:
    from viewer.project_model import ProjectModel, ScanEntry, ScanLayer

    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    points, rgb, known = mathematical_room()
    cloud = root / "synthetic-room.ply"
    with cloud.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\ncomment mathematical synthetic two-bay room\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("property float conf_mean\nproperty int n_views\nend_header\n")
        for point, colour in zip(points, rgb):
            stream.write(" ".join(f"{value:.6f}" for value in point))
            stream.write(" " + " ".join(str(int(value)) for value in colour) + " 2.0 3\n")
    model = ProjectModel(
        project_id="synthetic-survey-v1", name="Synthetic two-bay room",
        scans=[ScanEntry(
            scan_id="synthetic-room", rail_order=0, source_path=str(cloud),
            layers=[ScanLayer(key="rgb/full", path=str(cloud), role="PRIMARY_SCAN_POINTS",
                             point_count=len(points), visible=True,
                             metadata={"source_axis": "CAD_Z_up", "synthetic": True})],
            metadata={"synthetic": True},
        )], metadata={"synthetic": True, "generator": known["generator"]},
    )
    project = model.save(root / "demo.ccapoints.json")
    (root / "geometry.json").write_text(json.dumps(known, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return cloud, project, known


def run_demo(output: str | Path, *, sheets: bool = True) -> tuple[Path, dict]:
    """Run the same survey service called by the application, without substituting results."""
    from export.survey_pack import SurveyPackConfig
    from geometry.survey_frame import SurveyFrameConfig
    from geometry.survey_plan import SurveyPlanConfig
    from viewer.project_model import ProjectModel
    from viewer.survey_service import run_survey_pack
    import ezdxf

    root = Path(output).resolve()
    cloud, project, known = generate_inputs(root / "input")
    before = hashlib.sha256(cloud.read_bytes()).hexdigest()
    model = ProjectModel.open(project)
    config = SurveyPackConfig(frame=SurveyFrameConfig(front_side="keep"),
                              plan=SurveyPlanConfig(wall_min_length_m=0.5), sheets=sheets,
                              dxf=True, report=True, source_axis="CAD_Z_up")
    summary = run_survey_pack(model.scans[0], project_name=model.name, config=config,
                              stamp=DEMO_STAMP, output_root=root / "survey",
                              progress=lambda message: print(message, flush=True))
    manifest = json.loads(summary.manifest_path.read_text(encoding="ascii"))
    result = summary.result
    after = hashlib.sha256(cloud.read_bytes()).hexdigest()
    document = ezdxf.readfile(result.dxf_path)
    audit = document.audit()
    if before != after or result.point_count != known["point_count"] or audit.has_errors:
        raise RuntimeError("Synthetic survey did not preserve the input or pass the DXF audit")
    extents = result.frame.extents_m
    measured = {axis: extents[axis][1] - extents[axis][0] for axis in "xyz"}
    expected = {"x": WIDTH_M + 2 * THICKNESS_M,
                "y": DEPTH_M + 2 * THICKNESS_M, "z": RIDGE_M}
    errors = {axis: round(measured[axis] - expected[axis], 6) for axis in "xyz"}
    within_tolerance = all(abs(error) < 0.04 for error in errors.values())
    if not within_tolerance:
        raise RuntimeError(f"Synthetic frame dimensions exceeded the 40 mm demo tolerance: {errors}")
    structures = json.loads((result.output_dir / "structures.json").read_text(encoding="ascii"))["structures"]
    nominal_reference = {"width_m": WIDTH_M / 2, "depth_m": DEPTH_M,
                         "eave_z_m": EAVE_M, "ridge_z_m": RIDGE_M}
    structure_checks = [{
        "id": structure["id"], "confidence": structure["confidence"],
        "computed": {key: structure[key] for key in nominal_reference},
        "nominal_centreline_reference": nominal_reference,
        "difference_from_nominal_m": {
            key: None if structure[key] is None else round(structure[key] - value, 6)
            for key, value in nominal_reference.items()
        },
    } for structure in structures]
    payload = {
        "schema": "cca.synthetic-survey-demo.v1", "synthetic": True,
        "provenance": "Mathematical geometry only; no real scan, customer, site or project input.",
        "generator": "examples/synthetic_survey.py", "known_geometry": known,
        "pipeline": "viewer.survey_service.run_survey_pack -> export.survey_pack.build_survey_pack",
        "input": {"cloud": "input/synthetic-room.ply", "project": "input/demo.ccapoints.json",
                  "sha256_before": before, "sha256_after": after,
                  "raw_unchanged": before == after, "point_count": result.point_count},
        "survey": {
            "config_overrides": {"front_side": "keep", "wall_min_length_m": 0.5,
                                 "source_axis": "CAD_Z_up"},
            "walls": result.wall_count, "cross_walls": result.cross_wall_count,
            "structures": result.structure_count, "partitions": result.partition_count,
            "confidence_counts": result.confidence_counts,
            "hand_measure_items": list(result.hand_measure),
            "ground_sigma_mm": round(result.frame.ground_sigma_m * 1000, 5),
            "ground_tilt_deg": round(result.frame.ground_tilt_deg, 6),
            "yaw_correction_deg": round(result.frame.yaw_deg, 6),
            "effective_horizontal_rotation_deg": round(float(np.rad2deg(np.arctan2(
                result.frame.rotation()[1, 0], result.frame.rotation()[0, 0]))), 6),
            "derived_structure_checks": structure_checks,
            "expected_extents_m": expected,
            "computed_extents_m": {axis: round(value, 6) for axis, value in measured.items()},
            "extent_error_m": errors, "demo_tolerance_m": 0.04,
            "extents_within_demo_tolerance": within_tolerance,
            "frame": manifest["frame"],
        },
        "dxf": {**manifest["dxf"], "path": result.dxf_path.relative_to(root).as_posix(),
                "reopened_audit_errors": len(audit.errors),
                "entities": len(document.modelspace())},
        "sheets": [path.relative_to(root).as_posix() for path in result.sheet_paths],
        "output_dir": result.output_dir.relative_to(root).as_posix(),
        "limits": [
            "This controlled synthetic test is not a survey accuracy claim.",
            "The pipeline returns uncalibrated survey drafts and confidence labels.",
            "Detected features may split, merge or miss sampled geometry.",
            "Opening locations are generator annotations, not validated opening detections.",
            "Nominal bay references use partition centrelines; the detector may use a wall face.",
            "A HIGH support label does not certify measurement accuracy or remove geometric bias.",
        ],
    }
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return summary_path, payload


def publish_demo_outputs(generated: str | Path, destination: str | Path) -> Path:
    """Prepare reviewable synthetic text/CAD evidence, excluding generated input files."""
    import ezdxf

    root = Path(generated).resolve()
    target = Path(destination).resolve()
    target.mkdir(parents=True, exist_ok=True)
    payload = json.loads((root / "summary.json").read_text(encoding="ascii"))
    if payload.get("synthetic") is not True:
        raise ValueError("Only the generated synthetic demo can be prepared for publication")
    source_dxf = root / payload["dxf"]["path"]
    document = ezdxf.readfile(source_dxf)
    entity_count = len(document.modelspace())
    metadata = document.ezdxf_metadata()
    for key in ("CREATED_BY_EZDXF", "WRITTEN_BY_EZDXF"):
        metadata[key] = f"ezdxf {ezdxf.__version__} / mathematical synthetic demo"
    metadata["CCA_PROVENANCE"] = "Mathematical geometry only; no real site, scan or customer data"
    document.header["$PROJECTNAME"] = "Synthetic two-bay room"
    document.header["$LASTSAVEDBY"] = "ezdxf"
    for field in ("$TDCREATE", "$TDUCREATE", "$TDUPDATE", "$TDUUPDATE"):
        document.header[field] = 2451545.0
    destination_dxf = target / "synthetic-survey.dxf"
    previous = ezdxf.options.write_fixed_meta_data_for_testing
    try:
        ezdxf.options.write_fixed_meta_data_for_testing = True
        document.saveas(destination_dxf)
    finally:
        ezdxf.options.write_fixed_meta_data_for_testing = previous
    # ezdxf regenerates WRITTEN_BY_EZDXF while saving. The fixed write mode
    # makes that value, dates and file GUIDs machine-independent.
    reopened = ezdxf.readfile(destination_dxf)
    audit = reopened.audit()
    if audit.has_errors or len(reopened.modelspace()) != entity_count:
        raise RuntimeError("Publication metadata normalization changed or invalidated the CAD output")
    text = destination_dxf.read_text(encoding="utf-8")
    if str(root) in text or str(target) in text:
        raise RuntimeError("Generated CAD evidence contains an absolute output path")
    tags = text.splitlines()
    stored_dates = {field: float(tags[tags.index(field) + 2]) for field in (
        "$TDCREATE", "$TDUCREATE", "$TDUPDATE", "$TDUUPDATE")}
    if any(value != 2451545.0 for value in stored_dates.values()):
        raise RuntimeError("Published CAD metadata does not contain fixed creation/update dates")
    payload["dxf"]["path"] = destination_dxf.name
    payload["dxf"]["sha256"] = hashlib.sha256(destination_dxf.read_bytes()).hexdigest()
    payload["dxf"]["metadata_normalized"] = True
    payload["dxf"]["geometry_entity_count_unchanged"] = True
    payload["dxf"]["serialized_header_dates"] = stored_dates
    payload["generated_output_dir"] = payload.pop("output_dir")
    payload["outputs"] = {name.removesuffix(".json"): name for name in (
        "walls.json", "partitions.json", "structures.json", "facade_profile.json")}
    (target / "summary.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="ascii")
    for name in ("walls.json", "partitions.json", "structures.json", "facade_profile.json"):
        shutil.copyfile(root / payload["generated_output_dir"] / name, target / name)
    return destination_dxf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("generated/synthetic-survey"))
    parser.add_argument("--no-sheets", action="store_true", help="Skip application PNG sheets; keep actual CAD and metrics")
    parser.add_argument("--input-only", action="store_true", help="Generate fictional input without running the survey")
    parser.add_argument("--publish-dir", type=Path, help="Prepare synthetic JSON/CAD evidence for review in this directory")
    args = parser.parse_args()
    if args.input_only:
        cloud, project, known = generate_inputs(args.output / "input")
        print(f"SYNTHETIC INPUT READY / {known['point_count']} points / {cloud.name} / {project.name}")
    else:
        summary_path, payload = run_demo(args.output, sheets=not args.no_sheets)
        print(f"SYNTHETIC SURVEY VERIFIED / {payload['survey']['walls']} walls / "
              f"{payload['dxf']['entities']} CAD entities / {summary_path.name}")
        if args.publish_dir:
            published = publish_demo_outputs(args.output, args.publish_dir)
            print(f"SYNTHETIC PUBLICATION EVIDENCE READY / {published.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
