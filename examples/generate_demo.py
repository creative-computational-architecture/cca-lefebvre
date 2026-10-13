"""Generate a mathematical point set; no external input is read."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from viewer.project_model import ProjectModel, ScanEntry, ScanLayer


def generate(output: str | Path) -> tuple[Path, Path]:
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(-1.0, 1.0, 21)
    a, b = np.meshgrid(grid, grid)
    planes = [
        np.column_stack((a.ravel(), b.ravel(), np.zeros(a.size))),
        np.column_stack((a.ravel(), np.full(a.size, -1.0), b.ravel() + 1.0)),
        np.column_stack((np.full(a.size, -1.0), a.ravel(), b.ravel() + 1.0)),
    ]
    points = np.vstack(planes)
    cloud = root / "synthetic-room.ply"
    with cloud.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\ncomment mathematical synthetic example\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("property float conf_mean\nproperty int n_views\nend_header\n")
        for x, y, z in points:
            stream.write(f"{x:.6f} {y:.6f} {z:.6f} 200 16 46 2.0 3\n")
    model = ProjectModel(
        project_id="synthetic-demo-v1", name="Mathematical synthetic room",
        scans=[ScanEntry(
            scan_id="synthetic-room", rail_order=0, source_path=str(cloud),
            layers=[ScanLayer(key="rgb/full", path=str(cloud),
                role="PRIMARY_SCAN_POINTS", point_count=len(points), visible=True)],
            metadata={"synthetic": True},
        )], metadata={"synthetic": True, "generator": "mathematical-planes-v1"},
    )
    project = model.save(root / "demo.ccapoints.json")
    return cloud, project


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("generated"))
    args = parser.parse_args()
    cloud, project = generate(args.output)
    print(f"Created synthetic data: {cloud.name}, {project.name}")
