# Build and verification

This is a source-only Python distribution. It does not compile or distribute
CloudCompare, Qt, VTK or other external projects. The dependencies supply their
own platform wheels; their licenses and copyright notices remain in force.

## Reproduce the tested runtime

Use Python 3.12 on 64-bit Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
.\.venv\Scripts\python.exe -m pip install --no-deps .
.\.venv\Scripts\python.exe -m pip install pytest build
```

`requirements-tested.txt` pins direct runtime dependencies. Their transitive
dependencies are resolved by the package manager; it is not a full hash lock.
The public package does not depend on a development checkout or saved project.

## Checks

```powershell
.\.venv\Scripts\python.exe -m compileall -q viewer geometry ingest field export
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m viewer.desktop_app --smoke
.\.venv\Scripts\python.exe examples/generate_demo.py --output generated
.\.venv\Scripts\python.exe -m viewer.desktop_app --project generated/demo.ccapoints.json --smoke
```

The synthetic checks exercise project serialization, point counts, property
preservation, derived transforms and refusal to overwrite source geometry.
`--smoke` checks project loading and reports metadata; it does not open the GUI.
To verify the native Qt/VTK window in a graphics-capable Windows session:

```powershell
.\.venv\Scripts\python.exe -m viewer.desktop_app --project generated/demo.ccapoints.json --calculate-points --frame-scans --screenshot generated/gui-check.png
```

Inspect the captured window and rendered point count. The application exits
after capture. The image is a local verification artifact ignored by Git.
An offscreen Qt test alone does not verify a working VTK viewport.

## Package

```powershell
.\.venv\Scripts\python.exe -m build
```

The wheel and source archive are local verification artifacts. This repository
publishes source only; no executable installer or prebuilt wheel is released.
Inspect package manifests before distributing a future binary. Preserve
dependency notices and the Qt/LGPL replacement and corresponding-source
requirements where they apply.

The optional ReCap helper scripts use an existing Autodesk installation and
license selected at runtime. They are included as source and checked for
privacy and packaging, but ReCap/Revit interoperability is not verified by
this release's synthetic checks. No Autodesk executable is distributed.

## Local model configuration

The planner is optional and does not download models or start services.
Set `CCA_LEFEBVRE_LOCAL_MODEL_URL` to an explicitly chosen loopback endpoint.
Leave it unset to disable the planner. Credentials and redirects are rejected.
Set `CCA_LEFEBVRE_PROJECT` only to opt into opening a chosen local project.
Application of an
alignment remains a separate user action. No source data is transmitted by
the default startup or synthetic example.

## Release boundary

The release contains source, SVG UI assets, documentation, license texts,
synthetic generators/tests and reviewed publication images. The own-office
demonstration publishes selected geometry views and analysis figures; its raw
scan and real CAD/measurement files remain private. The downloadable DXF and
JSON examples in `docs/demo` are generated from mathematical geometry.
Environment files, real project manifests, point clouds, customer CAD/BIM,
photographs, correspondence, logs, caches, machine configuration and previous
development history are excluded. See [RELEASE_CHECKS.md](RELEASE_CHECKS.md).
