# Source release checks

Release candidate: 0.1.0, Windows 64-bit, Python 3.12.13.

This file records the reviewed publication boundary. Native tests, real scan
processing and package builds use a separate validation environment. The
application's original development checkout is not modified.

## Reviewed source boundary

- Independent Python runtime with SVG source assets and generic ReCap scripts.
- Newly written installation/build documents and synthetic generator/tests.
- Reviewed native-window, survey-analysis and CAD drawing images.
- Mathematical survey outputs in `docs/demo`, including an audited synthetic DXF.
- Full GPL-3.0-or-later license and upstream dependency references/notices.
- Fresh Git history; no imported branches, objects, authors or tags.

## Excluded material

Customer/project presets, real scans, point clouds, real CAD/BIM files, photographs,
correspondence, old documents, stored private placement/measurement results,
machine configuration, internal addresses, credentials, environment files,
binaries, caches and development history are excluded.

Only selected geometric views of the studio's own office are published. The
office input, acquisition panoramas, camera poses and real CAD/JSON outputs
remain private. The displayed region is a bounded upper-room interior cutaway;
its clipping boundaries do not describe the full office footprint. Customer
projects and other floors or street imagery are not part of the demonstration.

Source strings, comments, docstrings, document links, SVG/JSON/DXF metadata,
packaging manifests and source archives are inspected. Automatic checks search
for known private identifiers, fixed machine paths, private hosts, credential
material and unexpected file types. A separate reviewer inspects the source.
Those checks are combined with a curated file selection. Publication images
are opened for visual review, checked for embedded metadata and approved by
content hash. PNG files retain binary Git attributes.

## Validation results

- The original 93 Python files compile; all added Python files also parse.
- Eight synthetic tests pass: loading/counts, deterministic generation, source
  hash preservation, custom point-field preservation, derived transforms,
  refusal to overwrite source geometry, source SVG assets and opt-in startup.
- An empty project contract reports zero scans, drawings and accepted alignments.
- An empty Qt/VTK window opens in a native Windows graphics session.
- A separate virtual environment installs the pinned direct dependencies.
- A source archive and pure-Python wheel build successfully.
- The installed wheel imports all 86 non-package runtime modules from its own
  installed location. SVG and ReCap source assets are included.
- The installed native window displays one synthetic scan and 1,323 points;
  screenshot capture completes with exit code zero. The local image is excluded.
- Curated source and metadata pass automatic privacy checks and an independent
  source inspection without unresolved findings.
- All 122 fingerprinted original source/asset/script files retain their hashes.
- The mathematical survey processes 176,180 points without changing its input.
  It produces 10 wall segments, 2 structure candidates and a 1,832-entity DXF
  with 9 dimensions. The reopened DXF has zero audit errors and centimetre units.
- Synthetic DXF dates and GUIDs are normalized; numerical geometry is unchanged.
- Native synthetic projection and plan-analysis panels are captured from the
  application. Their interactive element counts are separate from the automatic
  survey-pack wall count; opening detection is not established by this example.

## Limits

The initial 1,323-point example verifies basic loading/rendering and derived
transforms. The larger mathematical benchmark verifies source preservation,
known-envelope regression and CAD integrity. Its inferred dimensions include
reported biases, including 3.91/4.09 m bays against nominal 4.00 m spans.
Fit residuals and support labels do not establish real-world survey accuracy.

The own-office walkthrough demonstrates processing of a selected real interior
region. It has no independent control survey. It does not validate every large
production scan, curved walls, occluded features or general sensor calibration.
ReCap/Revit indexing, local model inference and optional control integrations
remain outside these checks.

No executable, dependency DLL, wheel, real point cloud or real CAD file is
released. Reviewed publication images and the mathematical DXF are included.
