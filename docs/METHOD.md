# From point evidence to a survey draft

[README](../README.md) | [Synthetic reproduction](demo/README.md) | [Exact results](demo/summary.json)

Lefebvre has an automatic survey-pack pipeline and an interactive projection/plan workflow. They share point evidence but are different algorithms. Candidate counts from one must not be described as results of the other. Neither workflow establishes calibrated survey accuracy.

## Input and frame

The desktop invokes `viewer.survey_service.run_survey_pack`, which resolves a visible PLY layer and calls `export.survey_pack.build_survey_pack`. The service carries the explicitly chosen placement transform and source-axis information into the pack. The underlying pack also includes an E57 conversion route; the benchmark here exercises PLY.

`geometry.survey_frame.normalize_source_points` supports an explicit signed up axis or an automatic probe. The probe scores ground support, points above the ground and fit noise. An automatic result can be rejected when ground evidence is insufficient. The benchmark explicitly selects `CAD_Z_up`.

The frame estimator selects low points, uses RANSAC to propose a ground plane and refines its normal with SVD. It levels the cloud, searches 0-90 degrees for sharp projected histograms in a wall-height band, refines that direction and can rotate the long axis by 90 degrees. Front-side selection is a heuristic based on ground support unless explicitly overridden. Translation moves the horizontal minima to the survey origin. The resulting matrix is a proper rigid transform: scale 1, determinant +1.

| `SurveyFrameConfig` field | Default |
|---|---:|
| `ground_percentile` | 15 |
| `ground_inlier_m` | 0.04 m |
| `ransac_iterations` / `seed` | 400 / 0 |
| `max_tilt_deg` | 60 degrees |
| `wall_band_m` | 0.9-1.4 m |
| `yaw_step_deg` / `yaw_fine_step_deg` | 0.25 / 0.02 degrees |
| `projection_bin_m` | 0.02 m |
| `sample_points` | 400,000 |
| `ground_test_z_m` | 0.1 m |
| `front_side` | `auto` |

These are computational thresholds. A 0.04 m inlier threshold or 0.02 m raster does not establish a corresponding measurement accuracy.

## Occupancy and supported geometry

`geometry.survey_plan.build_slices` builds XY occupancy rasters for seven height bands and a roof-height projection. `vectorize_walls` finds histogram peaks in four wall-height bands, examines the point runs around each peak, and merges compatible segments. This method favours axis-aligned, sufficiently supported geometry in the estimated survey frame.

| `SurveyPlanConfig` field | Default |
|---|---|
| `slice_bands_m` | 0.25-0.45; 0.55-0.75; 0.95-1.25; 1.45-1.65; 1.95-2.15; 2.45-2.65; 2.95-3.15 m |
| `raster_m` | 0.02 m |
| `wall_bands_m` | 0.6-0.8; 0.95-1.25; 1.5-1.7; 2.0-2.2 m |
| `wall_histogram_bin_m` | 0.02 m |
| `wall_peak_share` / `wall_peak_min_count` | 0.02 / 30 |
| `wall_corridor_m` | 0.04 m |
| `wall_run_bin_m` / `wall_run_min_count` | 0.05 m / 4 |
| `wall_gap_m` | 0.25 m |
| `wall_min_length_m` | 0.30 m |
| `wall_merge_tolerance_m` | 0.05 m |
| `cross_wall_min_points` | 900 |
| `cross_wall_extent_bin_m` / `cross_wall_extent_min_count` | 0.1 m / 10 |
| `cross_wall_dedupe_m` | 0.3 m |
| `structure_min_width_m` | 0.6 m |
| `roof_min_z_m` | 0.5 m |
| `front_margin_m` / `profile_bin_m` | 5.0 / 0.1 m |

Wall labels encode repeated support, rather than probabilities or surveyed tolerances. A merged wall is HIGH when it is supported in at least three bands and its position spread is below 0.02 m; otherwise support in at least two bands gives MEDIUM, and the remaining case gives LOW. The pipeline also derives cross walls, partition positions, structure candidates and a facade profile. Occlusion, clutter, sparse coverage and incorrect orientation can split, merge or miss features.

## Sheets and CAD

`SurveyPackConfig` defaults to centimetre labels and DXF units, 150 DPI sheets, and enabled sheet, DXF and report outputs. Its source-axis default is `auto`; it accepts an explicit rigid pre-transform. `export.survey_dxf.export_survey_dxf` creates drawing entities and calls ezdxf's audit.

The benchmark includes these layers:

| Layer | Entities | Role |
|---|---:|---|
| `CCA_SURVEY_SLICE` | 1,541 | Projected slice evidence |
| `CCA_SURVEY_ELEV_LIDAR` | 212 | Projected elevation evidence |
| `CCA_SURVEY_GRID` | 34 | Reference grid |
| `CCA_SURVEY_TEXT` | 10 | Labels |
| `CCA_SURVEY_DIM` | 9 | Dimension entities |
| `CCA_SURVEY_WALL` | 8 | Wall candidates outside the LOW label |
| `CCA_SURVEY_WALL_LOWCONF` | 2 | Low-support wall candidates |
| `CCA_SURVEY_ELEV_FRONT` | 6 | Derived front elevation |
| `CCA_SURVEY_SECTION` | 6 | Derived sections |
| `CCA_SURVEY_STRUCT` | 4 | Structure outlines |

The total is 1,832 modelspace entities. `$INSUNITS=5` means centimetres. Geometry is unchanged when the synthetic CAD file's descriptive publication metadata is normalized; the released file is reopened and audited. An audit with zero errors demonstrates file integrity, rather than dimensional correctness.

## Reproduce the controlled benchmark

After installation:

```powershell
.\.venv\Scripts\python.exe examples/synthetic_survey.py --output generated/survey-demo
.\.venv\Scripts\python.exe -m viewer.desktop_app --project generated/survey-demo/input/demo.ccapoints.json --calculate-points --frame-scans
```

The generator constructs a two-bay room with an 8.00 x 5.00 m interior, 0.18 m walls, a centre partition at x=4.00 m, a 3.00 m eave and a 3.80 m ridge. Nominal sampling is 0.04 m. Gaussian noise sigma is 0.0015 m per coordinate, the seed is 2026 and the source yaw is 17 degrees. There are 176,180 vertices.

The example overrides exactly three default settings: `source_axis="CAD_Z_up"`, `front_side="keep"` and `wall_min_length_m=0.5`. This matters when comparing the benchmark with a default desktop survey run. The generated project and point cloud remain local; only the reviewed numerical evidence, derived synthetic DXF and figures are published.

The [summary](demo/summary.json) reports unchanged input SHA-256, 176,180 processed points, scale 1 and determinant +1. The effective horizontal rotation is -17 degrees: a 73-degree yaw followed by the long-axis correction. It returns 10 wall candidates, 3 cross walls, 3 partition positions and 2 structure candidates.

| Quantity | Known reference | Calculated | Difference |
|---|---:|---:|---:|
| Exterior X extent | 8.360000 m | 8.372167 m | +0.012167 m |
| Exterior Y extent | 5.360000 m | 5.371415 m | +0.011415 m |
| Exterior Z extent | 3.800000 m | 3.810225 m | +0.010225 m |
| First / second nominal centreline span | 4.00 / 4.00 m | 3.91 / 4.09 m | -0.09 / +0.09 m |
| Interior depth | 5.00 m | 5.20 m | +0.20 m |
| Eave | 3.00 m | 3.09 m | +0.09 m |
| Ridge | 3.80 m | 3.79 m | -0.01 m |

The envelope comparison uses a 0.04 m demo tolerance. That tolerance applies to this fixture and does not validate all derived dimensions. The ground-plane fit sigma is 1.23722 mm; the added XYZ noise RMS is 2.59838 mm. The fit uses selected ground inliers, so its sigma is not a whole-cloud error estimate or an instrument specification. Centreline references and detected wall faces can also differ.

## Two demonstrations, two roles

The own-office images in the README document operation on an authorized studio capture. They use a bounded upper-room region and a cutaway view, so the illustrated boundaries describe that selected region rather than the full office footprint. They show reviewed geometric views and derived drawing previews; the original acquisition data and real CAD file are not distributed. There is no independent office control survey in this release.

The synthetic fixture permits a comparison with explicitly known geometry. It deliberately reports geometric biases alongside successful source preservation and CAD-file integrity. The interactive plan-analysis screenshot is a third view of that same synthetic input through a separate detector: its element types and count do not establish validated opening detection and must not replace the automatic-pack wall count.

Both workflows require operator review. Scale, source axis, frame direction, slice evidence, merged features and dimensions must be checked before a drawing is used. The current benchmark covers one mathematical geometry and does not validate curved walls, clutter, sensor calibration, large production scans or ReCap/Revit handoff.
