# Reproducible synthetic survey

These files are produced by Lefebvre's actual survey service from a mathematical
two-bay room. No real building, scan, customer project or CAD drawing was used.
The input is generated locally and is not committed.

```powershell
python examples/synthetic_survey.py --output generated/synthetic-survey --publish-dir docs/demo
```

The command calls `viewer.survey_service.run_survey_pack`, which calls
`export.survey_pack.build_survey_pack`. The application uses this same service.
It creates survey sheets, JSON candidates, a report and a centimetre DXF through
`export.survey_dxf.export_survey_dxf`. The CAD file below is that export with
only its descriptive metadata normalized for publication.

| Known input | Value |
|---|---:|
| Interior width / depth | 8.00 / 5.00 m |
| Eave / ridge | 3.00 / 3.80 m |
| Wall thickness / partition centreline | 0.18 / 4.00 m |
| Nominal sampling / added Gaussian noise sigma | 0.04 / 0.0015 m |
| Artificial source yaw / random seed | 17 degrees / 2026 |
| Generated vertices | 176,180 |

The demo explicitly uses `wall_min_length_m=0.5`, `front_side="keep"` and
`source_axis="CAD_Z_up"`; other survey settings retain their default values.

| Actual output | Value |
|---|---:|
| Wall candidates / low confidence walls | 10 / 2 |
| Cross walls / partition positions / structure candidates | 3 / 3 / 2 |
| DXF modelspace entities | 1,832 |
| Dimension / section / front elevation entities | 9 / 6 / 6 |
| DXF units (`$INSUNITS`) | Centimetres (5) |
| DXF audit errors, including reopening | 0 |
| Effective horizontal frame rotation | -17.00 degrees |
| Computed ground fit sigma | See `summary.json` |
| Raw input before/after SHA-256 | Identical; see `summary.json` |

## Files

- [synthetic-survey.dxf](synthetic-survey.dxf): actual plan, elevations, sections,
  dimensions and confidence layers, in one CAD modelspace.
- [summary.json](summary.json): input provenance, exact settings, counts,
  hashes, frame and comparisons against the known geometry.
- [walls.json](walls.json), [partitions.json](partitions.json),
  [structures.json](structures.json), [facade_profile.json](facade_profile.json):
  unedited numerical candidates from the application's survey pipeline.

Generated PNG sheets stay in the chosen output directory. Reviewed graphical
previews appear in `docs/media`; the generator is the source for reproduction.
The fixed `20000101T000000Z` output folder identifier is a reproducibility marker,
not a recording date. CAD dates and file GUIDs use ezdxf's fixed metadata mode;
the creator marker identifies this mathematical synthetic example.

## What the checks demonstrate

The source point count and bytes are preserved. The rigid survey frame retains
scale 1 and determinant +1. Its overall dimensions agree with the synthetic
8.36 x 5.36 x 3.80 m exterior envelope within a 40 mm demo tolerance; those
differences include artificial sampling and noise, rather than a real-world
survey calibration. DXF geometry remains unchanged when metadata is normalized.

The actual bay widths are 3.91 and 4.09 m against two nominal 4.00 m centreline
spans. The computed structure depth is 5.20 m against a known 5.00 m interior,
and the computed eave is 3.09 m against 3.00 m. These biases are reported in
`summary.json`. A HIGH support label is not an accuracy certificate. The results
remain **SURVEY DRAFT / UNCALIBRATED** and need operator review. The doorway and
window are generator annotations; this demo does not validate opening detection.
