# Licensing and third-party notices

## CCA Lefebvre source license

Copyright (c) 2026 Creative Computational Architecture - Caglar Celik Architects (CCA)

CCA-owned source code in this repository is distributed under the GNU General
Public License, version 3 or, at your option, any later version
(`GPL-3.0-or-later`). The complete GPL version 3 text is provided in [LICENSE](LICENSE)
and [COPYING](COPYING). The application-specific grant is also recorded in
[licenses/CCA-NOTICE.txt](licenses/CCA-NOTICE.txt).

Existing third-party copyright, license, and attribution notices remain applicable
to the material they identify. The CCA copyright statement does not claim ownership
of Python, Qt, third-party libraries, or their algorithms.

## CloudCompare boundary

This release is an independent native Python implementation. It includes no
CloudCompare source code, icons, or binaries. A separately installed CloudCompare
application remains a separate program under its own original license; this
repository does not replace or relicense it.

CloudCompare is acknowledged as a reference for point-cloud workflows. Its source
license notice grants GPL version 2 or later and preserves the original EDF R&D /
TELECOM ParisTech attribution. See the [official source license notice](https://github.com/CloudCompare/CloudCompare/blob/master/license.txt).
That acknowledgment does not imply that this repository is a CloudCompare fork or
that CloudCompare's authors endorse CCA Lefebvre.

## Separately installed Python dependencies

The following packages are imported by the released source or its optional tools.
They are installed separately through Python packaging. Their source trees,
wheels, DLLs, and native libraries are not bundled in this source repository.
The license references below identify the upstream projects; they do not change
those projects' licenses. Reference license texts are supplied in [licenses/](licenses/).

| Python package | Role | Upstream license | Primary reference |
|---|---|---|---|
| `numpy` | Arrays and numerical geometry | BSD-3-Clause | [NumPy LICENSE](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| `scipy` | Spatial, scientific, and numerical processing | BSD-3-Clause | [SciPy LICENSE](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |
| `pandas` | Tabular scan and capture metadata | BSD-3-Clause | [pandas LICENSE](https://github.com/pandas-dev/pandas/blob/main/LICENSE) |
| `pyarrow` | Arrow and Parquet data access | Apache-2.0, with additional upstream notices | [Arrow LICENSE](https://github.com/apache/arrow/blob/main/LICENSE.txt), [NOTICE](https://github.com/apache/arrow/blob/main/NOTICE.txt) |
| `opencv-contrib-python` | Image processing and marker support | MIT for Python packaging; Apache-2.0 for current OpenCV; bundled-library terms depend on the installed wheel | [Wrapper LICENSE](https://github.com/opencv/opencv-python/blob/4.x/LICENSE.txt), [OpenCV LICENSE](https://github.com/opencv/opencv/blob/4.x/LICENSE), [wheel third-party notices](https://github.com/opencv/opencv-python/blob/4.x/LICENSE-3RD-PARTY.txt) |
| `imageio` | Image input and output | BSD-2-Clause | [imageio LICENSE](https://github.com/imageio/imageio/blob/master/LICENSE) |
| `OpenEXR` | EXR depth/image access | BSD-3-Clause | [OpenEXR LICENSE](https://github.com/AcademySoftwareFoundation/openexr/blob/main/LICENSE.md) |
| `open3d` | Point-cloud geometry and registration | MIT, with its own bundled-library notices | [Open3D LICENSE](https://github.com/isl-org/Open3D/blob/main/LICENSE) |
| `ezdxf` | DXF geometry export | MIT | [ezdxf LICENSE](https://github.com/mozman/ezdxf/blob/master/LICENSE) |
| `PySide6` | Qt for Python native interface | Community Edition: LGPLv3/GPLv3; commercial licensing is a separate alternative | [Official Qt for Python licensing](https://doc.qt.io/qtforpython-6/commercial/index.html), [Qt licensing](https://doc.qt.io/qt-6/licensing.html) |
| `pyvista` | Python interface to VTK geometry/rendering | MIT | [PyVista LICENSE](https://github.com/pyvista/pyvista/blob/main/LICENSE) |
| `pyvistaqt` | Qt/VTK viewport bridge | MIT | [PyVistaQt LICENSE](https://github.com/pyvista/pyvistaqt/blob/main/LICENSE) |
| `vtk` | Geometry and rendering engine | BSD-3-Clause, with additional component notices | [VTK Copyright](https://github.com/Kitware/VTK/blob/master/Copyright.txt) |
| `pye57` | E57 scan input/output | MIT; the separately installed libE57Format component uses BSL-1.0 | [pye57 LICENSE](https://github.com/davidcaron/pye57/blob/master/LICENSE), [libE57Format LICENSE](https://github.com/asmaloney/libE57Format/blob/master/LICENSE.md) |
| `psutil` | Process and resource information | BSD-3-Clause | [psutil LICENSE](https://github.com/giampaolo/psutil/blob/master/LICENSE) |
| `Pillow` | Raster images | MIT-CMU | [Pillow LICENSE](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| `matplotlib` | Plotting and scientific output | Matplotlib license, based on the PSF license; additional bundled notices apply | [Matplotlib LICENSE](https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE) |
| `reportlab` | PDF document output | BSD-3-Clause for the open-source toolkit | [Official ReportLab licensing FAQ](https://docs.reportlab.com/developerfaqs/#13-licensing), [upstream license file](https://hg.reportlab.com/hg-public/reportlab/file/tip/src/reportlab/license.txt) |
| `mcp` (optional) | Local Model Context Protocol bridge | MIT | [Official Python SDK LICENSE](https://github.com/modelcontextprotocol/python-sdk/blob/main/LICENSE) |

The table covers the direct package names used by this source. Packages may depend
on other libraries or contain bundled components; the license files and notices
shipped with the exact installed distributions remain authoritative for those
components. Dependency version requirements describe the supported environment;
they are not a frozen binary license inventory.

## Qt and future binary packaging

The documented installation uses the community `PySide6` distribution. Qt offers
LGPLv3/GPLv3 and commercial alternatives; specific Qt modules can be GPL-only in
the open-source distribution. Use the terms associated with the actual Qt modules
and package versions installed. A commercial Qt license is not claimed by this
release.

A future installer that redistributes Qt or any other dependency must include
the notices and complete license texts for the exact files it ships. Preserve
the applicable LGPL library replacement/relinking mechanism and provide the
corresponding source and installation information required by the license path
chosen for that binary distribution. The present source-only release does not
include an installer or redistribute dependency binaries.

## Reference copies

The files under `licenses/` are upstream license reference copies retrieved on
2026-10-13. Their URLs and SHA-256 hashes are recorded in
[licenses/source-manifest.json](licenses/source-manifest.json). These copies are
not a claim that the referenced upstream code is vendored here or that a complete
inventory of any future bundled binary has been performed.

The original Pillow license contains non-ASCII copyright symbols. Its unmodified
upstream license is linked above rather than reproducing an altered text here.
Arrow's full composite license and OpenCV's full wheel notices are likewise
linked to the upstream originals; their exact package versions supply the
relevant complete notices when installed or redistributed.
