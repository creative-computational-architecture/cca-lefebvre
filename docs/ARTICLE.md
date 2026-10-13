# A drawing should keep its evidence

[Open the application walkthrough](../README.md) | [Read the method](METHOD.md)

A point cloud records a room differently from a drawing. Furniture interrupts surfaces. Some walls carry dense evidence; others disappear behind an object or fall outside the capture. The architect still needs a plan, a section and dimensions, but those drawings should explain how they were obtained.

CCA Lefebvre puts that transition in a native workspace. You can inspect the points, change the view, examine a placement and open the projection tools. The automatic survey pack proposes a coordinate frame and architectural geometry, then writes sheets, numerical candidates and CAD entities. The operator can compare the proposal with the capture instead of receiving an unexplained finished drawing.

## Start with our own office

The main demonstration uses a bounded upper-room region of the studio's own office. A cutaway geometric view shows that selected region in the actual application. Its boundaries describe the region, not the full office footprint. The analysis and drawing previews show the route towards a survey draft. Acquisition photographs, raw scan files and the office's real CAD export stay private.

That distinction also concerns what the example can prove. A real capture demonstrates the working route, with its uneven coverage and objects. Without an independently measured control survey, it cannot establish the accuracy of the resulting dimensions.

## Give the method a known question

The separate mathematical example supplies known geometry: two bays inside an 8.00 x 5.00 m room, with a centre partition and a pitched roof. The generator creates 176,180 points, adds a controlled rotation and noise, then calls the same survey service as the desktop application.

The input hash stays unchanged. The frame remains rigid, and the generated CAD file reopens without audit errors. Ten proposed wall segments retain their support labels: seven HIGH, one MEDIUM and two LOW. The released summary allows anyone to examine those observations and rerun the experiment.

There are also differences that a polished rendering could conceal. The two nominal 4.00 m spans become 3.91 and 4.09 m. A nominal 5.00 m interior depth becomes 5.20 m. The calculated eave is 3.09 m against 3.00 m. These results are reported alongside the successful file checks. Support across several bands can make a candidate plausible while its interpretation remains biased.

## Keep the uncertain parts visible

The survey method estimates the ground, searches horizontal directions, examines height bands and assembles supported runs. It uses those runs to propose walls, partitions, structures and drawing elements. Every threshold has a purpose, but a raster size or fitting tolerance does not become a survey accuracy simply because it is small.

Low-support geometry needs a different treatment in the drawing. It remains on an identifiable CAD layer and visible in the analysis. A zero-error DXF audit establishes that the file is readable; the architect still has to check whether its lines and dimensions describe the room correctly. Opening candidates need the same care. A known window in a mathematical generator does not validate a window detector.

## Open the transition

We are publishing Lefebvre so this transition can be inspected and improved: from capture, to frame, to evidence, to drawing proposal. The code is an independent Python implementation using Qt, VTK and point-cloud libraries. CloudCompare remains an independent tool and workflow reference. Their licenses and required notices are retained.

The next useful contribution may be a better detector, a clearer review interaction or a small mathematical case that exposes a failure. A reproducible input with known geometry is particularly valuable. The aim is a workstation where a drawing remains connected to what supports it, and where the operator can see what still needs to be measured.

Creative Computational Architecture - Caglar Celik Architects (CCA)

*Redefining space through computation.*
