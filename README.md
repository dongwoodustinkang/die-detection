# Semiconductor Die Surface Inspection System

[English](README.md) | [한국어](README_KO.md)

> An OpenCV-based program designed to rapidly detect defects on the top, bottom, left, and right surfaces of semiconductor dies.

<img src="assets/thumnail.png" width=1280>

## Background

This project combines rule-based computer vision with AI instead of relying exclusively on an object-detection AI model throughout the inspection process. The goal is to reduce the computational burden of high-resolution image processing while improving inspection speed, accuracy, decision traceability, and maintainability. This approach also aligns with human-in-the-loop, high-speed inspection and precision-review, and layered rule–AI workflows publicly presented by semiconductor manufacturers and inspection-equipment providers.

## Current Features

#### Side Inspection

Side inspection checks the die surface and shoulder balls (shoulder bumps) for damage. The process first converts the B page to grayscale, applies thresholding, and extracts the surface contour.

1. Surface inspection
   - Measures the first contact points from the top, bottom, left, and right edges.
   - Selects the top and bottom cutting lines from the contact-point frequency and density distribution, helping compensate for slight rotation or uneven contour geometry.
   - Calculates a separate pillar-based reference from the A page and applies both reference methods to identical coordinates on the A/B pages for crop comparison.

2. Shoulder-ball inspection

   The current experimental workflow (updated September 30, 2026) runs in the following order. Slot geometry, brightness scans, representative points, contour extraction, ROIs, and ROI previews are implemented in `side/ball.py`; `side/pipeline.py` assembles the results, and `ui.py` displays them.

   1. **Establish the surface references.** Extract the B-page surface contours and calculate the bottom surface reference line. Use the A-page pillar reference for the upper surface and create the A/B surface crop comparison. The outer left/right surface reference extensions are hidden; their contact markers and reference geometry remain available.
   2. **Define the slots.** Use the surface contour's horizontal range to establish the original three-slot geometry. Keep its two outer vertical boundaries fixed when changing the slot count; four-slot mode redistributes the internal boundaries evenly. Two-slot mode uses only **Slot 1 and Slot 3 of the three-slot layout**, retaining their positions and labels rather than dividing the surface into two halves. Vertical slot guides extend downward from the bottom surface reference.
   3. **Find a yellow horizontal line in each active slot.** Scan the original A-page grayscale image downward from the bottom surface reference. Select the first row where pixels with brightness **240–255 occupy at least 20% of the slot width**. For a tilted reference, only pixels below the reference at their own x-coordinate count toward the numerator; the denominator remains the full slot width. A slot with no qualifying row receives no yellow line.
   4. **Find a representative point.** Starting one row below the yellow line, scan downward at every x-coordinate in the slot. Find the first pixel with brightness **230–255** in each column, then select the point with the greatest vertical distance from the yellow line. Break distance ties by proximity to the slot center. Columns without a matching pixel are excluded. Draw the selected point in yellow, with a radius of 3 px. Both this scan and the yellow-line scan use original A pixels, before enhancement.
   5. **Extract experimental ball contours.** Apply a **3×3 median filter**, followed by unsharp masking with a **3×3 Gaussian blur, sigma 1.0, and enhancement amount 2.5**. Within each active slot below its yellow line, use inverse binary thresholding at **249**: values at or below 249 become foreground. Extract external contours and keep candidates with contour area **160–3,200 px²**. Draw all retained candidates as **1 px green contours** on the A/B analysis images. The current implementation uses fixed-threshold extraction, not Watershed or a U-shaped boundary assumption. `ENABLE_SLOT_CONTOUR_EXPERIMENT` can disable this contour experiment.
   6. **Create and fit the ROIs.** Create an ROI only when the representative point is **at least 10 px below** the yellow line. Its initial width is **40 px**, centered horizontally on the representative point; its top is the yellow line and its bottom is the representative point. Among contours whose bounding boxes overlap the initial ROI, choose the one nearest the representative point, using larger contour area to break distance ties. Expand the ROI as needed to contain that contour's full bounding box. Consequently, a fitted ROI can exceed 40 px in width or extend beyond the initial yellow-line/point bounds. If no contour overlaps it, retain the initial ROI. Show the rectangle on both A/B analysis images.
   7. **Show the side-panel ROI preview and measurements.** Crop each valid ROI from the original A page and enlarge it 3× for display. Keep the preview free of green contour overlays. Below each crop, show the original slot label and **W / H in original pixels**. Measurements are the axis-aligned bounding-box width and height of the largest-area retained contour whose bounding-box center lies inside the fitted ROI; they are not the ROI dimensions. Show `--` when no contour qualifies. No dimension arrows are drawn. Crops are clipped to the image bounds when necessary.

   Detection status currently requires both a surface contour and at least one valid ROI. Contour candidates, measurements, and thresholds are experimental and do not establish a final product OK/NG decision. Changes to the enhancement or area limits should be assessed on original images; strong enhancement can also amplify internal features.

   **Possible contour-free alternative:** if contour extraction is removed later, use each active slot's horizontal boundaries directly as the ROI width, with the yellow line and representative point defining the vertical bounds and the same 10 px minimum gap. This is a proposed alternative, not the current ROI implementation. Contour-based W/H measurements would then be unavailable.

#### Bottom Inspection · Chip Location and Circle Candidates

- Only `Bottom Detection` processes page A for this flow. Page B is an original reference.
- A 3×3 Gaussian blur and threshold `max(background + 20, (background + Otsu) / 2)` separate the chip, including its gray substrate.
- Six 40×40 px ROIs follow the chip position and rotation: 10%/90% across its width and 10%/50%/90% down its height.
- Dark components are extracted from the binary image before closing or filling its interior. External background and components smaller than 10 px² are excluded.
- Circle candidate conditions: **contour area 300–450 px², circularity ≥ 0.70, short/long side ratio ≥ 0.70, and distance from the expected ROI center ≤ 12 px**. Circularity is `4π × area / perimeter²`.
- Rejected components retain their actual contours and review reasons; empty ROIs remain missing. Components touching the search boundary and multiple qualifying components also require review.
- A shows the ROIs and actual contours: green for candidates, orange for review, red for missing. Detail captions use C for circularity and A for area. The chip outline rectangle and silhouette panel stay hidden.
- The status distinguishes six candidate regions from results requiring review. **Candidate acceptance is not a product pass/fail classification.**
- The detector assumes one rectangular chip on a dark background and rejects clipped, small, or elongated chips. Original A pixels and the original crop remain unchanged.

#### Top Surface Inspection

_Planned for a future release._

## Project Structure

```text
├── app.py             # Starts the PyQt5 application
├── ui.py              # Main window layout, state, and user interaction
├── ui_components.py   # Reusable image, histogram, and modal widgets
├── general.py         # Shared image I/O, capture-path, and notes helpers
├── contour.py         # Detection-independent contour extraction and geometry
├── bottom/
│   ├── pipeline.py    # A thresholding, chip location, and circle pipeline
│   └── circles.py     # Six ROIs, dark components, candidate rules, and previews
├── side/
│   ├── pipeline.py    # Orchestrates the complete Side detection flow
│   ├── surface.py     # Side surface detection and crop previews
│   └── ball.py        # Side ball detection
├── styles.py          # UI styles
├── requirements.txt   # Python dependencies
├── assets/            # Application icons and other resources
└── dataset/           # Inspection image data
```

Verify chip localization, circle candidates, and mode switching with `python -m unittest discover -s tests -v`.

## Requirements

- Python 3.10–3.12 recommended
- Input `TIFF` images must contain identically sized A/B pages. The program assumes a specific dataset format.

## Installation

Run the following commands after receiving the project.

### macOS / Linux

```bash
cd diehand_cv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
cd diehand_cv
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
python app.py
```

> This repository was developed for specific experimental research and may not be suitable for other projects.
