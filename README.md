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
   - Scans downward from each x-coordinate of the bottom contour on the A page and detects the first white pixel as a side-ball candidate.
   - The bottom contour range is divided into three sections, and the lowest point in each section is evaluated to select valid ball positions.
   - Detected positions are displayed as square crop regions. A sample is classified as detected only when both a valid surface contour and valid ball positions are found.

#### Top and Bottom Surface Inspection

_Planned for a future release._

## Project Structure

```text
├── app.py             # Starts the PyQt5 application
├── ui.py              # Main window layout, state, and user interaction
├── ui_components.py   # Reusable image, histogram, and modal widgets
├── general.py         # Shared image I/O, capture-path, and notes helpers
├── contour.py         # Detection-independent contour extraction and geometry
├── side/
│   ├── pipeline.py    # Orchestrates the complete Side detection flow
│   ├── surface.py     # Side surface detection and crop previews
│   └── ball.py        # Side ball detection
├── styles.py          # UI styles
├── requirements.txt   # Python dependencies
├── assets/            # Application icons and other resources
└── dataset/           # Inspection image data
```

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
