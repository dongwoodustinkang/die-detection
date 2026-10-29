# Semiconductor Die Surface Inspection System

[English](README.md) | [한국어](README_KO.md)

> An OpenCV-based program for rapidly detecting defects on the top, bottom, left, and right surfaces of semiconductor dies.

<img src="assets/thumnail.gif" width=1280>

## Background
Rather than applying an object-detection AI model on its own across the entire inspection process, this research combines rule-based computer vision with several vision models. This reduces the computational burden of high-resolution image processing and secures inspection speed and accuracy as well as traceability of decisions and maintainability. The approach also matches the human-in-the-loop, high-speed inspection with precision review, and layered rule–AI practices that semiconductor manufacturers and inspection-equipment vendors have recently presented.

## Current Features

### Side Inspection

Side inspection checks **the die surface and shoulder balls (shoulder bumps)** for damage.
First, the B page is converted to grayscale and thresholded to extract the surface contour.

#### Surface Inspection
   - Measures the first contact points from the top, bottom, left, and right edges.
   - Selects the top and bottom cutting lines from the frequency and density distribution of the contact points, compensating for slight rotation or uneven contour shapes on the side.
   - Calculates a separate pillar reference on the A page, then applies both reference methods to identical coordinates on the A/B pages and compares the crops.

#### Shoulder-Ball Inspection

   2-1. `find_downward_contour_white_points`
      - Starting from the bottom-surface coordinates obtained on the B page, scans downward and collects every position that is white (255).
      - Only points at least 5 px below the bottom-surface reference line (`SCAN_START_OFFSET=5`) are collected. (This excludes points that turned white because of damage or light reflection just below the bottom surface.)

   2-2. `get_distant_bottommost_points`
      - Finds the lowest point among all white pixels (`global_max_y`), which is the bottom of a ball.
      - Selects additional bottom points only where they are separated by 10 px on the y-axis and 40 px on the x-axis.

   2-3. `get_ball_square_bounds`, `draw_ball_squares`
      - Draws bounding boxes from the selected ball bottom points and passes them to the UI.

   > Optional) Precise arc measurement and contour validation (`ball_arc.measure_ball_arcs`)
      > - Extracts Canny edges and fits them to calculate radius, error, contour, and symmetry.

![alt text](assets/side-ball-example.png)

### Bottom Inspection · Chip Location and Circle Candidates

> The expected centers of the six ROIs are set by combining a horizontal ratio (`fx`) and a vertical ratio (`fy`) from the chip's top-left corner.
> The ROI size and the area and center-distance conditions for circle candidates are checked as ratios of the chip size.

1. Binarization and chip location

   1-1. `find_bottom_chip`, `_binarize`, `_locate_bottom_chip`
      - Binarizes the gray chip using both the background brightness and the Otsu threshold.
      - From the largest near-rectangular chip contour, finds the four corners (`chip.corners`), center, rotation angle, and chip mask.

   1-2. `find_circle_regions`
      - Calculates the expected centers of the six ROIs from the top-left corner using the `fx` and `fy` ratios. The positions are `10%` left/right and `10%·50%·90%` top/middle/bottom.
      - Builds each ROI by multiplying the chip's top and left edge vectors by `23.53%`. As a result, when the chip rotates or scales, the ROIs follow in the same direction and proportion.
      - Finds the actual contour of the dark component in each ROI and measures its area, circularity, aspect ratio, and distance from the expected center.
      - A circle candidate must have circularity of at least `0.80`, aspect ratio of at least `0.70`, area of `0.93%–1.56%` of the chip, and center distance of at most `7.06%` of the chip width. Components touching the search boundary or multiple candidates are left for review.

   1-3. `draw_circle_regions`, `create_circle_roi_preview`
      - Draws the ROIs and actual contours in colors matching the candidate, review, and missing states.
      - Arranges the six ROIs as 3 × 2 tiles and shows the circularity and area for each position.

      > - Area: computes the chip and circle areas and takes their ratio `circle // chip * 100`
      > - Circularity: area relative to perimeter `(4 * pi * area) / perimeter ^ 2`
      > - Aspect ratio: the circle's width and height should be similar
      > - Distance between the expected and actual centers `(center distance // chip width * 100)`



### Top Surface Inspection

_Planned for a future release._

## Project Structure

```text
├── app.py               # Starts the PyQt5 application (--dark, --install-desktop-entry options)
├── ui.py                # Main window layout, state, and user interaction
├── ui_components.py     # Shared image and modal widgets, SVG icons
├── styles.py            # Light and dark UI styles, icon and caption colors
├── general.py           # Image I/O, capture paths, notes, filename labels (OK/NG/DK)
├── contour.py           # Detection-independent contour extraction and geometry
├── bottom/
│   ├── pipeline.py      # A-page thresholding, chip location, and circle-candidate flow
│   └── circles.py       # Six ROIs, dark components, candidate rules, and display
├── side/
│   ├── pipeline.py      # Assembles the full Side detection flow
│   ├── surface.py       # Side surface detection and crop previews
│   ├── surface_crop.py  # Surface crop tiles shared by training-data creation and inference
│   ├── ball.py          # Side ball detection
│   ├── ball_arc.py      # Shoulder-ball bottom arc measurement
│   ├── classifier.py    # Surface OK/NG classification CNN
│   ├── detection.py     # Surface defect object detection (Faster R-CNN)
│   └── segmenter.py     # Surface defect segmentation U-Net
├── tools/               # Training and evaluation scripts (see "Training and Evaluation Tools")
│   ├── data/            # Build training datasets
│   ├── train/           # Train models
│   ├── eval/            # Evaluate models
│   ├── results/         # Summarize evaluation results
│   └── CNN_guide.md     # Detailed guide: data creation → training → evaluation
├── models/              # Trained models (.pt) with settings, graphs, and evaluation results (private)
├── assets/              # Application icon, button SVG icons, and other resources
├── requirements.txt     # Python dependencies
└── dataset/             # Inspection image data (private)
```

Files created while the program runs:

- `captures/`: screen-capture JPGs (`captures/{timestamp}/{source filename}_{number}.jpg`)
- `notes.csv`: per-image notes

## Requirements

- Python 3.10–3.12 recommended
- Input `TIFF` images must contain identically sized A/B pages. The program assumes a specific dataset format.

## Dataset and Models

> **The inspection image dataset (`dataset/`) and the trained models (`models/`) are not public while the research is in progress.**

- This repository contains only the inspection program and the training and evaluation code.
- To try the program, prepare your own TIFF images with A/B pages and open them with `불러오기` (Load).
- The default paths and example commands of the `tools/` scripts are written for the private dataset and models. When running them on your own data, specify paths with the input folder and options such as `--data`, `--model`, and `--output`.

## Installation

Run the following commands after getting the project.

### macOS / Linux

```bash
git clone https://github.com/dongwoodustinkang/die-detection.git
cd die-detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
git clone https://github.com/dongwoodustinkang/die-detection.git
cd die-detection
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
python app.py
```

To run in dark mode, add the `--dark` option.

```bash
python app.py --dark
```

Some Linux desktops (e.g. GNOME on Wayland) look up the app icon through a `.desktop` file. If the dock or taskbar shows a generic icon, run this once from the virtual environment you use. Run it again after moving the project folder or the virtual environment.

```bash
python app.py --install-desktop-entry
```

### Using the Window

The interface is in Korean; the button names are given below with their English meaning.

| Action | How |
|---|---|
| Change inspection mode | Click the title at the top left (`Side Detection` ▾) and choose Side / Bottom / Top |
| Load images | `불러오기` (Load) → select several TIFF files or one folder |
| Detect again | `검출하기` (Detect) |
| Move between images | Keyboard `←` / `→`, or the previous/next buttons on the bottom navigation bar |
| Notes | Note button on the navigation bar (saved to `notes.csv`) |
| Screen capture | Camera button on the navigation bar (saved to `captures/`) |
| Enlarge an image | Click a result image or preview |
| Quit | `Esc` |

The label on the right of the header is the ground-truth label taken from the **filename**, not a detection result.

| Label | Condition |
|---|---|
| `NG` (defective) | The filename contains `NG` |
| `OK` (normal) | The filename contains `OK` (`NG` wins if both appear) |
| `DK` (unknown) | Neither appears |

## Training and Evaluation Tools (tools)

Scripts for the side-surface defect models (classification CNN, Faster R-CNN, U-Net). Run them from the repository root.

| Folder | Script | Role |
|---|---|---|
| `tools/data/` | `make_data.py` | Builds train/val/test classification datasets from surface crop tiles |
| | `make_sg_data.py` | Builds a segmentation (U-Net) dataset from NG masks |
| | `make_od_data.py` | Builds an object-detection (box) dataset from NG masks |
| `tools/train/` | `train_classifier.py` | Trains the surface OK/NG classification CNN |
| | `train_segmenter.py` | Trains the surface defect segmentation U-Net |
| | `train_detection.py` | Trains the surface defect Faster R-CNN |
| | `train_graphs.py` | Saves training graphs (used by the other training scripts) |
| `tools/eval/` | `eval_classifier.py` | Evaluates the classification CNN |
| | `eval_segmenter.py` | Evaluates the U-Net |
| | `eval_detection.py` | Evaluates the Faster R-CNN |
| `tools/results/` | `summarize_results.py` | Summarizes model × evaluation-set accuracy into a CSV |
| | `export_errors.py` | Collects the original and crop images listed in a misclassification CSV into `dataset/errors/` |

Examples (the dataset and model paths are private; replace them as described in [Dataset and Models](#dataset-and-models)):

```bash
python tools/data/make_data.py --overwrite
python tools/train/train_classifier.py
python tools/eval/eval_classifier.py --model models/Classifier/side_surface_cnnA_102211.pt
```

Check each script's options with `--help`. Besides `requirements.txt`, the training scripts need `matplotlib`, and Faster R-CNN needs `torchvision`.

> This repository was developed for specific experimental research and may not be suitable for other projects.
