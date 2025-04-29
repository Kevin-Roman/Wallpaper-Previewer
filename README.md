# Wallpaper Previewing System

This project presents a functional wallpaper and texture previewing software that leverages open-source models. Created by Kevin Roman.

## Project Structure

```
Wallpaper-Previewer/
│
├── external/           # Unmodified external repositories used.
├── adapted/            # External repositories with custom modifications. 
├── src/                # Core source code developed for the project.
│   ├── app/            # GUI and surface previewers.
│   ├── interfaces/     # Interfaces for each key system component.
│   ├── models/         # Integration of adapted/external models.
│   └── rendering/      # 3D texture rendering functionality.
│
├── weights/            # Model weights (see below for instructions).
├── tests/              # Unit tests.
├── data/               # Contains sample open licensed input images and wallpapers.
└── evaluation/         # Evaluation scripts used for the dissertation.
```

## Model Weights

Due to large file sizes, model weights must be downloaded directly from their respective authors:

### Room Layout Estimation

<https://github.com/leVirve/lsun-room>  
File: `model_retrained.ckpt`
Place in: `weights/room_layout_estimation`

### Wall Segmentation

<https://github.com/bjekic/WallSegmentation/tree/main/model_weights>  
Files:

- `best_encoder_epoch_19.pth`
- `best_decoder_epoch_19.pth`  

Place in: `weights/wall_segmentation`

### Illumination Estimation

<https://github.com/Wanggcong/StyleLight>  
File: `network-snapshot-002000`  
Place in: `weights/illumination_estimation`

## Setup Instructions

### Python Version

This software only works on Python 3.11 due to the use of `bpy`.

### Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the App

```bash
python -m src.app.main
```

## License Compliance

All external and adapted repositories used in this project have been carefully reviewed to ensure license compliance. The corresponding license files are included within each repository in `external/` and `adapted/`.
