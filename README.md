# Simple IVUS-pullback Viewer

Open source DICOM viewers (e.g., [OHIF]()) do not work well for IVUS pullback 
data. More than often the entire pullback is stored as a Multi-frame Object 
in a single series instance, and in addition the individual frames are not 
true RGB images. The base data are stored as 8-bit grayscale, but the images are 
represented as 3D-array solely because of scale-mark annotations that are added 
by the IVUS-device.

This script is a simple viewer largely based on just matplotlib package. It allows 
frame-scrolling of images, viewing and searches in header, and some basic 
processing to remove scale-marks and apply basic annotations.

## Run

After cloning the repo, if necessary, create a conda environment with python>=3.8 and install
virtualenv
```bash
conda create --name viewer python=3.9
conda activate viewer
pip install virtualenv
```

create a virtual environment inside and install the dependencies:
```bash
python -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

on Windows the environment activation goes something like:
```bash
.\venv\Scripts\activate
```
and to run:
```bash
python viewer.py -f path/to/pullback/series
```

## Usage

### Frame-scrolling

### Header viewer/search

### Frame cleaning

### Basic annotations