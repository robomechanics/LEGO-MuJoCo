# LEGO Project MuJoCo

LEGO Project MuJoCo simulation framework with data recording.

Works in WSL Ubuntu-20.04, Conda environment with Python 3.10.16

## Coordinate Convention

Repo-facing parameters and analysis use:

- `+x`: forward walking direction
- `+y`: robot-left / lateral
- `+z`: down into the ground

MuJoCo model files and runtime APIs remain in MuJoCo's usual z-up frame. Code
that crosses that boundary should convert explicitly with `coordinate_frame.py`.

## Installation
Clone the repository and install dependencies.

## 🔧 Setting Up the Conda Environment

### **Using Conda (`environment.yml`)**
Create and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate my_env  # Replace 'my_env' with the actual environment name
```

## 🚀 Running the Simulation

### **Start Example Sims**
Use the following command to start the `Duplo` simulation with GUI enabled for 10 seconds:
```bash
python -m src.duplo -n example -t 10 -gui
```
Check all possible args with:
```bash
python -m src.duplo -h
```
Settings can be found in `utils/all_args.py`.

### **Edit Mass Properties**
Mass properties of `Duplo` can be found in `robots/duplo_ballfeet_mjcf/mass_config.yaml`. Changes to the file will be loaded automatically when the simulation starts.

## 🎥 Recording simulation

### **Recording Scene and Data**
Use the following command to start the `Duplo` simulation with for 10 seconds and record the scene and data to `data/videos/example/`
```bash
python -m src.duplo -n example -t 10 -r
```
Resulting videos are the data plot, the scene, and a composite video.

![GitHub last commit](https://img.shields.io/github/last-commit/stevenwman/LEGO-MuJoCo)
