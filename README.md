# MEGL

## Preparation

### 1. Clone this repo
```
git clone --branch llava --single-branch https://github.com/TnTerry/MEGL.git
cd MEGL
pip install --upgrade pip  # enable PEP 660 support
pip install -r requirements.txt
```

### 2. Dataset Preparation
Use gdown to download the dataset from google drive.

```Shell
mkdir Datasets
cd Datasets
mkdir Action_Classification
mkdir Object_Classification
cd Action_Classification
gdown 1V_kah3MZuHG7UUyPAlg27M1kjnil-uPI
sudo apt-get update
sudo apt-get install unzip
unzip Action_Classification.zip
cd ..
cd Object_Classification
gdown 1SybY478ZMaTUhOWCGQIYPSKbSx-dUJfE
unzip Object_Classification.zip
```

Install libgl for opencv
```Shell
sudo apt-get update
sudo apt-get install libgl1-mesa-glx
```
