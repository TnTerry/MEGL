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
Action:
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
```

Object:
```Shell
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

Now, you can simply fine-tune the model

### 3. Prepare for Multimodal Explanation-Guided Learning

Clone the LLaVA Model
```
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install -e.
```

Install the packages for training
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

Clone the vision tower (e.g. CLIP)
```
git clone https://huggingface.co/openai/clip-vit-large-patch14-336
```

Clone the LLaVA Model weight
```

```
