# MEGL

## Preparation

### 1. Clone this repo
```
git clone https://github.com/TnTerry/MEGL.git
cd MEGL
```

### 2. Clone the models
Clone the vision tower (e.g. CLIP)
```
git clone https://huggingface.co/openai/clip-vit-large-patch14-336
```

Clone the LLM (e.g. Qwen2-1.5B-Instruct, Qwen2-0.5B-Instruct)
```
git clone https://huggingface.co/Qwen/Qwen2-1.5B-Instruct
```
```
git clone https://huggingface.co/Qwen/Qwen2-0.5B-Instruct
```

Add a special token in tokenizer_config.json of LLM to encode the image.
```python
"151646": {
      "content": "<image>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    }
```
```python
"additional_special_tokens": ["<|im_start|>", "<|im_end|>", "<image>"]
```
Follow the instructions in `test_llava_init.ipynb` to initialize the llava model


### 3. Dataset Preparation
Use gdown to download the dataset from google drive.

If gdown is not installed:
```Shell
pip install gdown
```
```
mkdir Datasets
cd Datasets
mkdir Action_Classification
mkdir Object_Classification
cd Action_Classification
gdown 1V_kah3MZuHG7UUyPAlg27M1kjnil-uPI
```

```Shell
sudo apt-get update
sudo apt-get install unzip
unzip Action_Classification.zip
```


### 4. 
