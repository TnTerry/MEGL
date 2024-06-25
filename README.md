# MEGL

## Preparation

### 1. Clone this repo
```
git clone git@github.com:TnTerry/MEGL.git
cd MEGL
```

### 2. Clone the models
Clone the vision tower (e.g. CLIP) and the LLM (e.g. Qwen2-1.5B-Instruct)
```
git clone https://huggingface.co/openai/clip-vit-large-patch14-336
git clone https://huggingface.co/Qwen/Qwen2-1.5B-Instruct
```

Add a special token in tokenizer_config.json of LLM to encode the image.
```
"151646": {
      "content": "<image>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    }
```

### 3. Dataset Preparation
```
mkdir Datasets
cd Datasets
mkdir Action_Classification
mkdir Object_Classification
cd Action_Classification
gdown 1V_kah3MZuHG7UUyPAlg27M1kjnil-uPI
```

4. 
