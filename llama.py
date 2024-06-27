from transformers import AutoTokenizer, LlamaForCausalLM
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import login, HfApi
import os

# Define your Hugging Face API token
hf_token = "your_token"

model_name = "meta-llama/Llama-2-7b-hf"

class CustomLlamaModel(nn.Module):
    def __init__(self, llama_model_name = model_name):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llama = LlamaForCausalLM.from_pretrained(llama_model_name, token=hf_token).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_name, token=hf_token)
        # Assuming your embeddings are 1024-dimensional, which matches LLaMA's hidden size
        self.embedding_dim = self.llama.config.hidden_size
    
    def forward(self, input_embeddings, labels=None):
        prompt = "What is the capital of France?"
        input_embeddings = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        # input_embeddings = self.tokenizer.decode(input_embeddings, skip_special_tokens=True)
        # Bypass the embedding layer, input_embeddings should match the model's embedding size
        outputs = self.llama.generate(input_embeddings, labels=labels)
        outputs_sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return outputs, outputs_sentence

model = CustomLlamaModel()
outputs = model.forward(torch.randn(1, 1024))
print(outputs)

# # Load pre-trained model and tokenizer
# model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# # Assuming your embeddings are 1024-dimensional, which matches LLaMA's hidden size
# embedding_dim = model.config.hidden_size

# class CustomLlamaModel(nn.Module):
#     def __init__(self, llama_model):
#         super().__init__()
#         self.llama = llama_model
    
#     def forward(self, input_embeddings, labels=None):
#         # Bypass the embedding layer, input_embeddings should match the model's embedding size
#         outputs = self.llama(inputs_embeds=input_embeddings, labels=labels)
#         return outputs

# # Instantiate the custom model
# custom_model = CustomLlamaModel(model)

# # Example dataset
# class MyDataset(Dataset):
#     def __init__(self, embeddings, texts, tokenizer):
#         self.embeddings = embeddings
#         self.texts = texts
#         self.tokenizer = tokenizer
    
#     def __len__(self):
#         return len(self.texts)
    
#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         embedding = self.embeddings[idx]
#         inputs = self.tokenizer(text, return_tensors="pt")
#         return {"input_embeddings": embedding, "labels": inputs["input_ids"].squeeze()}

# # Assume embeddings and texts are your data
# dataset = MyDataset(embeddings, texts, tokenizer)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# # Training loop
# optimizer = torch.optim.Adam(custom_model.parameters(), lr=5e-5)
# custom_model.train()

# for epoch in range(num_epochs):
#     for batch in dataloader:
#         optimizer.zero_grad()
#         outputs = custom_model(batch["input_embeddings"], labels=batch["labels"])
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#         print(f"Loss: {loss.item()}")
