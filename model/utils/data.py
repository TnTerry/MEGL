from PIL import Image
from pathlib import Path
import json
from dataclasses import dataclass

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoProcessor

class MEGLDataset(Dataset):
    def __init__(self, dataset_dir:str) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.chat_data, self.image_dir = self.build_dataset(dataset_dir)
    
    def build_dataset(self, data_dir:str) -> tuple[list[dict], Path]:
        data_dir = Path(data_dir)
        chat_file_old = data_dir.joinpath("exp_annotation.json")
        image_dir = data_dir.joinpath("images")

        # The old version of chat file is a dict form
        # Transform it to a standard json file
        with open(chat_file_old) as f:
            chat_old = json.load(f)
        chat_new = []
        for chat_id in chat_old:
            for txt_exp in chat_old[chat_id]["exp"]: # Multiple visual explanations for one image
                chat_new.append(
                    {
                        "id": chat_id,
                        "class_id": chat_old[chat_id]["class_id"],
                        "visual_exp": chat_old[chat_id]["visual_exp"],
                        "ans": chat_old[chat_id]["ans"],
                        "image": chat_old[chat_id]["image_path"].split("/")[-1],
                        "conversations": [
                            {
                                "from": "human",
                                "value": "Provide a brief description of the given image.\n<image>"
                            },
                            {
                                "from": "gpt",
                                "value": txt_exp
                            }
                        ]
                    }
                )
        
        return chat_new, image_dir
    
    def __len__(self) -> int:
        return len(self.chat_data)
    
    def __getitem__(self, idx:int) -> tuple[str, str, Path]:
        current_data = self.chat_data[idx]
        human_input = current_data["conversations"][0]["value"]
        gpt_output = current_data["conversations"][1]["value"]
        image_path = self.image_dir.joinpath(current_data["image"])
        if current_data['visual_exp']:
            visual_exp_path = self.dataset_dir.joinpath(current_data['visual_exp'])
        else:
            visual_exp_path = None
        class_id = current_data["class_id"]
        ans = current_data["ans"]

        return (human_input, gpt_output, image_path, visual_exp_path, class_id, ans)

@dataclass
class QaImageOutput:
    q_inputs_ids: torch.Tensor
    pixel_values: torch.Tensor
    a_inputs_ids: torch.Tensor

def build_qaimage(processor:AutoProcessor, q_text:str, a_text:str, image_path:Path, visual_exp_path:Path|None):

    chat_template = """
    "USER: <image>\n<prompt> ASSISTANT:"
    """

    # Set the chat template
    processor.tokenizer.chat_template = chat_template

    # QA + instructions
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": q_text},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Image
    raw_image = Image.open(image_path)

    if visual_exp_path:
        visual_exp = np.load(visual_exp_path)
        if visual_exp.ndim == 3 and visual_exp.shape[0] == 3:  # Check if it's a 3-channel image
            visual_exp = visual_exp.transpose(1, 2, 0)  # Reorder dimensions to (height, width, channels)
            if visual_exp.dtype == np.float64:
                visual_exp = (visual_exp * 255).astype(np.uint8)  # Assuming the data is in range [0, 1]
        visual_exp = Image.fromarray(visual_exp.astype('uint8'))  # Assume it's grayscale
        visual_exp = visual_exp.convert('RGB')
    else:
        visual_exp = Image.new('RGB', raw_image.size, (0, 0, 0)) # Dummy image of zeros

    inputs = processor(prompt, visual_exp, return_tensors="pt")

    a_inputs_ids = processor.tokenizer(
        a_text, 
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )['input_ids']
    
    return QaImageOutput(
        q_inputs_ids=inputs['input_ids'],
        pixel_values=inputs['pixel_values'],
        a_inputs_ids=a_inputs_ids
    )

transform_image = transforms.Compose([
                  transforms.Resize((256, 256)),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]) 
        
transform_visualexp = transforms.Compose([
                      transforms.Resize((256, 256)),
                      transforms.ToTensor()
                      ])

class TrainMEGLCollator:
    def __init__(self, processor:AutoProcessor, INGNORE_INDEX:int) -> None:
        self.processor = processor
        self.ignore_index = INGNORE_INDEX
    
    def convert_one_piece(self, 
                          q_input_ids:torch.Tensor,
                          a_input_ids:torch.Tensor):
        
        # input_ids: concatenate q_input_ids, a_input_ids, and eos_token_id
        input_ids = torch.concat([
            q_input_ids,
            a_input_ids,
            torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1,-1)
        ], dim=1)

        # labels: concatenate ignore_index, a_input_ids, and eos_token_id
        labels = torch.concat([
            torch.full_like(q_input_ids, self.ignore_index),
            a_input_ids,
            torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1,-1)
        ], dim=1)
        
        return input_ids, labels
    
    def __call__(self, features:list) -> dict:
        # TODO: Modify this to intergrate the original collator
        input_ids_list = []
        labels_list = []
        pixel_values_list = []
        max_input_len_list = []

        for feature in features:
            qaimage_output = build_qaimage(
                processor=self.processor,
                q_text=feature[0],
                a_text=feature[1],
                image_path=feature[2],
                visual_exp_path=feature[3]
            )
            tmp_input_ids, tmp_labels = self.convert_one_piece(
                q_input_ids=qaimage_output.q_inputs_ids,
                a_input_ids=qaimage_output.a_inputs_ids
            )
            max_input_len_list.append(tmp_input_ids.shape[1])
            input_ids_list.append(tmp_input_ids)
            labels_list.append(tmp_labels)
            pixel_values_list.append(qaimage_output.pixel_values)

        max_input_len = max(max_input_len_list)
        
        final_input_ids = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.processor.tokenizer.pad_token_id,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(input_ids_list)
            ]
        )
        final_labels = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.ignore_index,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(labels_list)
            ]
        )
        final_pixel_values = torch.concat(pixel_values_list, axis=0)
        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids ==
                       self.processor.tokenizer.pad_token_id] = 0

        visual_exp_lst = []
        image_lst = []
        class_id_lst = []

        for feature in features:
            image = Image.open(feature[2]).convert('RGB')

            if feature[3]:
                visual_exp = np.load(feature[3])
                if visual_exp.ndim == 3 and visual_exp.shape[0] == 3:  # Check if it's a 3-channel image
                    visual_exp = visual_exp.transpose(1, 2, 0)  # Reorder dimensions to (height, width, channels)
                    if visual_exp.dtype == np.float64:
                        visual_exp = (visual_exp * 255).astype(np.uint8)  # Assuming the data is in range [0, 1]
                
                visual_exp = Image.fromarray(visual_exp.astype('uint8'))  # Assume it's grayscale
                visual_exp = visual_exp.convert('RGB')
            else:
                visual_exp = Image.new('RGB', image.size, (0, 0, 0)) # Dummy image of zeros
            
            image = transform_image(image)
            image_lst.append(image)

            visual_exp = transform_visualexp(visual_exp)
            visual_exp_lst.append(visual_exp)

            class_id = torch.tensor(feature[4], dtype=torch.long)
            class_id_lst.append(class_id)
        
        images = torch.stack(image_lst)
        visual_exps = torch.stack(visual_exp_lst)
        class_ids = torch.tensor(class_id_lst)

        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "pixel_values": final_pixel_values,
            "attention_mask": attention_mask,
            "images":images,
            "visual_exps":visual_exps,
            "class_ids":class_ids
        }
