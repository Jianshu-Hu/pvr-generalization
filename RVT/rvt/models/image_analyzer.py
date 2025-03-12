from PIL import Image
import argparse
import os
import json
import time
from typing import List
import requests
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import os
import io
import re
import numpy as np

from swift.llm import (
    InferEngine, InferRequest, PtEngine, RequestConfig, get_template, load_dataset, load_image
)
from swift.utils import get_model_parameter_info, get_logger, seed_everything


class ImageAnalyzer():
    def __init__(self):
        # model
        # TODO:change the directory here
        last_model_checkpoint = '/bd_byta6000i0/users/jhu/VLM-finetune/output/checkpoint-351'
        self.output_dir = last_model_checkpoint + 'eval_results'
        model_id_or_path = 'Qwen/Qwen2.5-VL-3B-Instruct'

        # generation_config
        self.max_new_tokens = 512
        self.temperature = 0
        self.stream = True

        # Get model and template, and load LoRA weights.
        self.engine = PtEngine(model_id_or_path, adapters=[last_model_checkpoint])

    def infer_stream(self, img, lang_goal):
        # img (numpy [3, h, w]) to byte
        pil_image = Image.fromarray(np.transpose(img, (1, 2, 0)))
        image_bytes_io = io.BytesIO()
        pil_image.save(image_bytes_io, format="PNG")  # PNG or JPEG
        image_bytes = image_bytes_io.getvalue()
        single_data = {'messages': [{'role': 'user',
                                     'content': '<image>What is the current step for solving the task: ' + lang_goal + '?'},
                                    {'role': 'assistant',
                                     'content': 'The arm stack the green cup onto the magenta cup.'}],
                       'images': [{'bytes': image_bytes,
                                   'path': None}]}

        infer_request = InferRequest(**single_data)

        request_config = RequestConfig(max_tokens=self.max_new_tokens, temperature=self.temperature, stream=self.stream)
        gen_list = self.engine.infer([infer_request], request_config)
        response = ''
        for resp in gen_list[0]:
            if resp is None:
                continue
            delta = resp.choices[0].delta.content
            response += delta
            # print(delta, end='', flush=True)
        # print(response)
        return response

class PretrainedImageAnalyzer:
    def __init__(self):
        """
        Initialize the ImageAnalyzer with Hugging Face transformers
        """
        model_path = 'meta-llama/Llama-3.2-11B-Vision-Instruct'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model from {model_path}")
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        print(f"Model loaded successfully on {self.device}")

        self.color_list = [
            "azure", "blue", "navy", "purple", "violet", "cyan", "green", "lime", "olive", "teal",
            "magenta", "maroon", "red", "rose", "black", "gray", "silver", "white", "orange", "yellow"
        ]

    def extract_response(self, output_text: str) -> str:
        """Extract only the assistant's response from the output"""
        # Find the assistant's response part
        assistant_start = output_text.find("<|start_header_id|>assistant<|end_header_id|>")
        if assistant_start != -1:
            response = output_text[assistant_start:].replace("<|start_header_id|>assistant<|end_header_id|>",
                                                             "").strip()
            # Remove the end token if present
            response = response.replace("<|eot_id|>", "").strip()
            return response
        return output_text  # Return original text if pattern not found

    def analyze_single_image(self, image, lang_goal: str):
        """
        Analyze a single image
        """
        if lang_goal.split()[-1] == 'cup':
            prompt = f'This is the scene of a robot {lang_goal}.' \
                     f' What are the colors of the cups in this image? You can only choose from ' +\
                     ", ".join(self.color_list)
        image = Image.fromarray(image.transpose(1, 2, 0))
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)
        batch_inputs = [inputs]

        # Generate response
        outputs = []
        for inputs in batch_inputs:
            output = self.model.generate(**inputs, max_new_tokens=1024)
            decoded_output = self.processor.decode(output[0])
            outputs.append(self.extract_response(decoded_output))

        step_list = self.process_response(outputs[0], lang_goal)

        return step_list

    def process_response(self, response, lang_goal):
        if lang_goal.split()[-1] == 'cup':
            print(f'the response from llama is: {response}')
            # List of common colors

            # Create a regex pattern to match color names
            color_pattern = r"\b(" + "|".join(self.color_list) + r")\b"

            # Find all matches in the text
            colors_found = re.findall(color_pattern, response, flags=re.IGNORECASE)

            # Return unique colors found (case insensitive)
            colors = list(color.lower() for color in colors_found)
            if lang_goal.split()[-2] not in colors:
                return []
            # target cup should be put at the end
            colors.remove(lang_goal.split()[-2])
            colors.append(lang_goal.split()[-2])
            print(f'the colors of the cups are: {colors}')
            if len(colors) == 3:
                step_description_list = [
                    f'The robot arm moves downwards, positioning itself to grasp the {colors[0]} cup.',
                    f'The robot arm grasps the {colors[0]} cup with its gripper.',
                    f'The arm lifts the {colors[0]} cup.',
                    f'The robot arm moves the {colors[0]} cup above the {colors[2]} cup.',
                    f'The arm lowers the {colors[0]} cup, releasing it to stack successfully.',
                    f'The robot arm adjusts its position, preparing to grab the {colors[1]} cup.',
                    f'The robot arm grasps the {colors[1]} cup with its gripper.',
                    f'The arm lifts the {colors[1]} cup.',
                    f'The robot arm moves the {colors[1]} cup above the {colors[2]} cup.',
                    f'The arm lowers the {colors[1]} cup onto the stack, releasing it and completing the task.'
                ]
            elif len(colors) == 4:
                step_description_list = [
                    f'The robot arm moves downwards, positioning itself to grasp the {colors[0]} cup.',
                    f'The robot arm grasps the {colors[0]} cup with its gripper.',
                    f'The arm lifts the {colors[0]} cup.',
                    f'The robot arm moves the {colors[0]} cup above the {colors[3]} cup.',
                    f'The arm lowers the {colors[0]} cup, releasing it to stack successfully.',
                    f'The robot arm adjusts its position, preparing to grab the {colors[1]} cup.',
                    f'The robot arm grasps the {colors[1]} cup with its gripper.',
                    f'The arm lifts the {colors[1]} cup.',
                    f'The robot arm moves the {colors[1]} cup above the {colors[3]} cup.',
                    f'The arm lowers the {colors[1]} cup onto the stack, releasing it to stack successfully.'
                    f'The robot arm adjusts its position, preparing to grab the {colors[2]} cup.',
                    f'The robot arm grasps the {colors[2]} cup with its gripper.',
                    f'The arm lifts the {colors[2]} cup.',
                    f'The robot arm moves the {colors[2]} cup above the {colors[3]} cup.',
                    f'The arm lowers the {colors[2]} cup onto the stack, releasing it and completing the task.'
                ]
            else:
                step_description_list = []
        return step_description_list