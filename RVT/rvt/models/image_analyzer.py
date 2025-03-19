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
    def __init__(self, model_ckpt):
        # model
        # TODO:change the directory here
        last_model_checkpoint = '/bd_byta6000i0/users/jhu/VLM-finetune/saved_ckpts/'+model_ckpt
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