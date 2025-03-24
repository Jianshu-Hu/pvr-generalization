from PIL import Image
import io
import numpy as np
import torchvision.transforms as transforms
import re
import torch
from rvt.mvt.utils import generate_hm_from_pt

from swift.llm import (
    InferEngine, InferRequest, PtEngine, RequestConfig, get_template, load_dataset, load_image
)


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


class KeypointPredictor():
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

        self.to_pil = transforms.ToPILImage()

    def infer_stream(self, img, lang_goal):
        # img (tensor [1, 3, 10, h, w])
        img_size = img.size(3)
        all_view_image_bytes = []
        for view in range(img.size(1)):
            pil_image = self.to_pil(img[0, view, 3:6])
            image_bytes_io = io.BytesIO()
            pil_image.save(image_bytes_io, format="PNG")  # PNG or JPEG
            image_bytes = image_bytes_io.getvalue()
            all_view_image_bytes.append(image_bytes)
        single_data = {"messages": [{"role": "user",
                                     "content": "<image><image><image> Here are the different views of"
                                                " the scene where " + lang_goal +
                                                " What are the keypoints in each view?"},
                                    {"role": "assistant",
                                     "content": f'{[0, 0]}, {[0, 0]}, {[0, 0]}'}],
                       'images': [{'bytes': all_view_image_bytes[view],
                                   'path': None} for view in range(img.size(1))]}

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

        # Extract numbers using regex
        numbers = list(map(int, re.findall(r'\d+', response)))

        # Reshape into (N, 2) array
        coordinates = np.array(numbers).reshape(-1, 2)
        if coordinates.shape[0] != 3:
            raise ValueError('unexpected response!')
        print('transfer response to pixel positions: ')
        print(coordinates)
        action_trans = generate_hm_from_pt(
            torch.from_numpy(coordinates).to(img.device),
            (img_size, img_size),
            sigma=0.5,
            thres_sigma_times=3,
        )

        return action_trans.unsqueeze(0)