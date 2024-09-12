from groundingdino.util.inference import load_model, load_image, predict, annotate
# import groundingdino.datasets.transforms as T
import torchvision.transforms as T
import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class GroundingDinoHeatMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.root_path = '../../GroundingDINO/'
        self.model = load_model(self.root_path+"groundingdino/config/GroundingDINO_SwinT_OGC.py",
                                self.root_path+"weights/groundingdino_swint_ogc.pth")
        self.resize = T.Resize(800)
        self.box_threshold = 0.1
        self.text_threshold = 0.1

    @torch.no_grad()
    def forward(self, images, text_prompts):
        # images (B, num_views, 10, h, w), text_prompts (B, 1)
        b, nv, _, h, w = images.shape
        rgb_views_images = self.resize(images[:, :, 3:6, :, :].reshape(b*nv, 3, h, w)).reshape(b, nv, 3, 800, 800)
        hm = torch.zeros(b, nv, h, w).cuda()
        for (i, text) in enumerate(text_prompts):
            for view_ind in range(nv):
                boxes, logits, phrases = predict(
                    model=self.model,
                    image=rgb_views_images[i, view_ind],
                    caption=text[0],
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold
                )
                max_ind = torch.argmax(logits)
                max_box = boxes[max_ind] * torch.Tensor([w, h, w, h])
                # print(torch.tensor([i, view_ind, max_box[1], max_box[0]]))
                hm[i, view_ind, int(max_box[1]), int(max_box[0])] = 1.0

        out = {"trans": hm}

        return out
