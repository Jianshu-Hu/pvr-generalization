from typing import Tuple, List
from groundingdino.util.inference import load_model, load_image, predict, annotate, preprocess_caption
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
        # the number of top bounding boxes in each view
        self.K = 5

        self.all_combination = []
        self.num_view = None

    def batch_predict(self, image: torch.Tensor, captions: List):
        captions = list(map(preprocess_caption, captions))

        with torch.no_grad():
            outputs = self.model(image, captions=captions)

        prediction_logits = outputs["pred_logits"].sigmoid()  # (bs, nq, 256)
        max_logits = prediction_logits.max(dim=-1)[0]  # (bs, nq)
        prediction_boxes = outputs["pred_boxes"]  # (bs, nq, 4)
        # top-k
        bs = prediction_boxes.size(0)
        top_ind = torch.topk(max_logits, self.K, dim=-1).indices  # (bs, K)
        top_box = prediction_boxes[torch.arange(bs).reshape(bs, 1), top_ind]   # (bs, K, 4)

        return top_box

    def find_all_combinations(self, all_centers, view_index, to_tensor):
        # search for all combinations
        new_combination = []
        if len(self.all_combination) == 0:
            for k in range(self.K):
                new_combination.append([all_centers[:, view_index, k, :]])
        else:
            for point in self.all_combination:
                for k in range(self.K):
                    temp_point = list(point)
                    temp_point.append(all_centers[:, view_index, k, :])
                    if to_tensor:
                        temp_point = torch.stack(temp_point, dim=1)
                    new_combination.append(temp_point)
            if to_tensor:
                new_combination = torch.stack(new_combination, dim=1)
        self.all_combination = new_combination

    def find_closest_center(self, all_center_corr):
        # Input: the corresponding global coordinates of the centers (b, nv, K, 3),
        # For all nv images, we can have K^nv possible centers pairs
        # Find the centers pair whose sum of relative distance is smallest .
        b, nv, K, _ = all_center_corr.shape
        self.all_combination = []
        for num_view in range(nv):
            if num_view == nv-1:
                # save as tensor
                to_tensor = True
            else:
                to_tensor = False
            self.find_all_combinations(all_center_corr, num_view, to_tensor)
        # self.all_combination.shape (b, K^nv, nv, 3)
        mean = torch.mean(self.all_combination, dim=2, keepdim=True)
        diff = torch.mean(torch.sum((self.all_combination-mean)**2, dim=-1), dim=-1)
        min_dis_ind = torch.argmin(diff, dim=-1)
        # [b, 3]
        final_center = mean[torch.arange(b), min_dis_ind, 0, :]
        return final_center

    @torch.no_grad()
    def forward(self, images, text_prompts):
        # images (B, num_views, 10, h, w), text_prompts (B, 1)
        b, nv, _, h, w = images.shape
        rgb_views_images = self.resize(images[:, :, 3:6, :, :].reshape(b*nv, 3, h, w))
        text_prompts = text_prompts.repeat(nv, axis=1).reshape(b*nv)
        boxes = (self.batch_predict(rgb_views_images, text_prompts.tolist()))  # (b*nv, K, 4)
        centers = (boxes[:, :, 0:2]*torch.Tensor([w, h]).to(boxes.device)).int()  # (b*nv, K, 2)
        all_global_corr = images[:, :, 0:3, :, :].reshape(b*nv, 3, h, w)  # (b*nv, 3, h, w)

        all_center_corr = []
        for k in range(self.K):
            # (b*nv, 3)
            center_corr = all_global_corr[torch.arange(b*nv).reshape(-1, 1), :, centers[:, k, 1:2], centers[:, k, 0:1]]
            all_center_corr.append(center_corr)
        all_center_corr = torch.stack(all_center_corr, dim=1)  # (b*nv, K, 3)
        final_center = self.find_closest_center(all_center_corr.reshape(b, nv, self.K, 3))

        # boxes = (boxes*torch.Tensor([w, h, w, h]).to(boxes.device)).int()
        # center = boxes[:, 0, 0:2]  # (b*nv, 2)
        #
        # hm = torch.zeros(b*nv, h, w).to(boxes.device)
        # hm[torch.arange(b*nv).reshape(-1, 1), center[:, 1:2], center[:, 0:1]] = 1.0
        # hm = hm.reshape(b, nv, h, w)

        # hm = torch.zeros(b, nv, h, w).cuda()
        # for (i, text) in enumerate(text_prompts):
        #     for view_ind in range(nv):
        #         boxes, logits, phrases = predict(
        #             model=self.model,
        #             image=rgb_views_images[i, view_ind],
        #             caption=text[0],
        #             box_threshold=self.box_threshold,
        #             text_threshold=self.text_threshold
        #         )
        #         max_ind = torch.argmax(logits)
        #         max_box = boxes[max_ind] * torch.Tensor([w, h, w, h])
        #         # print(torch.tensor([i, view_ind, max_box[1], max_box[0]]))
        #         hm[i, view_ind, int(max_box[1]), int(max_box[0])] = 1.0

        out = {"trans": final_center}

        return out
