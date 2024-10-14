import os
from typing import Tuple, List
from groundingdino.util.inference import load_model, load_image, predict, annotate, preprocess_caption
# import groundingdino.datasets.transforms as T
import torchvision.transforms as T
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import clip
from einops import rearrange, repeat


class GroundingDinoHeatMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.root_path = '../../GroundingDINO/'
        self.model = load_model(self.root_path+"groundingdino/config/GroundingDINO_SwinT_OGC.py",
                                self.root_path+"weights/groundingdino_swint_ogc.pth")
        self.resize_size = 224
        self.resize = T.Resize(self.resize_size)
        # the number of top bounding boxes in each view
        self.K = 5

        self.all_combination = []
        self.num_view = None

        self.device = 'cuda'
        self.clip_model, self.clip_preprocess = clip.load("RN50", self.device)
        self.clip_model.to(device=self.device)
        self.clip_model.eval()

        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)

    def batch_predict(self, image: torch.Tensor, captions: List, h: int, w: int):
        captions = list(map(preprocess_caption, captions))

        with torch.no_grad():
            self.model.to('cuda')
            image.to('cuda')
            outputs = self.model(image, captions=captions)

        prediction_logits = outputs["pred_logits"].sigmoid()  # (bs, nq, 256)
        max_logits = prediction_logits.max(dim=-1)[0]  # (bs, nq)
        prediction_boxes = outputs["pred_boxes"]  # (bs, nq, 4)

        # filter the logits: the bounding box may include the whole image sometimes, filter them.
        # filter the logits: the height or width of the bounding box may be 1, filter them.
        box_size = prediction_boxes[:, :, 2]*prediction_boxes[:, :, 3]  # (bs, nq)
        filter = box_size > 0.81
        max_logits[filter] = 0.0

        filter = (prediction_boxes[:, :, 2]*w).to(torch.int) <= 1
        max_logits[filter] = 0.0

        filter = (prediction_boxes[:, :, 3]*h).to(torch.int) <= 1
        max_logits[filter] = 0.0

        # top-k
        bs = prediction_boxes.size(0)
        top_ind = torch.topk(max_logits, self.K, dim=-1).indices  # (bs, K)
        top_box = prediction_boxes[torch.arange(bs).reshape(bs, 1), top_ind]   # (bs, K, 4)

        return top_box

    def feature_fusion(self, image, box):
        # image tensor (3, H, W)
        # box tensor (K, 4)
        _, H, W = image.shape
        img = (image.permute(1, 2, 0).cpu().numpy()*255.0).astype(np.uint8)
        box = (box.cpu()*torch.tensor([W, H, W, H])).numpy()
        with torch.cuda.amp.autocast():
            # print("Extracting global CLIP features...")
            _img = self.clip_preprocess(Image.fromarray(img)).unsqueeze(0)
            global_feat = self.clip_model.encode_image(_img.to(self.device))
            global_feat /= global_feat.norm(dim=-1, keepdim=True)
        global_feat = global_feat.half().to(self.device)
        global_feat = torch.nn.functional.normalize(global_feat, dim=-1)  # --> (1, feature dim)
        feat_dim = global_feat.shape[-1]

        feat_per_roi = []
        roi_nonzero_inds = []
        similarity_scores = []
        for maskidx in range(self.K):
            _cx, _cy, _w, _h = tuple(box[maskidx])  # cxcywh bounding box
            _x, _y, _w, _h = map(int, (_cx - _w/2, _cy - _h/2, _w, _h))
            mask = torch.zeros(H, W, dtype=torch.bool)
            mask[_y: _y + _h, _x: _x + _w] = True
            nonzero_inds = torch.argwhere(mask)
            # Note: Image is (H, W, 3). In SAM output, y coords are along height, x along width
            img_roi = img[_y: _y + _h, _x: _x + _w, :]
            img_roi = Image.fromarray(img_roi)
            img_roi = self.clip_preprocess(img_roi).unsqueeze(0).to(self.device)
            roifeat = self.clip_model.encode_image(img_roi)
            roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
            feat_per_roi.append(roifeat)
            roi_nonzero_inds.append(nonzero_inds)
            _sim = self.cosine_similarity(global_feat, roifeat)
            similarity_scores.append(_sim)

        similarity_scores = torch.cat(similarity_scores)
        softmax_scores = torch.nn.functional.softmax(similarity_scores, dim=0)
        outfeat = torch.zeros(H, W, feat_dim, dtype=torch.half)
        for maskidx in range(self.K):
            _weighted_feat = softmax_scores[maskidx] * global_feat + (1 - softmax_scores[maskidx]) * feat_per_roi[
                maskidx]
            _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
            outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] += _weighted_feat[
                0].detach().cpu().half()
            outfeat[
                roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] = torch.nn.functional.normalize(
                outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]].float(), dim=-1
            ).half()

        outfeat = outfeat.unsqueeze(0).float()  # interpolate is not implemented for float yet in pytorch
        # outfeat = outfeat.permute(0, 3, 1, 2)  # 1, H, W, feat_dim -> 1, feat_dim, H, W
        # outfeat = torch.nn.functional.interpolate(outfeat, [H, W], mode="nearest")
        # outfeat = outfeat.permute(0, 2, 3, 1)  # 1, feat_dim, H, W --> 1, H, W, feat_dim
        outfeat = torch.nn.functional.normalize(outfeat, dim=-1)
        outfeat = outfeat[0].half()  # --> H, W, feat_dim
        return outfeat.to(self.device)

    @torch.no_grad()
    def forward(self, images, text_prompts):
        # images (B, num_views, 10, h, w), text_prompts (B, 1)
        b, nv, _, h, w = images.shape
        rgb_views_images = self.resize(images[:, :, 3:6, :, :].reshape(b*nv, 3, h, w))
        text_prompts = text_prompts.repeat(nv, axis=1).reshape(b*nv)
        boxes = (self.batch_predict(rgb_views_images, text_prompts.tolist(), h, w))  # (b*nv, K, 4)

        assert self.K == 1
        boxes = boxes.squeeze(1)  # (b*nv, 4)

        boxes = (boxes * torch.tensor([w, h, w, h]).to(boxes.device)).int()

        hm = torch.zeros([b*nv, h, w]).cuda()
        hm[torch.arange(hm.size(0)), boxes[:, 1], boxes[:, 0]] = 1.0
        # hm[torch.arange(hm.size(0)), int(h/2), int(w/2)] = 1.0
        hm = hm.reshape(b, nv, h, w)

        out = {"trans": hm}

        return out

    @torch.no_grad()
    def get_predicted_boxes(self, images, text_prompts):
        # images (B, num_views, 10, h, w), text_prompts (B, 1)
        b, nv, _, h, w = images.shape
        rgb_views_images = self.resize(images[:, :, 3:6, :, :].reshape(b*nv, 3, h, w))
        text_prompts = text_prompts.repeat(nv, axis=1).reshape(b*nv)
        boxes = (self.batch_predict(rgb_views_images, text_prompts.tolist(), h, w))  # (b*nv, K, 4)

        # (b*nv, K, 4)
        boxes = (boxes * torch.tensor([w, h, w, h]).to(boxes.device)).int()
        # # (b, nv, K, 4)
        boxes = rearrange(boxes, "(b v) ... -> b v ...", b=b, v=nv)
        return boxes