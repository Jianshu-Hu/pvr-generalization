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


class GroundingDinoHeatMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.root_path = '../../GroundingDINO/'
        self.model = load_model(self.root_path+"groundingdino/config/GroundingDINO_SwinT_OGC.py",
                                self.root_path+"weights/groundingdino_swint_ogc.pth")
        self.resize = T.Resize(800)
        # the number of top bounding boxes in each view
        self.K = 1

        self.all_combination = []
        self.num_view = None

        self.device = 'cuda'
        self.clip_model, self.clip_preprocess = clip.load("RN50", self.device)
        self.clip_model.to(device=self.device)
        self.clip_model.eval()

        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)

    def batch_predict(self, image: torch.Tensor, captions: List):
        captions = list(map(preprocess_caption, captions))

        with torch.no_grad():
            self.model.to('cuda')
            image.to('cuda')
            outputs = self.model(image, captions=captions)

        prediction_logits = outputs["pred_logits"].sigmoid()  # (bs, nq, 256)
        max_logits = prediction_logits.max(dim=-1)[0]  # (bs, nq)
        prediction_boxes = outputs["pred_boxes"]  # (bs, nq, 4)

        # filter the logits
        box_size = prediction_boxes[:, :, 2]*prediction_boxes[:, :, 3]  # (bs, nq)
        filter = box_size > 0.81
        max_logits[filter] = 0.0

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
    def forward_old(self, images, text_prompts):
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

        out = {"trans": final_center}

        return out

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
    def forward_old_2(self, images, text_prompts):
        # images (B, num_views, 10, h, w), text_prompts (B, 1)
        b, nv, _, h, w = images.shape
        rgb_views_images = self.resize(images[:, :, 3:6, :, :].reshape(b*nv, 3, h, w))
        text_prompts = text_prompts.repeat(nv, axis=1).reshape(b*nv)
        boxes = (self.batch_predict(rgb_views_images, text_prompts.tolist()))  # (b*nv, K, 4)

        print("Computing pixel-aligned features...")
        rgb_views_images = images[:, :, 3:6, :, :]
        boxes = boxes.reshape(b, nv, self.K, 4)
        all_hm = []
        for batch_ind in range(b):
            text_inputs = clip.tokenize(text_prompts.tolist()[batch_ind]).to(self.device)
            lang_query = self.clip_model.encode_text(text_inputs)
            sample_hm = []
            for view_ind in range(nv):
                img = rgb_views_images[batch_ind, view_ind]
                box = boxes[batch_ind, view_ind]
                # H, W, feature_dim
                one_view_feature = self.feature_fusion(img, box)
                # H, W, feature_dim
                point_wise_query = lang_query.unsqueeze(0).repeat(h, w, 1)
                # H, W
                point_wise_sim = self.cosine_similarity(point_wise_query, one_view_feature)
                point_wise_sim[torch.isnan(point_wise_sim)] = 0.0
                sample_hm.append(point_wise_sim)

            # nv, H, W
            all_hm.append(torch.stack(sample_hm, dim=0))
        # b, nv, H, W
        all_hm = torch.stack(all_hm, dim=0)

        out = {"trans": all_hm}

        return out

    @torch.no_grad()
    def forward_old(self, images, text_prompts):
        if not hasattr(self, 'count'):
            self.count = 0
        # images (B, num_views, 10, h, w), text_prompts (B, 1)
        b, nv, _, h, w = images.shape
        rgb_views_images = self.resize(images[:, :, 3:6, :, :].reshape(b*nv, 3, h, w))
        text_prompts = text_prompts.repeat(nv, axis=1).reshape(b*nv)
        boxes = (self.batch_predict(rgb_views_images, text_prompts.tolist()))  # (b*nv, K, 4)

        boxes_new = boxes.squeeze(1)  # (b*nv, 4)

        boxes_new = (boxes_new * torch.tensor([w, h, w, h]).to(boxes.device)).int()

        rgb_views_images = images[:, :, 3:6, :, :]
        boxes = boxes.reshape(b, nv, self.K, 4)

        hm = torch.zeros([b, nv, h, w]).cuda()
        for batch_ind in range(b):
            for view_ind in range(nv):
                # image tensor (3, H, W)
                img = rgb_views_images[batch_ind, view_ind]
                # box tensor (K, 4)
                box = boxes[batch_ind, view_ind]
                img = (img.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                box = (box.cpu() * torch.tensor([w, h, w, h])).numpy()

                maskidx = 0
                annotated_frame = annotate(image_source=img, boxes=(boxes[batch_ind, view_ind, maskidx:maskidx+1]).cpu(),
                                           logits=torch.tensor([0.0]), phrases=['drawer'])
                annotated_frame = Image.fromarray(annotated_frame)
                annotated_frame.save(f'annotated_{self.count}_image_view{view_ind}_mask{maskidx}.jpg')

                hm[batch_ind, view_ind, int(box[0, 1]), int(box[0, 0])] = 1.0
        self.count += 1

        out = {"trans": hm}

        return out

    @torch.no_grad()
    def forward(self, images, text_prompts):
        # images (B, num_views, 10, h, w), text_prompts (B, 1)
        b, nv, _, h, w = images.shape
        rgb_views_images = self.resize(images[:, :, 3:6, :, :].reshape(b*nv, 3, h, w))
        text_prompts = text_prompts.repeat(nv, axis=1).reshape(b*nv)
        boxes = (self.batch_predict(rgb_views_images, text_prompts.tolist()))  # (b*nv, K, 4)

        boxes = boxes.squeeze(1)  # (b*nv, 4)

        boxes = (boxes * torch.tensor([w, h, w, h]).to(boxes.device)).int()

        hm = torch.zeros([b*nv, h, w]).cuda()
        hm[torch.arange(hm.size(0)), boxes[:, 1], boxes[:, 0]] = 1.0
        hm = hm.reshape(b, nv, h, w)

        out = {"trans": hm}

        return out