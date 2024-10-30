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
import random
import trimesh
import open3d as o3d


class GroundingDinoHeatMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.root_path = '../../GroundingDINO/'
        self.model = load_model(self.root_path+"groundingdino/config/GroundingDINO_SwinT_OGC.py",
                                self.root_path+"weights/groundingdino_swint_ogc.pth")
        self.resize_size = 224
        self.resize = T.Resize(self.resize_size)
        # the number of top bounding boxes in each view
        self.K = 10

        self.all_combination = []
        self.num_view = None

        self.device = 'cuda'
        # self.clip_model, self.clip_preprocess = clip.load("RN50", self.device)
        # self.clip_model.to(device=self.device)
        # self.clip_model.eval()

        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)

        # build pc dataset during first forward
        self.dataset_size = None
        self.pc_dataset, self.img_feature_dataset = None, None

    def batch_predict(self, image: torch.Tensor, captions: List, h: int, w: int):
        captions = list(map(preprocess_caption, captions))

        with torch.no_grad():
            self.model.to(self.device)
            image.to(self.device)
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
    def get_predicted_boxes(self, images, text_prompts):
        # get the 3D box which can include the roi in the scene
        # images (B, num_views, 10, h, w), text_prompts (B, 1)
        b, nv, _, h, w = images.shape
        assert h == w
        rgb_views_images = self.resize(images[:, :, 3:6, :, :].reshape(b*nv, 3, h, w))
        text_prompts = text_prompts.repeat(nv, axis=1).reshape(b*nv)
        boxes = (self.batch_predict(rgb_views_images, text_prompts.tolist(), h, w))  # (b*nv, K, 4)

        # (b*nv, K, 4)
        boxes = (boxes * torch.tensor([w, h, w, h]).to(boxes.device)).int()
        # # (b, nv, K, 4)
        boxes = rearrange(boxes, "(b v) ... -> b v ...", b=b, v=nv)

        # from cxcywh boxes to xyxy boxes and find the smallest common box
        boxes_x = torch.clip((boxes[:, :, :, 0] - boxes[:, :, :, 2] / 2), min=0)
        boxes_y = torch.clip((boxes[:, :, :, 1] - boxes[:, :, :, 3] / 2), min=0)
        boxes_x1 = torch.clip(boxes_x + boxes[:, :, :, 2], max=h - 1)
        boxes_y1 = torch.clip(boxes_y + boxes[:, :, :, 3], max=h - 1)
        boxes_x_min = torch.min(boxes_x, dim=-1)[0]
        boxes_y_min = torch.min(boxes_y, dim=-1)[0]
        boxes_x_max = torch.max(boxes_x1, dim=-1)[0]
        boxes_y_max = torch.max(boxes_y1, dim=-1)[0]
        # (bs, num_img, 4)
        boxes = torch.stack([boxes_x_min, boxes_y_min, boxes_x_max, boxes_y_max], dim=-1).to(torch.int)

        # find the 3D common boxes:
        # compare view 0 view 1
        min_y0_x1 = torch.min(torch.stack((boxes[:, 0, 1], h - 1 - boxes[:, 1, 2]), dim=-1), dim=-1)[0]
        boxes[:, 0, 1] = min_y0_x1
        boxes[:, 1, 2] = h - 1 - min_y0_x1
        max_y1_x0 = torch.max(torch.stack((boxes[:, 0, 3], h - 1 - boxes[:, 1, 0]), dim=-1), dim=-1)[0]
        boxes[:, 0, 3] = max_y1_x0
        boxes[:, 1, 0] = h - 1 - max_y1_x0
        # compare view 0 view 2
        min_x0_x1 = torch.min(torch.stack((boxes[:, 0, 0], h - 1 - boxes[:, 2, 2]), dim=-1), dim=-1)[0]
        boxes[:, 0, 0] = min_x0_x1
        boxes[:, 2, 2] = h - 1 - min_x0_x1
        max_x1_x0 = torch.max(torch.stack((boxes[:, 0, 2], h - 1 - boxes[:, 2, 0]), dim=-1), dim=-1)[0]
        boxes[:, 0, 2] = max_x1_x0
        boxes[:, 2, 0] = h - 1 - max_x1_x0
        # compare view 1 view 2
        min_y0 = torch.min(torch.stack((boxes[:, 1, 1], boxes[:, 2, 1]), dim=-1), dim=-1)[0]
        boxes[:, 1, 1] = min_y0
        boxes[:, 2, 1] = min_y0
        max_y1 = torch.max(torch.stack((boxes[:, 1, 3], boxes[:, 2, 3]), dim=-1), dim=-1)[0]
        boxes[:, 1, 3] = max_y1
        boxes[:, 2, 3] = max_y1

        # (bs*num_img, 4)
        boxes = rearrange(boxes, 'b nv ... -> (b nv) ...')
        return boxes

    @torch.no_grad()
    def forward_old(self, images, text_prompts):
        # images (B, num_views, 10, h, w), text_prompts (B, 1)
        b, nv, _, h, w = images.shape
        rgb_views_images = self.resize(images[:, :, 3:6, :, :].reshape(b*nv, 3, h, w))
        text_prompts = text_prompts.repeat(nv, axis=1).reshape(b*nv)
        # cxcy
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
    def forward(self, images, text_prompts):
        # images (B, num_views, 10, h, w), text_prompts (B, 1)
        b, nv, _, h, w = images.shape
        # (bs, num_img, 4) xyxy
        boxes = self.get_predicted_boxes(images, text_prompts).reshape(b, nv, 4)

        # (bs, 2)
        # x0y1
        view_0_min = torch.stack([boxes[:, 0, 0], boxes[:, 0, 3]], dim=-1)
        # x1y0
        view_0_max = torch.stack([boxes[:, 0, 2], boxes[:, 0, 1]], dim=-1)
        # x0y1
        view_1_min = torch.stack([boxes[:, 1, 0], boxes[:, 1, 3]], dim=-1)
        # x1y0
        view_1_max = torch.stack([boxes[:, 1, 2], boxes[:, 1, 1]], dim=-1)
        # x1y1
        view_2_min = torch.stack([boxes[:, 2, 2], boxes[:, 2, 3]], dim=-1)
        # x0y0
        view_2_max = torch.stack([boxes[:, 2, 0], boxes[:, 2, 1]], dim=-1)

        # (bs, 3, 2)
        point_min_img = torch.stack([view_0_min, view_1_min, view_2_min], dim=1)
        point_max_img = torch.stack([view_0_max, view_1_max, view_2_max], dim=1)
        point_img = rearrange(torch.stack([point_min_img, point_max_img], dim=1), 'b x nv p->(b x nv) p')

        hm = torch.zeros([b*2*nv, h, w]).cuda()
        hm[torch.arange(hm.size(0)), point_img[:, 1], point_img[:, 0]] = 1.0
        # (bs*2, 3, h, w)
        hm = hm.reshape(b*2, nv, h, w)

        out = {"trans": hm}

        return out

    def build_pc_dataset(self):
        sample_rate = 20
        dataset_dir = '/bd_byta6000i0/users/jhu/YCB_dataset/models/ycb/'
        all_types = os.listdir(dataset_dir)

        pc_dataset = []
        img_feature_dataset = []
        for i in range(len(all_types)):
            if str(all_types[i]).endswith('cups'):
                continue
            obj_path = os.path.join(dataset_dir, all_types[i], 'clouds/merged_cloud.ply')
            if not os.path.exists(obj_path):
                continue
            pcd = o3d.io.read_point_cloud(obj_path)

            # Apply uniform downsampling
            downsampled_pcd = pcd.uniform_down_sample(sample_rate)

            points = torch.tensor(np.asarray(downsampled_pcd.points)).float()  # Extract the point coordinates
            colors = torch.tensor(np.asarray(downsampled_pcd.colors)).float()  # Extract the colors

            # normalize
            points_max = torch.max(points, dim=0)[0]
            points_min = torch.min(points, dim=0)[0]
            scale = torch.max(points_max)
            # if torch.sum((add_obj_pc_max - add_obj_pc_min) == 0) > 0:
            #     print(f"skip the object in {dir}")
            # else:
            pc = ((points - points_min) / scale * 2 - 1)
            pc_dataset.append(pc)
            img_feature_dataset.append(colors)
        print('finish loading the object point cloud dataset')
        self.dataset_size = len(pc_dataset)
        return pc_dataset, img_feature_dataset

    def filter_pc(self, box_global_point, pc, img_feat):
        b = len(pc)
        # (b, 2, 3)
        box_global_point = rearrange(box_global_point, '(b x) p-> b x p', b=b, x=2)
        xyz_max = box_global_point[:, 1]
        xyz_min = box_global_point[:, 0]

        pc_new = []
        img_feat_new = []
        for i in range(b):
            # (num_point)
            within_range = (torch.sum(pc[i] >= xyz_min[i], dim=-1) == 3) & (torch.sum(pc[i] <= xyz_max[i], dim=-1) == 3)
            pc_new.append(pc[i][within_range])
            img_feat_new.append(img_feat[i][within_range])
        return pc_new, img_feat_new

    # def filter_pc(self, box_global_point, pc, img_feat):
    #     if self.pc_dataset is None:
    #         # build pc dataset
    #         self.pc_dataset, self.img_feature_dataset = self.build_pc_dataset()
    #         print('Build point cloud data set for pc augmentation')
    #
    #     b = len(pc)
    #     # (b, 2, 3)
    #     box_global_point = rearrange(box_global_point, '(b x) p-> b x p', b=b, x=2)
    #     xyz_max = box_global_point[:, 1]
    #     xyz_min = box_global_point[:, 0]
    #
    #     # obj size range from 0.1 to 0.4
    #     obj_size = 0.25+(torch.rand(b, 1).to(box_global_point.device)*2-1)*0.15
    #     obj_size = obj_size.repeat(1, 3)
    #     obj_pos = torch.rand(b, 3).to(box_global_point.device)*2-1
    #     obj_max = obj_pos+obj_size/2
    #     obj_min = obj_pos-obj_size/2
    #     # Check for overlap in the x, y, z dimension
    #     valid_obj = torch.logical_or(torch.sum(xyz_max <= obj_min, dim=-1) > 0,
    #                                  torch.sum(obj_max <= xyz_min, dim=-1) > 0)
    #     obj_ind = torch.randint(low=0, high=self.dataset_size, size=(b, )).to(valid_obj.device)
    #
    #     pc_new = []
    #     img_feat_new = []
    #     for i in range(b):
    #         if valid_obj[i]:
    #             obj_pc = self.pc_dataset[obj_ind[i]].to(obj_size.device)*obj_size[i]
    #             obj_img_feature = self.img_feature_dataset[obj_ind[i]].to(obj_ind.device)
    #
    #             # TODO: add augmentation to the pc
    #             # shift the pc
    #             obj_pc = obj_pc+obj_pos[i]
    #             crop_out_of_space_points = torch.logical_and(torch.sum(obj_pc <= 1.0, dim=-1) == 3,
    #                                                          torch.sum(obj_pc >= -1.0, dim=-1) == 3)
    #             obj_pc = obj_pc[crop_out_of_space_points]
    #             obj_img_feature = obj_img_feature[crop_out_of_space_points]
    #
    #             # TODO:better way to combine two point cloud
    #             pc_new.append(torch.cat([pc[i], obj_pc], dim=0))
    #             img_feat_new.append(torch.cat([img_feat[i], obj_img_feature], dim=0))
    #         else:
    #             pc_new.append(pc[i])
    #             img_feat_new.append(img_feat[i])
    #     return pc_new, img_feat_new

