from PIL import Image
import numpy as np
from rlbench.backend import utils, const


def retrieve_obs(obs, data_cfg):
    if data_cfg.images.rgb:
        if data_cfg.cameras.left_shoulder:
            image = Image.fromarray(obs.left_shoulder_rgb)
        if data_cfg.cameras.right_shoulder:
            image = Image.fromarray(obs.right_shoulder_rgb)
        if data_cfg.cameras.overhead:
            image = Image.fromarray(obs.overhead_rgb)
        if data_cfg.cameras.wrist:
            image = Image.fromarray(obs.wrist_rgb)
        if data_cfg.cameras.front:
            image = Image.fromarray(obs.front_rgb)

    if data_cfg.images.depth:
        if data_cfg.cameras.left_shoulder:
            image = utils.float_array_to_rgb_image(
                obs.left_shoulder_depth, scale_factor=const.DEPTH_SCALE
            )
        if data_cfg.cameras.right_shoulder:
            image = utils.float_array_to_rgb_image(
                obs.right_shoulder_depth, scale_factor=const.DEPTH_SCALE
            )
        if data_cfg.cameras.overhead:
            image = utils.float_array_to_rgb_image(
                obs.overhead_depth, scale_factor=const.DEPTH_SCALE
            )
        if data_cfg.cameras.wrist:
            image = utils.float_array_to_rgb_image(
                obs.wrist_depth, scale_factor=const.DEPTH_SCALE
            )
        if data_cfg.cameras.front:
            image = utils.float_array_to_rgb_image(
                obs.front_depth, scale_factor=const.DEPTH_SCALE
            )

    if data_cfg.images.mask:
        if data_cfg.cameras.left_shoulder:
            image = Image.fromarray(
                (obs.left_shoulder_mask * 255).astype(np.uint8)
            )

        if data_cfg.cameras.right_shoulder:
            image = Image.fromarray(
                (obs.right_shoulder_mask * 255).astype(np.uint8)
            )

        if data_cfg.cameras.overhead:
            image = Image.fromarray(
                (obs.overhead_mask * 255).astype(np.uint8)
            )

        if data_cfg.cameras.wrist:
            image = Image.fromarray(
                (obs.wrist_mask * 255).astype(np.uint8)
            )

        if data_cfg.cameras.front:
            image = Image.fromarray(
                (obs.front_mask * 255).astype(np.uint8)
            )
    return np.array(image)
