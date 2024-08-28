import numpy as np
import os
import timm
import torch
from PIL import Image
import vc_models
from vc_models.models.vit import model_utils
import torchvision
encoder_name_dict = {'dinov2_base': "vit_base_patch14_dinov2.lvd142m",
                     "vc1_base": "vc1_vitb", "vc1_large": "vc1_vitl"}


def load_pretrained_model(encoder_name, root_dir, input_type=np.ndarray, *args, **kwargs):
    """
    Load the pretrained model based on the config corresponding to the encoder_name
    """
    encoder_name = encoder_name_dict[encoder_name]
    if encoder_name.startswith('vc1'):
        # load vc1_vitb or vc1_vitl
        model, embedding_dim, transforms, model_info = model_utils.load_model(encoder_name)

        def final_transforms(transforms):
            if input_type == np.ndarray:
                to_tensor = torchvision.transforms.ToTensor()
                return lambda input: transforms(to_tensor(input)).unsqueeze(0)
            else:
                return transforms
        transforms = final_transforms(transforms)
    else:
        # load general encoder from timm
        checkpoint_path = os.path.join(root_dir, 'pvr_ckpts', encoder_name+'.bin')
        if os.path.exists(checkpoint_path):
            model = timm.create_model(encoder_name, pretrained=True, num_classes=0,
                                      pretrained_cfg={'file': checkpoint_path})
        else:
            model = timm.create_model(encoder_name, pretrained=True, num_classes=0)
        model.eval()
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

        with torch.no_grad():
            zero_img = Image.new("RGB", (100, 100))
            transformed_img = transforms(zero_img).unsqueeze(0)
            embedding_dim = model.eval()(transformed_img).shape[1]

        def final_transforms(transforms):
            if input_type == np.ndarray:
                return lambda input: transforms(Image.fromarray(input)).unsqueeze(0)
            else:
                return transforms
        transforms = final_transforms(transforms)
    return model, embedding_dim, transforms
