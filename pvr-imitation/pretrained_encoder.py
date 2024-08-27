import numpy as np
import os
import timm
import torch
from PIL import Image


def load_pretrained_model(encoder_name, checkpoint_path, input_type=np.ndarray, *args, **kwargs):
    """
    Load the pretrained model based on the config corresponding to the encoder_name
    """

    if checkpoint_path is not None:
        if not os.path.isabs(checkpoint_path):
            model_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
            checkpoint_path = os.path.join(model_base_dir, checkpoint_path)
        model = timm.create_model(encoder_name, pretrained=True, num_classes=0, pretrained_cfg={'file': checkpoint_path})
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

    return model, embedding_dim, final_transforms(transforms)
