import os
from typing import Literal, get_args

__file_dir__ = os.path.dirname(__file__)
import torch
from torchvision import transforms
from dinov3.eval.text.tokenizer import Tokenizer
from dinov3.eval.text.dinotxt_model import DINOTxt

BACKBONE_MODELS = Literal[
    # ViT
    "dinov3_vits16",
    "dinov3_vits16plus",
    "dinov3_vitb16",
    "dinov3_vitl16",
    "dinov3_vith16plus",
    "dinov3_vit7b16",
    # ConvNEXT
    "dinov3_convnext_tiny",
    "dinov3_convnext_small",
    "dinov3_convnext_base" "dinov3_convnext_large",
]


def make_transform(resize_size: int | list[int] = 768, for_satellite: bool = False):
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    if for_satellite:
        normalize = transforms.Normalize(
            mean=(0.430, 0.411, 0.296),
            std=(0.213, 0.156, 0.143),
        )
    else:
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    return transforms.Compose([to_tensor, resize, normalize])


def get_dino_text(
    dinotxt_weights: str,
    backbone_weights: str,
    bpe_path: str,
    resize_size: int | list[int] = 768,
    model_name: str = "dinov3_vitl16_dinotxt_tet1280d20h24l",
) -> tuple[DINOTxt, Tokenizer, transforms.Compose]:
    model, tokenizer = torch.hub.load(
        __file_dir__,
        model_name,
        dinotxt_weights=dinotxt_weights,
        backbone_weights=backbone_weights,
        bpe_path_or_url=bpe_path,
        source="local",
    )  # type: ignore
    image_processor = make_transform(resize_size, False)
    return model, tokenizer, image_processor


def get_dino_backbone(
    weights: str,
    model_name: BACKBONE_MODELS,
    resize_size: int | list[int] = 768,
    for_satellite: bool = False,
) -> tuple[DINOTxt, transforms.Compose]:
    model = torch.hub.load(__file_dir__, model_name, source="local", weights=weights)
    transforms = make_transform(resize_size, for_satellite)
    return model, transforms

def list_models()->list[str]: 
    return get_args(BACKBONE_MODELS)