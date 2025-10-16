# -*- coding: utf-8 -*-
import os

import numpy as np
import torch
from PIL import Image
from torchvision.models.segmentation import (
    deeplabv3_resnet50, DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet101, DeepLabV3_ResNet101_Weights,
    lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
)

from style_trans.utils import IO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DeepLabModel(object):
    def __init__(self, model_name="deeplabv3_resnet50", device=None, verbose=False):
        self.model_name = model_name
        self.verbose = verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        alias = {
            "mobilenetv2_coco_voctrainaug": "lraspp_mobilenet_v3_large",
            "mobilenetv2_coco_voctrainval": "lraspp_mobilenet_v3_large",
            "xception_coco_voctrainaug": "deeplabv3_resnet101",
            "xception_coco_voctrainval": "deeplabv3_resnet101",
        }
        resolved = alias.get(model_name, model_name)

        if resolved == "deeplabv3_resnet50":
            weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
            self.model = deeplabv3_resnet50(weights=weights).to(self.device).eval()
            self.preprocess = weights.transforms()
        elif resolved == "deeplabv3_resnet101":
            weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
            self.model = deeplabv3_resnet101(weights=weights).to(self.device).eval()
            self.preprocess = weights.transforms()
        elif resolved == "lraspp_mobilenet_v3_large":
            weights = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
            self.model = lraspp_mobilenet_v3_large(weights=weights).to(self.device).eval()
            self.preprocess = weights.transforms()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        if self.verbose:
            print(f"[INFO] Loaded {resolved} on {self.device}")

    @torch.inference_mode()
    def run(self, pil_image):
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        out = self.model(tensor)["out"]
        seg = out.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        return seg


class SemanticSegmentation(object):

    def __init__(self, model_name="mobilenetv2_coco_voctrainaug", verbose=False):
        self.model_name = model_name
        self.verbose = verbose
        self.path = IO()

    def generate_mask(self, input_file):
        self.input_file = input_file
        model = self._fetch_deeplab_model()
        self._generate_mask(model)
        if os.path.exists(self.path.mask_image_path) and self.verbose:
            print("Mask Created!")

    def _fetch_deeplab_model(self):
        if self.verbose:
            print("Loading PyTorch DeepLab model (weights auto-download)...")
        model = DeepLabModel(self.model_name, verbose=self.verbose)
        if self.verbose:
            print("Model Loaded Successfully!")
        return model

    def _generate_mask(self, model):
        img = Image.open(self.input_file)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.verbose:
            print(f"Running DeepLab on {self.input_file} ...")
        seg_map = model.run(img)
        self._generate_semantic_segmentation(seg_map, img)

    def _generate_semantic_segmentation(self, seg_map, original_image):
        if seg_map.shape[0] != original_image.size[1] or seg_map.shape[1] != original_image.size[0]:
            seg_map = np.array(
                Image.fromarray(seg_map).resize(original_image.size, Image.NEAREST)
            )

        PERSON_CLASS = 15
        mask = (seg_map == PERSON_CLASS).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask, mode="L").convert("RGB")
        mask_img.save(self.path.mask_image_path)
        if self.verbose:
            print(f"Mask saved to {self.path.mask_image_path}")


if __name__ == "__main__":
    seg = SemanticSegmentation("deeplabv3_resnet50", verbose=True)
    seg.generate_mask("/Users/lethian/Documents/semester_114/image processing/final_project/Naive-Background-Style-Transfer/input_images/Content/portrait.jpg")
