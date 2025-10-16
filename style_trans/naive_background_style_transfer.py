# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import imageio
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models

from style_trans.semantic_segmentation import SemanticSegmentation
from style_trans.utils import IO

# Pillow 10+ resample enum fallback
try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS


# =============== VGG19 feature extractor (PyTorch) ===============
# Map các layer theo VGG19 của torchvision.features:
# conv1_1: 0, conv1_2: 2, pool: 4
# conv2_1: 5, conv2_2: 7, pool: 9
# conv3_1:10, conv3_2:12, conv3_3:14, conv3_4:16, pool:18
# conv4_1:19, conv4_2:21, conv4_3:23, conv4_4:25, pool:27
# conv5_1:28, conv5_2:30, conv5_3:32, conv5_4:34, pool:36

VGG_IDX = {
    "block1_conv1": 0,
    "block2_conv1": 5,
    "block3_conv1": 10,
    "block4_conv1": 19,
    "block5_conv1": 28,
    "block5_conv2": 30,
}

class VGG19Features(nn.Module):
    """
    Trả về danh sách kích hoạt theo thứ tự:
    [style_layers..., content_layers...]
    """
    def __init__(self, style_layers, content_layers):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.features = vgg.eval()
        for p in self.features.parameters():
            p.requires_grad = False

        # Chuyển tên block->index
        self.style_ids = [VGG_IDX[name] for name in style_layers]
        self.content_ids = [VGG_IDX[name] for name in content_layers]
        self.return_ids = sorted(set(self.style_ids + self.content_ids))
        # map từ id -> vị trí trong output
        self.style_positions = [self.return_ids.index(i) for i in self.style_ids]
        self.content_positions = [self.return_ids.index(i) for i in self.content_ids]

    def forward(self, x):
        outputs = []
        cur = x
        next_idx = 0
        for i, layer in enumerate(self.features):
            cur = layer(cur)
            # nếu index này nằm trong return_ids thì lưu
            if next_idx < len(self.return_ids) and i == self.return_ids[next_idx]:
                outputs.append(cur)
                next_idx += 1
            # tối ưu sớm nếu đã có đủ
            if next_idx >= len(self.return_ids):
                break
        # reorder theo [style..., content...]
        style_feats = [outputs[pos] for pos in self.style_positions]
        content_feats = [outputs[pos] for pos in self.content_positions]
        return style_feats + content_feats


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def pil_to_tensor(img_pil):
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    arr = np.array(img_pil).astype(np.float32) / 255.0  # HWC in [0,1]
    arr = np.transpose(arr, (2,0,1))  # CHW
    tensor = torch.from_numpy(arr).unsqueeze(0)  # [1,3,H,W]
    return tensor

def tensor_to_uint8_img(t):
    # t: [1,3,H,W] in [0,1]
    t = t.clamp(0.0, 1.0).detach().cpu().squeeze(0).numpy()
    t = np.transpose(t, (1,2,0)) * 255.0
    t = np.clip(t, 0, 255).astype(np.uint8)
    return t

def normalize(t):
    return (t - IMAGENET_MEAN.to(t.device)) / IMAGENET_STD.to(t.device)

def preprocess_tf(self, x01):
    x = x01 * 255.0
    x = torch.flip(x, dims=[1])  # RGB->BGR
    mean = torch.tensor([103.939, 116.779, 123.680], device=x.device).view(1,3,1,1)
    return x - mean


def denormalize(t):
    return t * IMAGENET_STD.to(t.device) + IMAGENET_MEAN.to(t.device)


def gram_matrix(feats):
    # feats: [B,C,H,W]
    B, C, H, W = feats.shape
    f = feats.view(B, C, -1)  # [B, C, HW]
    G = torch.bmm(f, f.transpose(1,2))  # [B, C, C]
    return G / (C * H * W)


class NaiveBackgroundStyleTransfer():

    def __init__(self,
                 number_of_epochs=1000,
                 content_weight=1e2,
                 style_weight=1e-1,
                 model_name="mobilenetv2_coco_voctrainaug",
                 enable_gpu=False,
                 verbose=False):

        self.number_of_epochs = number_of_epochs
        self.content_weight = content_weight
        self.style_weight = style_weight

        self.content_layers = ["block5_conv2"]
        self.style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]

        self.model_name = model_name
        self.verbose = verbose

        self.device = torch.device("cuda" if (enable_gpu and torch.cuda.is_available()) else "cpu")
        if self.verbose:
            print(f"[INFO] Using device: {self.device}")

        self.path = IO()

        self.vgg_feats = VGG19Features(self.style_layers, self.content_layers).to(self.device).eval()

        self.to_tensor = pil_to_tensor

    def perform(self, content_file, style_file):
        return self._perform(content_file, style_file)

    def show_image(self, image_np_uint8):
        img = Image.fromarray(image_np_uint8)
        os.makedirs(self.path.output_image_path, exist_ok=True)
        img.save(os.path.join(self.path.output_image_path, self.path.output_file_name))
        return

    def generate_gif(self, speed):
        assert os.path.exists(self.path.transition_output_path), \
            "Naive Background Style Transfer not performed. No Transition Images available."
        filenames = [os.path.join(self.path.transition_output_path, f)
                     for f in os.listdir(self.path.transition_output_path)
                     if f.endswith(".jpg")]
        filenames.sort()
        frames = [imageio.imread(f) for f in filenames]
        os.makedirs(self.path.output_gif_path, exist_ok=True)
        imageio.mimsave(os.path.join(self.path.output_gif_path, self.path.gif_file_name),
                        frames, duration=float(speed))

    # ------------ Internal helpers ------------
    def _load_image_tensor(self, filename):
        img = Image.open(filename)
        img = img.resize((img.size[0], img.size[1]), RESAMPLE_LANCZOS)  # giữ size
        t = self.to_tensor(img)
        return t

    def _generate_mask(self, input_file):
        seg = SemanticSegmentation(model_name=self.model_name, verbose=self.verbose)
        seg.generate_mask(input_file)

    def _load_mask(self, input_file):
        self._generate_mask(input_file)
        mask = Image.open(self.path.mask_image_path).convert('L')
        mask = mask.resize((mask.size[0], mask.size[1]), RESAMPLE_LANCZOS)
        mask = np.array(mask, dtype=np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0) >= 1.0
        return mask

    def _get_activations(self, image_tensor_norm):
        outputs = self.vgg_feats(image_tensor_norm)
        n_style = len(self.style_layers)
        style_feats = outputs[:n_style]
        content_feats = outputs[n_style:]
        return content_feats, style_feats

    def _content_loss(self, c_act, g_act):
        return F.mse_loss(g_act, c_act)

    def _style_loss(self, s_act, g_act):
        Gs = gram_matrix(s_act)
        Gg = gram_matrix(g_act)
        return F.mse_loss(Gg, Gs)

    def _perform(self, content_file, style_file):

        content_img = self._load_image_tensor(content_file).to(self.device)  # [1,3,H,W], [0,1]
        style_img   = self._load_image_tensor(style_file).to(self.device)

        # # content_norm = normalize(content_img.clone())
        # # style_norm   = normalize(style_img.clone())
        # content_norm = preprocess_tf(self, content_img.clone().to(self.device))
        # style_norm   = preprocess_tf(self, style_img.clone().to(self.device))
        #
        # with torch.no_grad():
        #     content_acts, _ = self._get_activations(content_norm)
        #     _, style_acts   = self._get_activations(style_norm)


        with torch.no_grad():
            content_acts, _ = self._get_activations(content_img)
            _, style_acts = self._get_activations(style_img)

        generated = torch.nn.Parameter(content_img.clone().to(self.device))
        optimizer = optim.Adam([generated], lr=0.02)

        # Mask foreground
        mask_bool = self._load_mask(content_file).to(self.device)
        mask3 = mask_bool.expand(-1, 3, -1, -1)  # [1,3,H,W]

        def make_gif_marks(E):
            base = [0,1,2,3,4,5,10,15,20,25,40,60,80,100,120,140,160,180,200,220,250,280,
                    300,330,360,390,420,450,480,510,540,580,620,660,700,725,750,775,800,825,875,900,950,E-1]
            return sorted(set([i for i in base if 0 <= i < E]))
        gif_interval = set(make_gif_marks(self.number_of_epochs))

        num_rows, num_cols = 2, 5
        display_interval = max(1, self.number_of_epochs // (num_rows*num_cols))
        intermediate = []

        os.makedirs(self.path.transition_output_path, exist_ok=True)
        os.makedirs(self.path.output_image_path, exist_ok=True)

        final_img_uint8 = None
        best_loss = float('inf')

        for epoch in range(self.number_of_epochs):

            optimizer.zero_grad()

            gen_norm = normalize(generated)
            gen_c_acts, gen_s_acts = self._get_activations(gen_norm)
            # Content loss
            w_c = 1.0 / float(len(self.content_layers))
            c_loss = 0.0
            for a_c, a_g in zip(content_acts, gen_c_acts):
                c_loss = c_loss + w_c * self._content_loss(a_c, a_g)

            # Style loss
            w_s = 1.0 / float(len(self.style_layers))
            s_loss = 0.0
            for a_s, a_gs in zip(style_acts, gen_s_acts):
                s_loss = s_loss + w_s * self._style_loss(a_s, a_gs)

            loss = self.content_weight * c_loss + self.style_weight * s_loss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                generated.clamp_(0.0, 1.0)

            cur_loss = loss.item()
            if cur_loss < best_loss:
                best_loss = cur_loss
                final_vis = generated.detach().clone()
                final_vis[mask3] = content_img[mask3]
                final_img_uint8 = tensor_to_uint8_img(final_vis)

            if self.verbose and (epoch % max(1, self.number_of_epochs // 20) == 0 or epoch == self.number_of_epochs-1):
                print(f"Epoch {epoch+1}/{self.number_of_epochs} - "
                      f"Loss: {cur_loss:.4f} (C:{(self.content_weight*c_loss).item():.4f}, S:{(self.style_weight*s_loss).item():.4f})")

            # GIF frame
            if self.verbose and (epoch in gif_interval):
                trans = generated.detach().clone()
                trans[mask3] = content_img[mask3]
                trans_img = tensor_to_uint8_img(trans)
                Image.fromarray(trans_img).save(
                    os.path.join(self.path.transition_output_path,
                                 f"{self.path.transition_output_file_name}{100000+epoch+1}.jpg")
                )
                plt.close()

            # Intermediate grid
            if self.verbose and (epoch % display_interval == 0):
                inter = tensor_to_uint8_img(generated.detach())
                intermediate.append(inter)

                plt.figure(figsize=(14, 4))
                for i, im in enumerate(intermediate[:num_rows*num_cols]):
                    plt.subplot(num_rows, num_cols, i+1)
                    plt.imshow(im)
                    plt.xticks([]); plt.yticks([])
                plt.tight_layout()
                plt.savefig(os.path.join(self.path.output_image_path, self.path.intermediate_images_file_name))
                plt.close()

        return final_img_uint8
