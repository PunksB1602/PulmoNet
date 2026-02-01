import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import pandas as pd

from models import get_model

LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)

        # Register backward hook (full if available)
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)
        else:
            self._bwd_handle = self.target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def close(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def generate(self, input_tensor, class_idx: int):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam[0, 0]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy()


def pick_target_layer(model_name: str, model):
    name = model_name.lower()
    if name == "resnet50":
        return model.layer4[-1]
    if name == "densenet121":
        return model.features.norm5
    if name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError("Unsupported model for Grad-CAM target layer.")


def resolve_default_image(data_dir: str) -> str:
    # Get first image path from valid.csv if --image is not specified
    data_dir = Path(data_dir)
    csv_path = data_dir / "CheXpert-v1.0-small" / "valid.csv"
    df = pd.read_csv(csv_path)

    p = str(df.iloc[0][df.columns[0]]).strip().lstrip("./\\")
    # Strip dataset prefix if present
    p = p.replace("CheXpert-v1.0-small/", "").replace("CheXpert-v1.0-small\\", "")

    # root_dir is data_dir, not data_dir/CheXpert-v1.0-small
    return str((data_dir / p).resolve())


def overlay_and_save(img_path: str, cam: np.ndarray, save_path: str, size: int = 224):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"cv2.imread failed for: {img_path}")

    img_bgr = cv2.resize(img_bgr, (size, size))

    cam_r = cv2.resize(cam, (size, size))
    heat = np.uint8(255 * cam_r)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    out = cv2.addWeighted(img_bgr, 0.6, heat, 0.4, 0)
    cv2.imwrite(save_path, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["densenet121", "resnet50", "efficientnet_b0"])
    parser.add_argument("--class_idx", type=int, default=None,
                        help="If set, generates Grad-CAM only for this class index (0-4).")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an image. If not provided, uses first image from valid.csv.")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Folder containing CheXpert-v1.0-small/")
    parser.add_argument("--artifacts_dir", type=str, default="./artifacts",
                        help="Folder containing checkpoints and where to save CAMs.")
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.artifacts_dir) / args.model / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = get_model(args.model, num_classes=len(LABELS), pretrained=False)

    ckpt_obj = torch.load(ckpt_path, map_location=device)
    state = ckpt_obj["model_state"] if isinstance(ckpt_obj, dict) and "model_state" in ckpt_obj else ckpt_obj
    model.load_state_dict(state)

    model.to(device)
    model.eval()

    # Disable in-place ops to avoid hook/backward issues for GradCAM
    for m in model.modules():
        if hasattr(m, "inplace"):
            try:
                m.inplace = False
            except Exception:
                pass

    target_layer = pick_target_layer(args.model, model)
    cam_engine = GradCAM(model, target_layer)

    img_path = args.image if args.image is not None else resolve_default_image(args.data_dir)
    if not os.path.exists(img_path):
        cam_engine.close()
        raise FileNotFoundError(f"Image not found: {img_path}")

    preprocess = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    out_dir = Path(args.artifacts_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = [args.class_idx] if args.class_idx is not None else list(range(len(LABELS)))

    for idx in indices:
        cam = cam_engine.generate(x, class_idx=idx)
        label_name = LABELS[idx].replace(" ", "_")
        save_path = out_dir / f"gradcam_{label_name}.png"
        overlay_and_save(img_path, cam, str(save_path), size=args.size)
        print(f"Saved: {save_path}")

    cam_engine.close()


if __name__ == "__main__":
    main()
