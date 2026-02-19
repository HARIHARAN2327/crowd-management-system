import argparse
import random
import time
from pathlib import Path
from threading import Event

import torch
import torchvision.transforms as standard_transforms
import numpy as np
import cv2
from PIL import Image

from engine import *
from models import build_model
import warnings

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  Args Parser
# ─────────────────────────────────────────────
def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    parser.add_argument('--backbone', default='vgg16_bn', type=str)
    parser.add_argument('--row',  default=2, type=int)
    parser.add_argument('--line', default=2, type=int)
    parser.add_argument('--output_dir',  default='')
    parser.add_argument('--weight_path', default='')
    parser.add_argument('--gpu_id', default=0, type=int)
    return parser


parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
print(f"[INFO] Args: {args}")


# ─────────────────────────────────────────────
#  Device — set ONCE globally
# ─────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")
if device.type == 'cuda':
    print(f"[INFO] GPU : {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ─────────────────────────────────────────────
#  Model — loaded ONCE globally
#  FIX: was being reloaded inside every function
#  call which caused massive slowdowns
# ─────────────────────────────────────────────
print("[INFO] Loading P2PNet model...")
model = build_model(args)
model.to(device)                                                         # ✅ Move to GPU
checkpoint = torch.load(Path('weights/SHTechA.pth'), map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()                                                             # ✅ Eval mode
print("[INFO] Model ready")


# ─────────────────────────────────────────────
#  Transform — created ONCE globally
# ─────────────────────────────────────────────
transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

THRESHOLD = 0.5    # P2PNet confidence threshold


# ─────────────────────────────────────────────
#  Helper: run inference on a single BGR frame
#  Returns (points, count)
# ─────────────────────────────────────────────
def _infer_frame(frame_bgr):
    """
    frame_bgr : numpy BGR image (any size, will be snapped to mult of 128)
    returns   : (points list, predict_cnt int)
    """
    # Snap dimensions to nearest multiple of 128 (P2PNet requirement)
    h, w = frame_bgr.shape[:2]
    new_w = (w // 128) * 128
    new_h = (h // 128) * 128
    if new_w == 0 or new_h == 0:
        return [], 0
    frame_bgr = cv2.resize(frame_bgr, (new_w, new_h))

    with torch.no_grad():                                                # ✅ No gradients
        img_tensor = transform(frame_bgr)                               # already a tensor
        samples    = img_tensor.unsqueeze(0).to(device)                 # ✅ CPU → GPU

        outputs        = model(samples)
        outputs_scores = torch.nn.functional.softmax(
            outputs['pred_logits'], -1
        )[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]

        mask        = outputs_scores > THRESHOLD
        points      = outputs_points[mask].detach().cpu().numpy().tolist()
        predict_cnt = int(mask.sum())

    print(f"[INFO] Count: {predict_cnt}")
    return points, predict_cnt


# ─────────────────────────────────────────────
#  Webcam prediction
#  FIX: removed cv2.imshow (crashes in Flask)
#  FIX: model no longer reloaded every call
#  FIX: torch.Tensor(img) → img directly (transform already returns tensor)
#  FIX: duplicate softmax/outputs_points lines removed
# ─────────────────────────────────────────────
def get_prediction_webcam(event: Event):
    print("[INFO] Starting webcam prediction...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Reduce buffer lag

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[ERROR] Failed to read frame")
            break

        # Scale down for inference
        scale_factor = 0.4
        inf_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        img_raw   = inf_frame.copy()

        # Run inference
        points, predict_cnt = _infer_frame(inf_frame)

        # Draw predictions on frame
        for p in points:
            cv2.circle(img_raw, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)   # red dot
            cv2.circle(img_raw, (int(p[0]), int(p[1])), 5, (255, 255, 255), 1) # white ring

        # Resize to display size and overlay count
        img_to_draw = cv2.resize(img_raw, (960, 540))
        cv2.putText(img_to_draw, f"Count: {predict_cnt}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img_to_draw, f"Device: {device.type.upper()}",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # ── REMOVED cv2.imshow — crashes in Flask/server context ──
        # Save frame to static folder so Flask can serve it
        cv2.imwrite('static/webcam_latest.jpg', img_to_draw)

        # Stop if event is signalled from Flask
        if event.is_set():
            print('[INFO] Webcam thread stopped.')
            break

    cap.release()


# ─────────────────────────────────────────────
#  Image / Video file prediction
#  FIX: model no longer reloaded every call
#  FIX: indentation bug in video branch fixed
#       (inference code was inside except block)
#  FIX: torch.Tensor(img) → img directly
#  FIX: duplicate softmax lines removed
#  FIX: cv2.imshow removed (crashes in Flask)
# ─────────────────────────────────────────────
def get_prediction(file):

    # ── Video file (.mp4) ──────────────────────
    if file.endswith(".mp4"):
        cap = cv2.VideoCapture(file)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {file}")
            return 0, ''

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        scale_factor = 0.4
        last_count   = 0

        while True:
            ret, frame = cap.read()

            # FIX: check ret BEFORE using frame (was crashing on end of video)
            if not ret or frame is None:
                print("[INFO] Video ended")
                cap.release()
                break

            # Scale for inference
            inf_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            img_raw   = inf_frame.copy()

            # Run inference — FIX: was inside except block before, never ran
            points, predict_cnt = _infer_frame(inf_frame)
            last_count = predict_cnt

            # Draw predictions
            for p in points:
                cv2.circle(img_raw, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

            img_to_draw = cv2.resize(img_raw, (960, 540))
            cv2.putText(img_to_draw, f"Count: {predict_cnt}",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Save latest frame so Flask can display it
            cv2.imwrite('static/video_latest.jpg', img_to_draw)

        return last_count, 'static/video_latest.jpg'

    # ── Image file ─────────────────────────────
    else:
        img_raw = Image.open(file).convert('RGB')

        # Snap to multiple of 128
        width, height = img_raw.size
        new_width  = (width  // 128) * 128
        new_height = (height // 128) * 128
        img_raw = img_raw.resize((new_width, new_height), Image.LANCZOS)

        img_bgr = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)

        # Run inference — FIX: torch.Tensor(img) replaced with img directly
        with torch.no_grad():                                            # ✅ No gradients
            img_tensor = transform(img_raw)
            samples    = img_tensor.unsqueeze(0).to(device)             # ✅ CPU → GPU

            outputs        = model(samples)
            outputs_scores = torch.nn.functional.softmax(
                outputs['pred_logits'], -1
            )[:, :, 1][0]
            outputs_points = outputs['pred_points'][0]

            mask        = outputs_scores > THRESHOLD
            points      = outputs_points[mask].detach().cpu().numpy().tolist()
            predict_cnt = int(mask.sum())

        print(f"[INFO] Image count: {predict_cnt}")

        h, w = img_bgr.shape[:2]

        # ── Panel 1: Original image with red dots ──
        original_with_dots = img_bgr.copy()
        for p in points:
            x0, y0 = int(p[0]), int(p[1])
            if 0 <= x0 < w and 0 <= y0 < h:
                cv2.circle(original_with_dots, (x0, y0), 2, (0, 0, 255), -1)
        cv2.putText(original_with_dots, f"Count: {predict_cnt}",
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ── Panel 2: Heatmap ───────────────────────
        heat = np.zeros((h, w), dtype=np.float32)
        for p in points:
            x0, y0 = int(p[0]), int(p[1])
            if 0 <= x0 < w and 0 <= y0 < h:
                heat[y0, x0] = 1.0
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=10, sigmaY=10)
        if heat.max() > 0:
            heat = heat / heat.max()
        heatmap_bgr = cv2.applyColorMap(
            (heat * 255).astype(np.uint8), cv2.COLORMAP_JET
        )

        # ── Panel 3: Scatter on white ──────────────
        scatter = np.full((h, w, 3), 255, dtype=np.uint8)
        for p in points:
            x0, y0 = int(p[0]), int(p[1])
            if 0 <= x0 < w and 0 <= y0 < h:
                cv2.circle(scatter, (x0, y0), 2, (0, 0, 0), -1)

        # ── 2x2 grid visualization ─────────────────
        # [original_with_dots | heatmap]
        # [scatter             | blank ]
        blank = np.full((h, w, 3), 255, dtype=np.uint8)
        top   = np.hstack((original_with_dots, heatmap_bgr))
        bottom = np.hstack((scatter, blank))
        vis   = np.vstack((top, bottom))

        x       = random.randint(1, 100000)
        density = f'static/density_map_p2p_{x}.jpg'
        cv2.imwrite(density, vis)

        return predict_cnt, density