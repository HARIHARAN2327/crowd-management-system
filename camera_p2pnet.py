import cv2
import json
import argparse
import time
import mysql.connector
from pathlib import Path
from threading import Thread, Lock
from collections import deque

import requests

import torch
import torchvision.transforms as standard_transforms
import numpy as np

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
    parser.add_argument('--row', default=2, type=int)
    parser.add_argument('--line', default=2, type=int)
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--weight_path', default='')
    parser.add_argument('--gpu_id', default=0, type=int)
    return parser


parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
args = parser.parse_args()


# ─────────────────────────────────────────────
#  Device Setup
# ─────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")
if device.type == 'cuda':
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ─────────────────────────────────────────────
#  Model Setup
# ─────────────────────────────────────────────
model = build_model(args)
model.to(device)
checkpoint = torch.load(Path('weights/SHTechA.pth'), map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()
print("[INFO] Model loaded and ready")


# ─────────────────────────────────────────────
#  Transform
# ─────────────────────────────────────────────
transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ─────────────────────────────────────────────
#  Output Dimensions
#  OUT_W x OUT_H = what the frontend displays
#  INF_W x INF_H = what the model sees (smaller, must be multiple of 128)
# ─────────────────────────────────────────────
OUT_W     = 960          # Display width
OUT_H     = 540          # Display height
INF_SCALE = 0.4          # Scale factor for inference frame
THRESHOLD = 0.5          # P2PNet confidence threshold

OFFICER_ALERT_URL = "http://localhost:8081/api/alert"
OFFICER_ALERT_COOLDOWN_SEC = 10.0
OFFICER_REGION_URL = "http://localhost:8081/api/region"
OFFICER_REGION_UPDATE_INTERVAL_SEC = 1.0
CAMERA_ID = "CAM-01"
CAMERA_LATITUDE = 0.0
CAMERA_LONGITUDE = 0.0

# ─────────────────────────────────────────────
#  Stampede Risk (heuristic) knobs
#  NOTE: Requires scene calibration via SCENE_AREA_M2.
# ─────────────────────────────────────────────
SCENE_AREA_M2 = 25.0

# Density thresholds (persons/m^2) — Fruin-style heuristics
D_WARN = 4.0
D_CRIT = 6.0

# Velocity proxy thresholds (m/s) — derived from optical flow with FLOW_M_PER_PX
V_WARN = 0.8
V_CRIT = 1.5

# Calibration for turning optical flow (pixels) into meters
FLOW_M_PER_PX = 0.02

# How quickly density changes (persons/m^2/s)
DRHO_WARN = 0.5
DRHO_CRIT = 1.0

# Optical flow performance knobs
FLOW_EVERY_N_FRAMES = 3
FLOW_DOWNSCALE = 0.5


def _safe_inf_size(raw_w: int, raw_h: int):
    inf_w = int(raw_w * INF_SCALE)
    inf_h = int(raw_h * INF_SCALE)
    inf_w = (inf_w // 128) * 128
    inf_h = (inf_h // 128) * 128
    if inf_w < 128:
        inf_w = 128
    if inf_h < 128:
        inf_h = 128
    return inf_w, inf_h


def _points_to_heatmap(points, h: int, w: int):
    heat = np.zeros((h, w), dtype=np.float32)
    for p in points:
        x0, y0 = int(p[0]), int(p[1])
        if 0 <= x0 < w and 0 <= y0 < h:
            heat[y0, x0] = 1.0
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=10, sigmaY=10)
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat


def _heatmap_bgr(heat: np.ndarray):
    return cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)


def _scatter_bgr(points, h: int, w: int):
    scatter = np.full((h, w, 3), 255, dtype=np.uint8)
    for p in points:
        x0, y0 = int(p[0]), int(p[1])
        if 0 <= x0 < w and 0 <= y0 < h:
            cv2.circle(scatter, (x0, y0), 2, (0, 0, 0), -1)
    return scatter


def _has_cuda_cv2() -> bool:
    try:
        return hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


class VideoCamera(object):
    def __init__(self, fileName):
        self.read_lock = Lock()
        self.shared_frame = None

        self.source = fileName
        if fileName == '':
            # Try multiple camera indices
            for cam_idx in range(3):  # 0, 1, 2
                self.video = cv2.VideoCapture(cam_idx)
                if self.video.isOpened():
                    print(f"[INFO] Webcam opened on index {cam_idx}")
                    break
            else:
                print("[ERROR] Could not open any webcam (indices 0-2)")
                self.video = cv2.VideoCapture(0)  # Fallback, even if not opened
        else:
            self.video = cv2.VideoCapture(fileName)

        # Minimize internal buffer lag
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.prev_time = time.time()
        # Last successfully captured frame (for smoothing over transient errors)
        self.last_frame = None

        # Shared state between stream and inference thread
        self.latest_inf_frame = None   # Small frame sent to model
        self.inf_w            = 1      # Inference frame width  (for coord scaling)
        self.inf_h            = 1      # Inference frame height (for coord scaling)
        self.pred_points      = []     # Raw predicted points (in inference frame coords)
        self.pred_count       = 0      # Crowd count
        self.lock             = Lock()

        # Risk tracking state
        self.scene_area_m2    = SCENE_AREA_M2
        self.last_rho         = 0.0
        self.last_rho_ts      = time.time()
        self.prev_gray        = None
        self.last_v           = 0.0
        self.flow_frame_idx   = 0
        self.use_cv2_cuda     = _has_cuda_cv2()
        self.prev_gray_gpu    = None

        self.latest_metrics   = {}

        self._last_officer_alert_ts = 0.0
        self._prev_risk_level = None
        self._last_region_post_ts = 0.0
        
        # MySQL Logging Setup
        self.running = True
        self.db_config = {
            "host": "localhost",
            "user": "root",
            "password": "root",
            "database": "crowd_management"
        }
        self.log_queue = deque(maxlen=100)
        self.db_thread = Thread(target=self._mysql_logger_loop, daemon=True)
        self.db_thread.start()

        if self.use_cv2_cuda:
            try:
                self.cuda_stream = cv2.cuda_Stream()
            except Exception:
                self.cuda_stream = None
            try:
                self.of = cv2.cuda_FarnebackOpticalFlow.create(
                    5, 0.5, False, 15, 3, 5, 1.2, 0
                )
            except Exception:
                self.of = None
                self.use_cv2_cuda = False

        # Start background inference thread
        self.running = True
        self.thread  = Thread(target=self._inference_loop, daemon=True)
        self.thread.start()
        print("[INFO] Inference thread started")

    def __del__(self):
        self.running = False
        self.video.release()

    def _mysql_logger_loop(self):
        """Background thread to log metrics to MySQL without blocking inference."""
        while self.running:
            if not self.log_queue:
                time.sleep(0.5)
                continue
            
            try:
                metrics = self.log_queue.popleft()
                db = mysql.connector.connect(**self.db_config)
                cursor = db.cursor()
                
                # Log to metrics table
                cursor.execute("""
                    INSERT INTO metrics (count, density, velocity, risk_score, risk_level, fps)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    metrics["count"], metrics["rho"], metrics["v"], 
                    metrics["risk"], metrics["risk_level"], metrics["fps"]
                ))
                
                # Log to incidents table if risk is high
                if metrics["risk_level"] in ["WARNING", "CRITICAL"]:
                    cursor.execute("""
                        INSERT INTO incidents (risk_level, risk_score, count, details)
                        VALUES (%s, %s, %s, %s)
                    """, (
                        metrics["risk_level"], metrics["risk"], 
                        metrics["count"], f"drho_dt: {metrics['drho_dt']:.2f}"
                    ))
                
                db.commit()
                cursor.close()
                db.close()
            except Exception as e:
                print(f"[MySQL Log Error] {e}")
                time.sleep(1)

    # ─────────────────────────────────────────
    #  Background Inference Thread
    # ─────────────────────────────────────────
    def _inference_loop(self):
        while self.running:
            with self.lock:
                frame = None
                if self.latest_inf_frame is not None:
                    frame = self.latest_inf_frame.copy()
                    self.latest_inf_frame = None   # IMPORTANT: clear after taking

            if frame is None:
                time.sleep(0.01)
                continue

            try:
                with torch.no_grad():
                    img_tensor = transform(frame)
                    samples    = img_tensor.unsqueeze(0).to(device)

                    outputs        = model(samples)
                    outputs_scores = torch.nn.functional.softmax(
                        outputs['pred_logits'], -1
                    )[:, :, 1][0]
                    outputs_points = outputs['pred_points'][0]

                    mask        = outputs_scores > THRESHOLD
                    points      = outputs_points[mask].detach().cpu().numpy().tolist()
                    predict_cnt = int(mask.sum())

                with self.lock:
                    self.pred_points = points
                    self.pred_count  = predict_cnt

            except Exception as e:
                print(f"[Inference Error] {e}")

            time.sleep(0.001)

    # ─────────────────────────────────────────
    #  Default Error Frame
    # ─────────────────────────────────────────
    def _default_frame(self, message="Webcam Error - Check Permissions/Index"):
        frame = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        cv2.putText(frame, message, (30, OUT_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    # ─────────────────────────────────────────
    #  Main Frame Getter
    #  panel: 'live', 'heatmap', 'detection', 'scatter', or 'grid'
    # ─────────────────────────────────────────
    def get_frame(self, panel='grid'):
    
        # ── Read frame safely ─────────────────────
        with self.read_lock:
            ret, frame = self.video.read()
    
            if not ret or frame is None:
            
                # restart video if file source
                if isinstance(self.source, str) and self.source != "":
                    self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.video.read()
    
                # fallback to last frame
                if not ret or frame is None:
                    if self.last_frame is not None:
                        frame = self.last_frame.copy()
                    else:
                        return self._default_frame()
    
            self.last_frame = frame.copy()
    
    
        # ── FPS calculation ─────────────────────
        current_time = time.time()
        dt = current_time - self.prev_time
        fps = 1.0 / dt if dt > 0 else 0.0
        fps = min(fps, 30.0)
        self.prev_time = current_time
    
    
        # ── Display frame ─────────────────────
        display_frame = cv2.resize(frame, (OUT_W, OUT_H))
    
    
        # ── Inference frame ───────────────────
        raw_h, raw_w = frame.shape[:2]
        inf_w, inf_h = _safe_inf_size(raw_w, raw_h)
        inf_frame = cv2.resize(frame, (inf_w, inf_h))
    
    
        # ── Send frame to inference thread safely ─────────
        with self.lock:
        
            if self.latest_inf_frame is None:
                self.latest_inf_frame = inf_frame.copy()
    
            self.inf_w = inf_w
            self.inf_h = inf_h
    
            points = list(self.pred_points)
            predict_cnt = self.pred_count


        # ── Stampede risk metrics (approximate) ─────────────────────────
        # Density (persons/m^2)
        rho = (float(predict_cnt) / float(self.scene_area_m2)) if self.scene_area_m2 > 0 else 0.0

        # dρ/dt
        now_ts = time.time()
        dt_rho = max(1e-3, now_ts - self.last_rho_ts)
        drho_dt = (rho - self.last_rho) / dt_rho
        self.last_rho = rho
        self.last_rho_ts = now_ts

        # Optical flow velocity proxy v (m/s)
        v = float(self.last_v)
        self.flow_frame_idx += 1
        if (self.flow_frame_idx % FLOW_EVERY_N_FRAMES) == 0:
            downscale = float(FLOW_DOWNSCALE) if (0.0 < FLOW_DOWNSCALE < 1.0) else 1.0

            if self.use_cv2_cuda and self.of is not None:
                # GPU optical flow (requires OpenCV built with CUDA)
                try:
                    g = cv2.cuda_GpuMat()
                    g.upload(inf_frame, stream=self.cuda_stream) if self.cuda_stream is not None else g.upload(inf_frame)
                    g_gray = cv2.cuda.cvtColor(g, cv2.COLOR_BGR2GRAY)
                    if downscale != 1.0:
                        new_size = (max(8, int(inf_w * downscale)), max(8, int(inf_h * downscale)))
                        g_gray = cv2.cuda.resize(g_gray, new_size)

                    if self.prev_gray_gpu is not None:
                        g_flow = self.of.calc(self.prev_gray_gpu, g_gray, None)
                        flow = g_flow.download()
                        mag, _ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        mag_px = float(np.percentile(mag, 75))
                        mag_px_full = mag_px / max(1e-6, downscale)
                        v = mag_px_full * FLOW_M_PER_PX * fps

                    self.prev_gray_gpu = g_gray
                    self.last_v = float(v)
                except Exception:
                    # CUDA path failed → fall back to CPU path
                    self.use_cv2_cuda = False
                    self.prev_gray_gpu = None

            if not self.use_cv2_cuda:
                gray = cv2.cvtColor(inf_frame, cv2.COLOR_BGR2GRAY)
                if downscale != 1.0:
                    gray = cv2.resize(gray, (0, 0), fx=downscale, fy=downscale)

                if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
                    flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None,
                                                        0.5, 2, 15, 2, 5, 1.2, 0)
                    mag, _ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    mag_px = float(np.percentile(mag, 75))
                    mag_px_full = mag_px / max(1e-6, downscale)
                    v = mag_px_full * FLOW_M_PER_PX * fps

                self.prev_gray = gray
                self.last_v = float(v)

        # Flow rate Q and pressure index P
        Q = rho * v
        P = rho * (v ** 2)

        # Risk score composite (0..1) based on normalized sub-scores
        def _norm(x, warn, crit):
            if crit <= warn:
                return 0.0
            return float(np.clip((x - warn) / (crit - warn), 0.0, 1.0))

        d_score = _norm(rho, D_WARN, D_CRIT)
        v_score = _norm(v, V_WARN, V_CRIT)
        dr_score = _norm(abs(drho_dt), DRHO_WARN, DRHO_CRIT)
        risk = float(np.clip(0.55 * d_score + 0.25 * v_score + 0.20 * dr_score, 0.0, 1.0))

        if risk >= 0.7 or rho >= D_CRIT:
            risk_level = "CRITICAL"
            risk_color = (0, 0, 255)
        elif risk >= 0.4 or rho >= D_WARN:
            risk_level = "WARNING"
            risk_color = (0, 165, 255)
        else:
            risk_level = "LOW"
            risk_color = (0, 255, 0)

        self.latest_metrics = {
            "ts": time.time(),
            "count": int(predict_cnt),
            "rho": float(rho),
            "drho_dt": float(drho_dt),
            "v": float(v),
            "risk": float(risk),
            "risk_level": str(risk_level),
            "fps": float(fps),
            "device": str(device.type),
        }

        try:
            lat_to_send = float(CAMERA_LATITUDE)
            lon_to_send = float(CAMERA_LONGITUDE)
            now_alert_ts = float(self.latest_metrics.get('ts', time.time()))

            entered_critical = (risk_level == "CRITICAL") and (self._prev_risk_level != "CRITICAL")
            cooldown_ok = (now_alert_ts - float(self._last_officer_alert_ts)) >= float(OFFICER_ALERT_COOLDOWN_SEC)
            if entered_critical and cooldown_ok:
                payload = {
                    "location": CAMERA_ID,
                    "risk_level": str(risk_level),
                    "crowd_count": int(predict_cnt),
                    "crowd_density": float(rho),
                    "latitude": float(lat_to_send),
                    "longitude": float(lon_to_send),
                    "timestamp": float(now_alert_ts),
                }
                requests.post(OFFICER_ALERT_URL, json=payload, timeout=1.5)
                self._last_officer_alert_ts = float(now_alert_ts)
        except Exception:
            pass
        finally:
            self._prev_risk_level = str(risk_level)

        try:
            now_region_ts = float(self.latest_metrics.get('ts', time.time()))
            interval_ok = (now_region_ts - float(self._last_region_post_ts)) >= float(OFFICER_REGION_UPDATE_INTERVAL_SEC)
            if interval_ok:
                region_payload = {
                    "camera_id": CAMERA_ID,
                    "latitude": float(CAMERA_LATITUDE),
                    "longitude": float(CAMERA_LONGITUDE),
                    "crowd_density": float(rho),
                    "risk_level": str(risk_level),
                    "timestamp": float(now_region_ts),
                }
                requests.post(OFFICER_REGION_URL, json=region_payload, timeout=1.0)
                self._last_region_post_ts = float(now_region_ts)
        except Exception:
            pass

        # Queue for MySQL background logging
        if self.flow_frame_idx % (FLOW_EVERY_N_FRAMES * 2) == 0:
            self.log_queue.append(self.latest_metrics)
        # Build 4 panels (all OUT_W/2 x OUT_H/2)
        panel_w = OUT_W // 2
        panel_h = OUT_H // 2

        # Points are in inference-frame coords. Build heat/scatter in inference coords first.
        heat = _points_to_heatmap(points, inf_h, inf_w)
        heat_bgr = _heatmap_bgr(heat)
        scatter_bgr = _scatter_bgr(points, inf_h, inf_w)

        # Original with dots (in inference coords, then upscaled)
        inf_with_dots = inf_frame.copy()
        for p in points:
            cv2.circle(inf_with_dots, (int(p[0]), int(p[1])), 1, (0, 0, 255), -1)
            cv2.circle(inf_with_dots, (int(p[0]), int(p[1])), 2, (255, 255, 255), 1)

        # Resize base panels
        p1 = cv2.resize(display_frame, (panel_w, panel_h))
        p2 = cv2.resize(heat_bgr, (panel_w, panel_h))
        p3 = cv2.resize(inf_with_dots, (panel_w, panel_h))
        p4 = cv2.resize(scatter_bgr, (panel_w, panel_h))

        # Overlay info on p3 (detection / live overlay)
        overlay = p3.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, 175), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, p3, 0.6, 0, p3)
        cv2.putText(p3, f"Count: {predict_cnt}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(p3, f"FPS: {fps:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(p3, f"Device: {device.type.upper()}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(p3, f"rho: {rho:.2f} p/m2", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(p3, f"v: {v:.2f} m/s", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Risk label banner
        cv2.rectangle(p3, (0, panel_h - 35), (panel_w, panel_h), (0, 0, 0), -1)
        cv2.putText(p3, f"Risk: {risk:.2f} {risk_level}", (20, panel_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, risk_color, 2)

        # Depending on requested panel, return specific content
        if panel == 'live':
            # Raw display frame with no overlays (scaled to output size)
            out = cv2.resize(display_frame, (OUT_W, OUT_H))
        elif panel == 'heatmap':
            out = cv2.resize(heat_bgr, (OUT_W, OUT_H))
        elif panel == 'detection':
            out = cv2.resize(p3, (OUT_W, OUT_H))
        elif panel == 'scatter':
            out = cv2.resize(scatter_bgr, (OUT_W, OUT_H))
        else:
            # 2x2 grid fallback
            top = np.hstack([p1, p2])
            bottom = np.hstack([p3, p4])
            out = np.vstack([top, bottom])

        _, jpeg = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self.read_lock:
            self.shared_frame = None
        return jpeg.tobytes()   