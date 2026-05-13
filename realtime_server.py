"""
realtime_server.py
==================
Self-contained real-time pose comparison server.

Two phases:
  1. OFFLINE  POST /prepare          — process instructor video once, save reference angles
  2. ONLINE   WS   /ws/compare/{id}  — stream patient frames, get feedback back instantly

Install:
    pip install fastapi uvicorn python-multipart ultralytics opencv-python pillow scipy websockets

Run:
    uvicorn realtime_server:app --host 0.0.0.0 --port 8000

Then open  http://localhost:8000  in a browser.

Demo with a recorded video:
    Open the browser client, click "Load video file" and select patient.mp4.
    It plays back frame by frame through the WebSocket exactly like a live stream.
"""

import base64
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# Joint definitions
# ─────────────────────────────────────────────────────────────────────────────
JOINT_ANGLE_DEFS = [
    ("right_shoulder", 5,  6,  8),
    ("right_arm",      6,  8,  10),
    ("left_shoulder",  7,  5,  6),
    ("left_arm",       9,  7,  5),
    ("right_hip",      11, 12, 14),
    ("right_leg",      12, 14, 16),
    ("left_hip",       13, 11, 12),
    ("left_leg",       11, 13, 15),
]
JOINT_NAMES = [d[0] for d in JOINT_ANGLE_DEFS]
N_JOINTS    = len(JOINT_ANGLE_DEFS)

SKELETON = [
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]
COCO_LR_PAIRS  = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
ANGLE_LR_PAIRS = [("left_shoulder","right_shoulder"),("left_arm","right_arm"),
                  ("left_hip","right_hip"),("left_leg","right_leg")]

# Drawing canvas
CW, CH     = 1280, 540
FRAME_W    = 220
FRAME_H    = 440
FRAME_Y    = 70
X_REF_ANG  = 6
X_REF_IMG  = 160
X_DIFF     = 400
X_TRGT_IMG = 760
X_TRGT_ANG = 1000

SMOOTH_SIGMA   = 5
DEFAULT_OFFSET = 25
DEFAULT_MODEL  = "yolov8m-pose.pt"

# In-memory reference store   ref_id -> { angles, meta }
REFS: Dict[str, dict] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────
def _angle(kpts, a, b, c) -> float:
    pa, pb, pc = kpts[a,:2], kpts[b,:2], kpts[c,:2]
    ang = math.degrees(
        math.atan2(pc[1]-pb[1], pc[0]-pb[0]) -
        math.atan2(pa[1]-pb[1], pa[0]-pb[0])
    )
    ang = ang + 360 if ang < 0 else ang
    ang = ang - 180 if ang > 270 else ang
    return float(ang)

def kpts_to_angles(kpts) -> np.ndarray:
    """(17,3) keypoints → (N_JOINTS,) angle array."""
    return np.array([_angle(kpts, a, b, c) for _, a, b, c in JOINT_ANGLE_DEFS],
                    dtype=np.float32)

def safe_int_angles(arr: np.ndarray) -> Dict[str, int]:
    return {jn: int(v) if not np.isnan(v) else 0
            for k, jn in enumerate(JOINT_NAMES) for v in [arr[k]]}


# ─────────────────────────────────────────────────────────────────────────────
# Mirror helpers
# ─────────────────────────────────────────────────────────────────────────────
def mirror_pil(img):    return img.transpose(Image.FLIP_LEFT_RIGHT)
def mirror_kpts(k, w):
    m = k.copy();  m[:,0] = w - m[:,0]
    for l,r in COCO_LR_PAIRS: m[[l,r]] = m[[r,l]]
    return m
def mirror_ang(d):
    out = dict(d)
    for l,r in ANGLE_LR_PAIRS:
        if l in out and r in out: out[l], out[r] = out[r], out[l]
    return out
def total_diff(a, b): return sum(abs(a[k]-b.get(k,0)) for k in a)


# ─────────────────────────────────────────────────────────────────────────────
# YOLO inference on a single PIL/numpy frame
# ─────────────────────────────────────────────────────────────────────────────
# Cache for YOLO angle extraction
CACHE_DIR = Path("yolo_cache")
CACHE_DIR.mkdir(exist_ok=True)

def _video_hash(video_path: str) -> str:
    """
    Content-based hash using first 1MB + last 1MB + file size.
    Reliable across uploads because it reads actual bytes, not
    filename or mtime (both change every time a file is uploaded
    to a temp directory).
    """
    size  = os.path.getsize(video_path)
    chunk = 1024 * 1024  # 1 MB
    h     = hashlib.md5()
    h.update(str(size).encode())
    with open(video_path, "rb") as f:
        h.update(f.read(chunk))          # first 1 MB
        if size > chunk:
            f.seek(max(0, size - chunk))
            h.update(f.read(chunk))      # last 1 MB
    return h.hexdigest()

def _save_cache(h: str, angles: np.ndarray, meta: dict):
    np.savez_compressed(CACHE_DIR / f"{h}.npz", angles=angles)
    with open(CACHE_DIR / f"{h}.json", "w") as f: json.dump(meta, f)

def _load_cache(h: str):
    cp = CACHE_DIR / f"{h}.npz"
    mp = CACHE_DIR / f"{h}.json"
    if not cp.exists() or not mp.exists(): return None
    return np.load(cp)["angles"], json.load(open(mp))


_model_cache: Dict[str, YOLO] = {}

def get_model(name: str) -> YOLO:
    if name not in _model_cache:
        _model_cache[name] = YOLO(name)
    return _model_cache[name]

def infer_frame(bgr: np.ndarray, model: YOLO, conf: float = 0.5
                ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
                           Optional[np.ndarray]]:
    """
    Run YOLO on a BGR frame.
    Returns (kpts_17x3, box_4, angles_N) or (None, None, None) if no detection.
    """
    results = model(bgr, verbose=False, conf=conf)
    res = results[0]
    if res.keypoints is None or len(res.keypoints.data) == 0:
        return None, None, None
    best = int(np.argmax(res.boxes.conf.cpu().numpy())) if res.boxes else 0
    kp   = res.keypoints.data[best].cpu().numpy()           # (17,3)
    box  = res.boxes.xyxy[best].cpu().numpy() if res.boxes else None
    ang  = kpts_to_angles(kp)
    return kp, box, ang


# ─────────────────────────────────────────────────────────────────────────────
# Offline phase — extract reference angles from instructor video
# ─────────────────────────────────────────────────────────────────────────────
def extract_reference(video_path: str, model: YOLO,
                      conf: float = 0.5) -> Tuple[np.ndarray, List, List, dict]:
    """
    Extract angle sequence + frames + kpts from instructor video.
    Returns (angles (N,K), pil_frames, kpts_list, meta).
    """
    video_hash = _video_hash(video_path)
    cached     = _load_cache(video_hash)

    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pil_frames, bgr_frames = [], []
    while True:
        ret, bgr = cap.read()
        if not ret: break
        pil_frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        bgr_frames.append(bgr)
    cap.release()

    if cached is not None:
        angles, meta = cached
        kpts_list  = [None] * len(pil_frames)
        boxes_list = [None] * len(pil_frames)
        print(f"[cache] HIT  {Path(video_path).name}")
    else:
        angles_list, kpts_list, boxes_list = [], [], []
        for bgr in bgr_frames:
            kp, box, ang = infer_frame(bgr, model, conf)
            angles_list.append(ang if ang is not None else np.full(N_JOINTS, np.nan, np.float32))
            kpts_list.append(kp)
            boxes_list.append(box)
        angles = np.array(angles_list, dtype=np.float32)
        det    = 100.0 * sum(1 for k in kpts_list if k is not None) / max(len(kpts_list), 1)
        meta   = {"fps": fps, "total": len(pil_frames), "width": width,
                  "height": height, "detection_pct": round(det, 1)}
        _save_cache(video_hash, angles, meta)
        print(f"[cache] MISS {Path(video_path).name} — saved to cache")

    # Smooth + z-score (for nearest-neighbour search)
    smoothed = angles.copy()
    for k in range(N_JOINTS):
        col = smoothed[:,k]; nm = np.isnan(col)
        if nm.all(): continue
        col[nm] = np.nanmedian(col)
        smoothed[:,k] = gaussian_filter1d(col.astype(np.float64), SMOOTH_SIGMA).astype(np.float32)
    mean = np.nanmean(smoothed, 0);  std = np.nanstd(smoothed, 0)
    std  = np.where(std < 1e-3, 1e-3, std)
    z    = ((smoothed - mean) / std).astype(np.float32)

    return angles, smoothed, z, pil_frames, kpts_list, boxes_list, mean, std, meta


# ─────────────────────────────────────────────────────────────────────────────
# Online phase — nearest-neighbour frame matching
# ─────────────────────────────────────────────────────────────────────────────
def find_best_ref_frame(patient_ang: np.ndarray,
                        ref_angles_clean: np.ndarray) -> int:
    """
    Find the instructor frame with the lowest total absolute angle difference
    from the patient's current angles.

    Simple L1 distance across all joints, compared against every instructor
    frame at once using numpy broadcasting. No z-scores, no windows, no EMA.
    Just: which instructor pose is numerically closest to the patient right now?

    ref_angles_clean: (N, K) array with NaN replaced by column median,
                      pre-computed once at prepare time.
    patient_ang:      (K,) array of current patient joint angles.
    """
    pat_clean = np.where(np.isnan(patient_ang), 0.0, patient_ang)
    diffs     = np.sum(np.abs(ref_angles_clean - pat_clean[np.newaxis, :]), axis=1)
    return int(np.argmin(diffs))


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────
def _try_font(size):
    for n in ["Arial.ttf","arial.ttf","DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try: return ImageFont.truetype(n, size=size)
        except: pass
    return ImageFont.load_default()

def _draw_skeleton(pil_img, kpts, color="red", conf_thr=0.3):
    img = pil_img.copy(); draw = ImageDraw.Draw(img)
    if kpts is None: return img
    for i,j in SKELETON:
        ci = kpts[i,2] if kpts.shape[1]>2 else 1.0
        cj = kpts[j,2] if kpts.shape[1]>2 else 1.0
        if ci >= conf_thr and cj >= conf_thr:
            draw.line([(float(kpts[i,0]),float(kpts[i,1])),
                       (float(kpts[j,0]),float(kpts[j,1]))], fill=color, width=3)
    for pt in range(5):
        c = kpts[pt,2] if kpts.shape[1]>2 else 1.0
        if c >= conf_thr:
            px,py = float(kpts[pt,0]), float(kpts[pt,1])
            draw.ellipse([(px-4,py-4),(px+4,py+4)], fill=color)
    return img

def _crop_resize(img, box):
    if box is not None:
        x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
        if x2>x1 and y2>y1:
            return img.crop((x1,y1,x2,y2)).resize((FRAME_W, FRAME_H))
    return img.resize((FRAME_W, FRAME_H))

def draw_realtime_frame(ref_pil, pat_pil, ref_kpts, pat_kpts,
                        ref_box, pat_box,
                        ref_ang: Dict[str,int], pat_ang: Dict[str,int],
                        ref_idx: int, pat_frame_num: int,
                        was_mirrored: bool, offset: int,
                        latency_ms: float) -> Image.Image:
    f14 = _try_font(14); f16 = _try_font(16); f20 = _try_font(20)

    ref_thumb = _crop_resize(_draw_skeleton(ref_pil, ref_kpts, "red"),  ref_box)
    pat_thumb = _crop_resize(_draw_skeleton(pat_pil, pat_kpts, "blue"), pat_box)

    canvas = Image.new("RGB", (CW, CH), (18,18,18))
    d = ImageDraw.Draw(canvas)

    canvas.paste(ref_thumb, (X_REF_IMG,  FRAME_Y))
    canvas.paste(pat_thumb, (X_TRGT_IMG, FRAME_Y))

    # Headers
    d.text((X_REF_IMG  + FRAME_W//2 - 50, 8),  "INSTRUCTOR",  font=f20, fill=(100,220,80))
    d.text((X_TRGT_IMG + FRAME_W//2 - 30, 8),  "PATIENT",     font=f20, fill=(40,185,230))
    d.text((X_REF_IMG  + FRAME_W//2 - 40, 46), f"ref #{ref_idx:04d}", font=f14, fill=(100,220,80))
    d.text((X_TRGT_IMG + FRAME_W//2 - 35, 46), f"frame #{pat_frame_num:04d}", font=f14, fill=(40,185,230))

    # Latency badge
    lat_color = (80,200,80) if latency_ms < 100 else (220,180,40) if latency_ms < 200 else (220,80,60)
    d.text((CW-140, 8), f"latency: {latency_ms:.0f}ms", font=f14, fill=lat_color)
    if was_mirrored:
        d.text((CW-140, 26), "[mirrored]", font=f14, fill=(200,160,40))

    # Dividers
    d.line([(X_DIFF, 0),(X_DIFF, CH)], fill=(60,60,60), width=1)
    d.line([(X_TRGT_IMG-20, 0),(X_TRGT_IMG-20, CH)], fill=(60,60,60), width=1)

    # Ref angles (left)
    d.text((X_REF_ANG, FRAME_Y), "Ref:", font=f14, fill=(150,150,150))
    for row,(jn,val) in enumerate(ref_ang.items()):
        d.text((X_REF_ANG, FRAME_Y+20+row*20), f"{jn}: {val}", font=f14, fill=(200,200,200))

    # Patient angles (right)
    d.text((X_TRGT_ANG, FRAME_Y), "Patient:", font=f14, fill=(150,150,150))
    for row,(jn,val) in enumerate(pat_ang.items()):
        d.text((X_TRGT_ANG, FRAME_Y+20+row*20), f"{jn}: {val}", font=f14, fill=(200,200,200))

    # Diff block (centre)
    cx, cy = X_DIFF+8, 8
    all_ok = True
    for jn in JOINT_NAMES:
        diff = ref_ang[jn] - pat_ang.get(jn, 0)
        ok   = abs(diff) < offset
        if not ok: all_ok = False
        tag  = "OK " if ok else "BAD"
        col  = (80,200,80) if ok else (220,80,60)
        sign = "+" if diff >= 0 else ""
        d.text((cx, cy), f"{tag}  {jn:<17s}{sign}{diff:d}",
               font=f16, fill=col)
        cy += 24

    cy += 6
    d.line([(cx,cy),(X_TRGT_IMG-25,cy)], fill=(60,60,60), width=1); cy+=8
    d.text((cx, cy), "ALL OK" if all_ok else "HAS ERRORS",
           font=f20, fill=(80,200,80) if all_ok else (220,80,60))

    return canvas


def pil_to_jpeg_b64(img: Image.Image, quality: int = 80) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Smart Mirror — Real-time Pose Comparison", version="2.0.0")


@app.get("/health")
def health(): return {"status": "ok"}


# Path to the pre-configured instructor video (placed in the project root)
_INSTRUCTOR_VIDEO = Path(__file__).parent / "instructor.mp4"

@app.get("/instructor-video")
def serve_instructor_video():
    """Serve the pre-configured instructor video so the browser can play it."""
    if not _INSTRUCTOR_VIDEO.exists():
        raise HTTPException(status_code=404, detail="instructor.mp4 not found")
    return FileResponse(str(_INSTRUCTOR_VIDEO), media_type="video/mp4")


# ─────────────────────────────────────────────────────────────────────────────
# OFFLINE endpoint — prepare reference from instructor video
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/prepare", summary="Process instructor video (offline phase)")
async def prepare(
    ref:        UploadFile = File(..., description="Instructor video"),
    model_name: str   = Form(DEFAULT_MODEL),
    conf:       float = Form(0.5),
):
    """
    Upload instructor video.
    Returns a ref_id to use in the WebSocket URL.
    Processing takes ~1-2 min per minute of video.
    """
    tmp_dir  = tempfile.mkdtemp(prefix="sm_ref_")
    ref_path = os.path.join(tmp_dir, "ref" + Path(ref.filename).suffix)
    with open(ref_path, "wb") as f: f.write(await ref.read())

    try:
        model = get_model(model_name)
        angles, smoothed, z, frames, kpts, boxes, mean, std, meta = \
            extract_reference(ref_path, model, conf)
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))

    ref_id = str(uuid.uuid4())
    # Pre-compute clean angle matrix for fast argmin matching
    # NaN replaced with column median so missing joints don't break the search
    angles_clean = angles.copy()
    for k in range(angles_clean.shape[1]):
        col = angles_clean[:, k]
        col[np.isnan(col)] = float(np.nanmedian(col)) if not np.all(np.isnan(col)) else 0.0

    REFS[ref_id] = {
        "angles":        angles,        # raw degrees (N, K)
        "angles_clean":  angles_clean,  # NaN-free (N, K) — used for argmin search
        "frames":        frames,        # list of PIL Images
        "kpts":          kpts,          # list of (17,3) or None
        "boxes":         boxes,         # list of box arrays or None
        "meta":          meta,
        "model":         model_name,
        "conf":          conf,
    }

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return {
        "ref_id":    ref_id,
        "ws_url":    f"/ws/compare/{ref_id}",
        "meta":      meta,
        "ready":     True,
        "message":   f"Reference ready. Connect to ws://HOST:PORT/ws/compare/{ref_id}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# ONLINE endpoint — WebSocket real-time comparison
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/compare", summary="Upload both videos, get JSON summary (blocking)")
async def compare_direct(
    ref:        UploadFile = File(..., description="Instructor video"),
    trgt:       UploadFile = File(..., description="Patient video"),
    model_name: str   = Form(DEFAULT_MODEL),
    offset:     int   = Form(DEFAULT_OFFSET),
    conf:       float = Form(0.5),
):
    """
    Simple one-shot endpoint. Upload instructor + patient videos, get the
    full comparison JSON back. Blocks until processing is complete.

    Usage:
        curl -X POST http://localhost:8000/compare \
          -F "ref=@instructor.mp4" \
          -F "trgt=@patient.mp4" \
          -F "model_name=yolov8m-pose.pt" \
          -F "offset=25"
    """
    tmp   = tempfile.mkdtemp(prefix="sm_cmp_")
    r_path = os.path.join(tmp, "ref"  + Path(ref.filename).suffix)
    t_path = os.path.join(tmp, "trgt" + Path(trgt.filename).suffix)
    try:
        with open(r_path, "wb") as f: f.write(await ref.read())
        with open(t_path, "wb") as f: f.write(await trgt.read())

        model = get_model(model_name)

        # Extract both videos
        ref_angles, _, _, ref_frames, ref_kpts, ref_boxes, _, _, meta_r =             extract_reference(r_path, model, conf)
        pat_angles, _, _, _,          _,        _,         _, _, meta_t =             extract_reference(t_path, model, conf)

        # Pre-compute clean ref angles for argmin
        ref_clean = ref_angles.copy()
        for k in range(ref_clean.shape[1]):
            col = ref_clean[:, k]
            col[np.isnan(col)] = float(np.nanmedian(col))                 if not np.all(np.isnan(col)) else 0.0

        # Compare every patient frame against closest instructor frame
        N           = len(pat_angles)
        session_ok  = 0
        records     = []

        for fp in range(N):
            pat_ang_arr  = pat_angles[fp]
            pat_ang_dict = safe_int_angles(pat_ang_arr)
            pat_ang_m    = mirror_ang(pat_ang_dict)
            best_idx     = find_best_ref_frame(pat_ang_arr, ref_clean)
            ref_ang_dict = safe_int_angles(ref_angles[best_idx])
            use_mirror   = total_diff(ref_ang_dict, pat_ang_m) <                            total_diff(ref_ang_dict, pat_ang_dict)
            chosen_ang   = pat_ang_m if use_mirror else pat_ang_dict

            joints = {}
            all_ok = True
            for jn in JOINT_NAMES:
                diff = ref_ang_dict[jn] - chosen_ang.get(jn, 0)
                ok   = abs(diff) < offset
                if not ok: all_ok = False
                joints[jn] = {"ref": ref_ang_dict[jn], "trgt": chosen_ang.get(jn, 0),
                               "diff": diff, "status": "OK" if ok else "BAD"}
            if all_ok: session_ok += 1

            if fp % 5 == 0:
                records.append({
                    "patient_frame": fp, "instructor_frame": best_idx,
                    "mirrored": bool(use_mirror), "all_ok": bool(all_ok),
                    "joints": joints,
                })

        # Build summary
        jstats = {}
        for jn in JOINT_NAMES:
            diffs = [r["joints"][jn]["diff"] for r in records]
            bad_n = sum(1 for r in records if r["joints"][jn]["status"] == "BAD")
            jstats[jn] = {
                "bad_frames":   bad_n,
                "bad_pct":      round(100 * bad_n / max(len(records), 1), 1),
                "mean_diff":    round(float(np.mean(diffs)), 1) if diffs else 0,
                "max_abs_diff": int(max(abs(d) for d in diffs)) if diffs else 0,
            }

        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname  = f"compare_{ts}.json"
        result = {
            "meta": {
                "total_patient_frames":    N,
                "total_instructor_frames": len(ref_frames),
                "ok_frames":               session_ok,
                "ok_pct":                  round(100 * session_ok / max(N, 1), 1),
                "angle_threshold":         offset,
                "ref_detection_pct":       meta_r["detection_pct"],
                "trgt_detection_pct":      meta_t["detection_pct"],
                "saved_to":                fname,
            },
            "joint_stats": jstats,
            "frames":      records,
        }
        with open(fname, "w") as f: json.dump(result, f, indent=2)
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.websocket("/ws/compare/{ref_id}")
async def ws_compare(websocket: WebSocket, ref_id: str,
                     offset: int = DEFAULT_OFFSET):
    if ref_id not in REFS:
        await websocket.close(code=4004, reason="ref_id not found — call /prepare first")
        return

    await websocket.accept()
    ref            = REFS[ref_id]
    model          = get_model(ref["model"])
    frame_n        = 0
    session_frames = []
    session_ok     = 0

    try:
        while True:
            t_recv = time.time()
            data   = await websocket.receive_text()
            msg    = json.loads(data)

            # ── Done signal: instructor video finished ────────────────────────
            if msg.get("done"):
                jstats = {}
                for jn in JOINT_NAMES:
                    diffs = [r["joints"][jn]["diff"] for r in session_frames if jn in r.get("joints",{})]
                    bad_n = sum(1 for r in session_frames if r.get("joints",{}).get(jn,{}).get("status")=="BAD")
                    jstats[jn] = {
                        "bad_frames": bad_n,
                        "bad_pct":    round(100*bad_n/max(len(session_frames),1),1),
                        "mean_diff":  round(float(np.mean(diffs)),1) if diffs else 0,
                        "max_abs_diff": int(max(abs(d) for d in diffs)) if diffs else 0,
                    }
                summary = {
                    "meta": {"total_frames": frame_n, "ok_frames": session_ok,
                             "ok_pct": round(100*session_ok/max(frame_n,1),1),
                             "angle_threshold": offset},
                    "joint_stats": jstats,
                    "frames": session_frames,
                }
                ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"session_{ts}_{ref_id[:8]}.json"
                with open(save_path, "w") as f: json.dump(summary, f, indent=2)
                await websocket.send_text(json.dumps(
                    {"status": "summary", "summary": summary, "saved_to": save_path}))
                break

            frame_b64 = msg.get("frame", "")
            offset    = int(msg.get("offset", offset))

            # Decode JPEG → BGR numpy
            img_bytes = base64.b64decode(frame_b64)
            np_arr    = np.frombuffer(img_bytes, np.uint8)
            bgr       = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr is None:
                await websocket.send_text(json.dumps({"error": "bad frame"}))
                continue

            pat_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            h, w    = bgr.shape[:2]

            # ── YOLO inference on patient frame ──────────────────────────────
            pat_kp, pat_box, pat_ang_arr = infer_frame(bgr, model, ref["conf"])

            if pat_ang_arr is None:
                # No person detected — send back original frame with message
                await websocket.send_text(json.dumps({
                    "status": "no_detection",
                    "frame_num": frame_n,
                }))
                frame_n += 1
                continue

            pat_ang_dict  = safe_int_angles(pat_ang_arr)
            pat_ang_m     = mirror_ang(pat_ang_dict)
            pat_kp_m      = mirror_kpts(pat_kp, w) if pat_kp is not None else None

            # ── Find instructor frame with lowest total angle diff ──────
            best_idx = find_best_ref_frame(pat_ang_arr, ref["angles_clean"])

            ref_ang_arr  = ref["angles"][best_idx]
            ref_ang_dict = safe_int_angles(ref_ang_arr)

            # Mirror selection: pick whichever orientation is closer
            use_mirror = total_diff(ref_ang_dict, pat_ang_m) < \
                         total_diff(ref_ang_dict, pat_ang_dict)
            if use_mirror:
                chosen_pil = mirror_pil(pat_pil)
                chosen_kp  = pat_kp_m
                chosen_ang = pat_ang_m
            else:
                chosen_pil = pat_pil
                chosen_kp  = pat_kp
                chosen_ang = pat_ang_dict

            latency_ms = (time.time() - t_recv) * 1000

            # ── Render full comparison image (server-side, same as draw_comparison) ──
            comp_img = draw_realtime_frame(
                ref_pil       = ref["frames"][best_idx],
                pat_pil       = chosen_pil,
                ref_kpts      = ref["kpts"][best_idx],
                pat_kpts      = chosen_kp,
                ref_box       = ref["boxes"][best_idx],
                pat_box       = pat_box,
                ref_ang       = ref_ang_dict,
                pat_ang       = chosen_ang,
                ref_idx       = best_idx,
                pat_frame_num = frame_n,
                was_mirrored  = use_mirror,
                offset        = offset,
                latency_ms    = latency_ms,
            )

            # Per-joint status (for the live stats panel in the browser)
            joints = {}
            all_ok = True
            for jn in JOINT_NAMES:
                diff = ref_ang_dict[jn] - chosen_ang.get(jn, 0)
                ok   = abs(diff) < offset
                if not ok: all_ok = False
                joints[jn] = {"diff": diff, "status": "OK" if ok else "BAD"}

            if all_ok: session_ok += 1
            if frame_n % 5 == 0:
                session_frames.append({
                    "patient_frame": frame_n, "instructor_frame": best_idx,
                    "mirrored": bool(use_mirror), "all_ok": bool(all_ok),
                    "joints": {jn: {"diff": joints[jn]["diff"], "status": joints[jn]["status"]}
                               for jn in JOINT_NAMES},
                })

            await websocket.send_text(json.dumps({
                "status":     "ok",
                "frame_num":  frame_n,
                "ref_frame":  best_idx,
                "mirrored":   bool(use_mirror),
                "all_ok":     bool(all_ok),
                "latency_ms": round(latency_ms, 1),
                "joints":     joints,
                "frame_b64":  pil_to_jpeg_b64(comp_img, quality=78),
            }))

            frame_n += 1

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Browser client — served at GET /
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health/webcam")
def webcam_info():
    """List available cameras via OpenCV (server-side check)."""
    found = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # CAP_DSHOW = Windows DirectShow
        if cap.isOpened():
            found.append({"index": i,
                          "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})
            cap.release()
    return {"cameras": found, "count": len(found)}


@app.post("/debug/frame")
async def debug_frame(frame: UploadFile = File(...),
                      model_name: str = Form(DEFAULT_MODEL),
                      conf: float = Form(0.5)):
    """
    POST a single image file, get back what YOLO detects.
    Use this to verify YOLO is working before streaming.
    """
    data  = await frame.read()
    np_arr = np.frombuffer(data, np.uint8)
    bgr   = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Could not decode image")
    model = get_model(model_name)
    kp, box, ang = infer_frame(bgr, model, conf)
    if ang is None:
        return {"detected": False, "message": "No person found. Try lower conf value."}
    return {"detected": True,
            "angles": {jn: int(ang[k]) for k, jn in enumerate(JOINT_NAMES)},
            "box": box.tolist() if box is not None else None}


@app.get("/test", response_class=HTMLResponse)
def test_page():
    return HTMLResponse(TEST_PAGE)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(BROWSER_CLIENT)


# ─────────────────────────────────────────────────────────────────────────────
# Browser client HTML  (self-contained, no external dependencies)
# ─────────────────────────────────────────────────────────────────────────────

TEST_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Smart Mirror — Webcam Test</title>
<style>
  body { background:#111; color:#eee; font-family:system-ui,sans-serif;
         display:flex; flex-direction:column; align-items:center; padding:20px; gap:12px; }
  h1   { color:#7df; }
  video  { border:2px solid #2a8fd4; border-radius:6px; max-width:640px; width:100%;
             background:#000; min-height:200px; display:block; }
  canvas { border:2px solid #444; border-radius:6px; max-width:640px; width:100%;
           background:#000; min-height:200px; display:block; }
  button { padding:10px 20px; border-radius:4px; border:none; cursor:pointer;
           background:#3a8fd4; color:#fff; font-size:1rem; margin:4px; }
  #log  { width:100%; max-width:640px; background:#1a1a1a; border-radius:6px;
          padding:12px; font-size:0.8rem; font-family:monospace; min-height:80px;
          white-space:pre-wrap; color:#8f8; }
  .err  { color:#f55 !important; }
</style>
</head>
<body>
<h1>Webcam Diagnostics</h1>
<video id="v" autoplay playsinline muted></video>
<canvas id="c" width="320" height="240"></canvas>
<div>
  <button onclick="startCam()">Start webcam</button>
  <button onclick="captureFrame()">Capture frame</button>
  <button onclick="sendToYolo()">Test YOLO detection</button>
  <button onclick="stopCam()">Stop</button>
</div>
<div id="log">Click "Start webcam" to begin...</div>

<script>
let stream = null;

function log(msg, err=false) {
  const el = document.getElementById("log");
  const line = document.createElement("div");
  line.className = err ? "err" : "";
  line.textContent = new Date().toLocaleTimeString() + "  " + msg;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

async function startCam() {
  try {
    log("Requesting camera access...");
    // Try specific constraints first
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal:640 }, height: { ideal:480 }, frameRate: { ideal:30 } },
        audio: false
      });
    } catch(e1) {
      log("Specific constraints failed (" + e1.message + "), trying any camera...");
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }
    const v = document.getElementById("v");
    v.srcObject = stream;
    await v.play();
    const track    = stream.getVideoTracks()[0];
    const settings = track.getSettings();
    log("Camera started: " + track.label);
    log("Resolution: " + settings.width + "x" + settings.height +
        " @ " + (settings.frameRate||"?") + "fps");
  } catch(e) {
    log("FAILED: " + e.name + ": " + e.message, true);
    if (e.name === "NotAllowedError")
      log("  -> Click the padlock/camera icon in the address bar and allow camera", true);
    else if (e.name === "NotFoundError")
      log("  -> No camera found. Check Device Manager.", true);
    else if (e.name === "NotReadableError")
      log("  -> Camera in use by another app (Teams, OBS, etc.)", true);
  }
}

function captureFrame() {
  const v = document.getElementById("v");
  const c = document.getElementById("c");
  if (!v.srcObject && !v.src) { log("Start camera first", true); return; }
  if (v.readyState < 2) { log("Video not ready yet — wait a moment", true); return; }
  c.width  = v.videoWidth  || 640;
  c.height = v.videoHeight || 480;
  const ctx = c.getContext("2d");
  ctx.drawImage(v, 0, 0, c.width, c.height);

  // Check if frame is blank (all black = video not rendering)
  const pixel = ctx.getImageData(c.width/2, c.height/2, 1, 1).data;
  const brightness = pixel[0] + pixel[1] + pixel[2];
  if (brightness === 0) {
    log("WARNING: captured frame is black! Centre pixel = 0,0,0", true);
    log("  -> Chrome may not render off-screen video. The video is shown above — try again.", true);
  } else {
    log("Frame captured: " + c.width + "x" + c.height +
        "  centre pixel: rgb(" + pixel[0]+","+pixel[1]+","+pixel[2]+")");
  }
}

async function sendToYolo() {
  const c = document.getElementById("c");
  if (c.width === 0) { captureFrame(); }

  log("Sending frame to YOLO...");
  c.toBlob(async blob => {
    const fd = new FormData();
    fd.append("frame", blob, "test.jpg");
    try {
      const res  = await fetch("/debug/frame", { method:"POST", body:fd });
      const data = await res.json();
      if (data.detected) {
        log("YOLO detected a person!");
        Object.entries(data.angles).forEach(([jn, val]) =>
          log("  " + jn + ": " + val + "deg")
        );
      } else {
        log("YOLO: " + data.message, true);
        log("  -> Try moving further back from camera so full body is visible", true);
        log("  -> Or try lower conf at /debug/frame?conf=0.3", true);
      }
    } catch(e) {
      log("Request failed: " + e.message, true);
    }
  }, "image/jpeg", 0.9);
}

function stopCam() {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  document.getElementById("v").srcObject = null;
  log("Camera stopped.");
}

// Auto list devices
navigator.mediaDevices.enumerateDevices().then(devices => {
  const cams = devices.filter(d => d.kind === "videoinput");
  if (cams.length === 0) {
    log("No cameras found by browser!", true);
  } else {
    log("Browser sees " + cams.length + " camera(s):");
    cams.forEach((c,i) => log("  [" + i + "] " + (c.label || "(label hidden until permission granted)")));
  }
}).catch(e => log("enumerateDevices failed: " + e.message, true));
</script>
</body>
</html>"""

BROWSER_CLIENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Smart Mirror — Live Pose Comparison</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0f0f0f; color: #eee;
  font-family: system-ui, sans-serif;
  display: flex; flex-direction: column; align-items: center;
  padding: 12px; gap: 10px;
}
h1 { font-size: 1.2rem; color: #7df; letter-spacing: 1px; }

/* Main output image — full comparison rendered by server */
#output {
  width: 100%; max-width: 1280px;
  background: #111; border-radius: 8px;
  display: block; min-height: 120px;
}
#nodet {
  display: none; width: 100%; max-width: 1280px;
  background: #1a1a1a; border-radius: 8px; padding: 20px;
  text-align: center; color: #fa0; font-size: 0.9rem;
}

/* Controls row */
#controls {
  width: 100%; max-width: 1280px;
  display: flex; flex-wrap: wrap; gap: 8px;
}
.card {
  background: #1c1c1c; border-radius: 8px;
  padding: 12px; flex: 1; min-width: 180px;
}
.card h2 {
  font-size: 0.7rem; color: #777; margin-bottom: 8px;
  text-transform: uppercase; letter-spacing: 1px;
}
input[type=file], select {
  width: 100%; padding: 5px 7px; background: #2a2a2a;
  border: 1px solid #444; color: #eee; border-radius: 4px;
  margin-bottom: 6px; font-size: 0.82rem;
}
input[type=range] {
  width: 100%; margin-bottom: 4px;
}
label { font-size: 0.75rem; color: #888; display: block; margin-bottom: 2px; }
button {
  padding: 7px 14px; border-radius: 4px; border: none;
  cursor: pointer; font-size: 0.82rem; font-weight: 600; margin: 2px 2px 2px 0;
}
.btn-blue  { background: #2a6cb5; color: #fff; }
.btn-green { background: #2a8a55; color: #fff; }
.btn-red   { background: #922;    color: #fff; }
button:disabled { opacity: 0.35; cursor: default; }

/* Stats bar */
#stats {
  width: 100%; max-width: 1280px;
  display: flex; gap: 12px; flex-wrap: wrap; align-items: center;
  font-size: 0.78rem; color: #777;
}
#stats span { background: #1c1c1c; padding: 4px 10px; border-radius: 4px; }
#stats .ok  { color: #6d6; }
#stats .bad { color: #f55; }
#stats .wrn { color: #fa0; }
#statusMsg  { color: #888; }
#statusMsg.ok  { color: #6d6; }
#statusMsg.err { color: #f55; }
#statusMsg.wrn { color: #fa0; }

/* Hidden */
#liveVideo, #captureCanvas {
  position: fixed; bottom:0; right:0;
  width:2px; height:2px; opacity:0;
}
</style>
</head>
<body>

<h1>🏃 Smart Mirror — Live Pose Comparison</h1>

<!-- Server-rendered comparison image -->
<img id="output" src="" alt="Waiting for stream…">
<div id="nodet">⚠ No person detected — make sure your full body is visible</div>

<!-- Instructor video plays automatically below — patient follows along -->
<div style="width:100%;max-width:1280px;background:#111;border-radius:8px;overflow:hidden">
  <div style="padding:6px 12px;background:#1a2a1a;font-size:0.72rem;color:#6d6;font-weight:600">
    ▶ INSTRUCTOR VIDEO — follow along
  </div>
  <video id="instrVideo" controls muted
         style="width:100%;display:block;background:#000;max-height:360px;object-fit:contain">
  </video>
</div>

<!-- Session summary — appears after instructor video ends -->
<div id="summaryBox" style="display:none;width:100%;max-width:1280px;
     background:#1c1c1c;border-radius:8px;padding:16px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
    <h2 style="font-size:1rem;color:#7df">📊 Session Summary</h2>
    <button onclick="dlSummary()"
            style="padding:4px 12px;background:#333;color:#eee;border:none;
                   border-radius:4px;cursor:pointer;font-size:0.75rem">⬇ Download JSON</button>
  </div>
  <div id="summaryMeta" style="font-size:0.85rem;color:#aaa;margin-bottom:10px"></div>
  <table style="width:100%;border-collapse:collapse;font-size:0.78rem">
    <thead>
      <tr style="background:#252525;color:#888">
        <th style="padding:6px 10px;text-align:left">Joint</th>
        <th style="padding:6px 10px">Bad %</th>
        <th style="padding:6px 10px">Mean diff</th>
        <th style="padding:6px 10px">Max diff</th>
      </tr>
    </thead>
    <tbody id="summaryTbody"></tbody>
  </table>
</div>

<!-- Stats bar -->
<div id="stats">
  <span id="sVerdict" class="wrn">WAITING</span>
  <span>ref frame: <b id="sRef">—</b></span>
  <span>patient frame: <b id="sPat">—</b></span>
  <span><b id="sOkPct">—</b> OK</span>
  <span><b id="sLatency">—</b> ms</span>
  <span><b id="sFps">—</b> fps</span>
  <span id="sMirror" style="display:none;color:#fa0">[mirrored]</span>
</div>

<!-- Controls -->
<div id="controls">

  <div class="card">
    <h2>1 · Instructor video</h2>
    <label>Video file</label>
    <input type="file" id="refFile" accept="video/*">
    <label>YOLO model</label>
    <select id="modelSel">
      <option value="yolov8n-pose.pt">yolov8n — fast</option>
      <option value="yolov8m-pose.pt" selected>yolov8m — balanced</option>
      <option value="yolov8x-pose.pt">yolov8x — accurate</option>
    </select>
    <button class="btn-blue" id="prepBtn" onclick="prepare()">
      ⚙ Process instructor
    </button>
  </div>

  <div class="card">
    <h2>2 · Patient source</h2>
    <label>Source</label>
    <select id="srcSel" onchange="toggleSrc()">
      <option value="webcam">Webcam (live)</option>
      <option value="video">Video file (demo)</option>
    </select>
    <div id="videoDiv" style="display:none">
      <label>Patient video</label>
      <input type="file" id="patFile" accept="video/*">
    </div>
    <button class="btn-green" id="startBtn" onclick="startStream()" disabled>
      ▶ Start
    </button>
    <button class="btn-red" id="stopBtn" onclick="stopStream()" disabled>
      ⏹ Stop
    </button>
  </div>

  <div class="card">
    <h2>3 · Settings</h2>
    <label>Angle threshold: <b id="offVal">25</b>°
      <small style="color:#555"> — diff below this = OK</small>
    </label>
    <input type="range" id="offSlider" min="10" max="60" value="25"
           oninput="document.getElementById('offVal').textContent=this.value">
    <label>Send rate: <b id="fpsVal">15</b> fps
      <small style="color:#555"> — lower = less GPU load</small>
    </label>
    <input type="range" id="fpsSlider" min="3" max="30" value="15"
           oninput="document.getElementById('fpsVal').textContent=this.value">

  </div>

  <div class="card">
    <h2>Status</h2>
    <div id="statusMsg">Upload instructor video to begin.</div>
    <br>
    <small style="color:#333">
      <a href="/test"          target="_blank" style="color:#444">webcam test</a> ·
      <a href="/health/webcam" target="_blank" style="color:#444">cameras</a> ·
      <a href="/docs"          target="_blank" style="color:#444">API docs</a>
    </small>
  </div>

</div>

<!-- Hidden capture elements -->
<video  id="liveVideo" autoplay playsinline muted></video>
<canvas id="captureCanvas"></canvas>

<script>
const HOST    = window.location.host;
const WS_PROT = window.location.protocol === "https:" ? "wss" : "ws";

let ws          = null;
let refId       = null;
let mediaStream = null;
let videoSrcURL = null;
let sendLoop    = null;
let wsReady     = true;
let frameN      = 0;
let okCount     = 0;
let fpsSmooth   = 0;
let lastT       = Date.now();

function setStatus(msg, cls="") {
  const el = document.getElementById("statusMsg");
  el.textContent = msg;  el.className = cls;
}

function toggleSrc() {
  document.getElementById("videoDiv").style.display =
    document.getElementById("srcSel").value === "video" ? "block" : "none";
}

// ── Load instructor video into player when file is selected ─────────────────
document.getElementById("refFile").addEventListener("change", function() {
  const file = this.files[0];
  if (!file) return;
  const v = document.getElementById("instrVideo");
  v.src = URL.createObjectURL(file);
  v.load();
});

// ── Prepare ──────────────────────────────────────────────────────────────────
async function prepare() {
  const file = document.getElementById("refFile").files[0];
  if (!file) { setStatus("Select instructor video first.", "err"); return; }
  setStatus("Processing instructor video… (takes 1-2 min per minute of video)", "wrn");
  document.getElementById("prepBtn").disabled = true;

  const fd = new FormData();
  fd.append("ref",        file);
  fd.append("model_name", document.getElementById("modelSel").value);
  try {
    const res  = await fetch("/prepare", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "prepare failed");
    refId = data.ref_id;
    setStatus(
      `Ready — ${data.meta.total} frames (${data.meta.detection_pct}% detected). Press Start.`,
      "ok"
    );
    document.getElementById("startBtn").disabled = false;
  } catch(e) {
    setStatus("Error: " + e.message, "err");
    document.getElementById("prepBtn").disabled = false;
  }
}

// ── Start ────────────────────────────────────────────────────────────────────
async function startStream() {
  if (!refId) { setStatus("Prepare instructor first.", "err"); return; }

  const v = document.getElementById("liveVideo");

  if (document.getElementById("srcSel").value === "webcam") {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width:{ideal:640}, height:{ideal:480}, frameRate:{ideal:30} },
        audio: false
      });
    } catch(e1) {
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({video:true,audio:false});
      } catch(e2) {
        setStatus("Camera error: " + e2.message + " — check /test page", "err");
        return;
      }
    }
    v.srcObject = mediaStream;
  } else {
    const file = document.getElementById("patFile").files[0];
    if (!file) { setStatus("Select patient video.", "err"); return; }
    if (videoSrcURL) URL.revokeObjectURL(videoSrcURL);
    videoSrcURL = URL.createObjectURL(file);
    v.srcObject = null;
    v.src       = videoSrcURL;
    v.loop      = false;
    v.onended   = () => stopStream();
  }

  // Wait for video to be ready
  await new Promise((res, rej) => {
    v.onloadedmetadata = res;
    v.onerror = rej;
    setTimeout(res, 3000);
  });
  await v.play().catch(() => {});

  // WebSocket
  ws = new WebSocket(`${WS_PROT}://${HOST}/ws/compare/${refId}`);
  ws.onopen = () => {
    wsReady = true;
    setStatus("Connected — streaming…", "ok");
    // Auto-play instructor video; send done when it ends
    const instr = document.getElementById("instrVideo");
    instr.currentTime = 0;
    instr.onended = () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ done: true }));
        setStatus("Instructor video finished — generating summary…", "ok");
      }
    };
    instr.play().catch(() => {});
  };
  ws.onclose   = () => { wsReady = false; };
  ws.onerror   = () => setStatus("WebSocket error.", "err");
  ws.onmessage = handleMessage;

  frameN = okCount = fpsSmooth = 0;
  document.getElementById("startBtn").disabled = true;
  document.getElementById("stopBtn").disabled  = false;

  const getInterval = () => 1000 / (parseInt(document.getElementById("fpsSlider").value)||15);
  sendLoop = setInterval(sendFrame, getInterval());
}

// ── Stop ─────────────────────────────────────────────────────────────────────
function stopStream() {
  if (sendLoop)    { clearInterval(sendLoop); sendLoop = null; }
  if (ws)          { ws.close(); ws = null; }
  if (mediaStream) { mediaStream.getTracks().forEach(t=>t.stop()); mediaStream = null; }
  wsReady = false;
  const instr = document.getElementById("instrVideo");
  instr.pause(); instr.onended = null;
  document.getElementById("startBtn").disabled = false;
  document.getElementById("stopBtn").disabled  = true;
  setStatus("Stopped.");
}

// ── Capture and send frame ───────────────────────────────────────────────────
function sendFrame() {
  if (!ws || ws.readyState !== WebSocket.OPEN || !wsReady) return;
  const v = document.getElementById("liveVideo");
  if (v.readyState < 2) return;

  const c = document.getElementById("captureCanvas");
  c.width  = v.videoWidth  || 640;
  c.height = v.videoHeight || 480;
  c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);

  const b64 = c.toDataURL("image/jpeg", 0.8).split(",")[1];
  ws.send(JSON.stringify({
    frame:  b64,
    offset: parseInt(document.getElementById("offSlider").value),
  }));
  wsReady = false;
}

// ── Handle response ──────────────────────────────────────────────────────────
function handleMessage(evt) {
  wsReady = true;
  const msg = JSON.parse(evt.data);

  if (msg.error) { setStatus("Server error: " + msg.error, "err"); return; }

  if (msg.status === "summary") { showSummary(msg.summary, msg.saved_to); stopStream(); return; }

  // No detection
  if (msg.status === "no_detection") {
    document.getElementById("nodet").style.display = "block";
    document.getElementById("output").style.opacity = "0.3";
    setStatus("No person detected — stand further back so full body is visible", "wrn");
    return;
  }

  // Show comparison image
  document.getElementById("nodet").style.display  = "none";
  document.getElementById("output").style.opacity = "1";
  document.getElementById("output").src = "data:image/jpeg;base64," + msg.frame_b64;

  // Stats bar
  frameN++; if (msg.all_ok) okCount++;
  const now = Date.now();
  fpsSmooth = 0.8*fpsSmooth + 0.2*(1000/(now-lastT)); lastT = now;

  document.getElementById("sRef").textContent     = String(msg.ref_frame).padStart(4,"0");
  document.getElementById("sPat").textContent     = String(msg.frame_num).padStart(4,"0");
  document.getElementById("sLatency").textContent = msg.latency_ms.toFixed(0);
  document.getElementById("sFps").textContent     = fpsSmooth.toFixed(1);
  document.getElementById("sOkPct").textContent   =
    frameN > 0 ? (okCount/frameN*100).toFixed(0)+"%" : "—";

  const verdict = document.getElementById("sVerdict");
  verdict.textContent = msg.all_ok ? "✓ GOOD FORM" : "✗ CORRECT FORM";
  verdict.className   = msg.all_ok ? "ok" : "bad";

  document.getElementById("sMirror").style.display = msg.mirrored ? "inline" : "none";

  setStatus(
    `Form accuracy: ${frameN>0?(okCount/frameN*100).toFixed(0):0}% OK`,
    msg.all_ok ? "ok" : "err"
  );
}

let _summaryData = null;

function showSummary(s, savedTo) {
  _summaryData = s;
  const box = document.getElementById("summaryBox");
  box.style.display = "block";
  const m = s.meta;
  const c = m.ok_pct >= 70 ? "#6d6" : m.ok_pct >= 40 ? "#fa0" : "#f55";
  document.getElementById("summaryMeta").innerHTML =
    `Form accuracy: <b style="color:${c}">${m.ok_pct}%</b> &nbsp;|&nbsp; ` +
    `OK: <b>${m.ok_frames} / ${m.total_frames}</b> frames &nbsp;|&nbsp; ` +
    `Saved to: <b>${savedTo}</b>`;
  document.getElementById("summaryTbody").innerHTML =
    Object.entries(s.joint_stats).map(([jn, st]) => {
      const col = st.bad_pct > 50 ? "#f55" : st.bad_pct > 20 ? "#fa0" : "#6d6";
      return `<tr style="border-bottom:1px solid #2a2a2a">
        <td style="padding:5px 10px">${jn.replace(/_/g," ")}</td>
        <td style="padding:5px 10px;text-align:center;color:${col}">${st.bad_pct}%</td>
        <td style="padding:5px 10px;text-align:center">${st.mean_diff>0?"+":""}${st.mean_diff}°</td>
        <td style="padding:5px 10px;text-align:center">${st.max_abs_diff}°</td>
      </tr>`;
    }).join("");
  box.scrollIntoView({ behavior: "smooth" });
  setStatus("Session complete — saved to " + savedTo, "ok");
}

function dlSummary() {
  if (!_summaryData) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(
    new Blob([JSON.stringify(_summaryData, null, 2)], {type:"application/json"}));
  a.download = "session_summary.json";
  a.click();
}

toggleSrc();

// ── Auto-configure when the agent pre-processed the instructor video ──────────
(function autoConfigFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const rid = params.get('ref_id');
  if (!rid) return;

  // Instructor already prepared server-side — set ref_id and unlock Start
  refId = rid;
  document.getElementById('startBtn').disabled = false;
  document.getElementById('prepBtn').disabled  = true;
  setStatus('Instructor ready — press Start whenever you are!', 'ok');

  // Load the instructor video directly from the server
  const instrVid = document.getElementById('instrVideo');
  instrVid.src = '/instructor-video';
  instrVid.load();

  // Dim the "Process instructor" card so it's clear nothing is needed there
  const cards = document.querySelectorAll('.card');
  if (cards.length > 0) cards[0].style.opacity = '0.45';
})();
</script>
</body>
</html>"""


