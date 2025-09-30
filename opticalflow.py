# opticalflow.py
# Mesures de flux optique Farneback et utilitaires d’affichage.

import cv2
import numpy as np

def lire_frame_a(cap: cv2.VideoCapture, t: float):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"Impossible de lire la frame à {t}s")
    return frame

def farneback_pair(g1, g2):
    return cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def magnitude_moyenne(flow):
    return float(np.mean(np.linalg.norm(flow, axis=2)))

def heatmap(flow):
    mag = np.linalg.norm(flow, axis=2)
    norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)

def vectors_overlay(frame, flow, step=16):
    img = frame.copy(); h, w = img.shape[:2]
    fx, fy = flow[...,0], flow[...,1]
    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = int(fx[y,x]), int(fy[y,x])
            cv2.arrowedLine(img, (x,y), (x+dx,y+dy), (0,255,0), 1, tipLength=0.3)
    return img

def serie_magnitude(video_path: str, stride_ms: int = 200):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d’ouvrir la vidéo : {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duree = total / fps if fps > 0 else 0
    t = 0.0
    times, mags = [], []
    ret, prev = cap.read()
    if not ret:
        cap.release(); return times, mags
    prevg = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    while True:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = farneback_pair(prevg, gray)
        times.append(t)
        mags.append(magnitude_moyenne(flow))
        prevg = gray
        t += stride_ms / 1000.0
        if t > duree:
            break
    cap.release()
    return times, mags
