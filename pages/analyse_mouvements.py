# pages/analyse_mouvements.py
# Analyse de mouvement : upload prioritaire, fallback sur la vidéo préparée.
# Pas d'import externe "opticalflow" : toutes les fonctions (serie_magnitude, lire_frame_a,
# farneback_pair, heatmap, vectors_overlay) sont définies ici.
# Pipeline robuste : FFmpeg -> JPG -> OpenCV Farneback -> métriques + vignettes.

import math
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st

from core_media import initialiser_repertoires, info_ffmpeg

# =========================
# Outillage système
# =========================

def _ffmpeg_path() -> Optional[str]:
    p, _ = info_ffmpeg()
    return p

def _run(cmd: List[str]) -> Tuple[bool, str]:
    import subprocess
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out = (res.stdout or "").strip()
        err = (res.stderr or "").strip()
        log = "\n".join([s for s in (out, err) if s]).strip()
        return True, log
    except subprocess.CalledProcessError as e:
        out = (e.stdout or "").strip()
        err = (e.stderr or "").strip()
        log = "\n".join([s for s in (out, err) if s]).strip() or str(e)
        return False, log
    except Exception as e:
        return False, f"Erreur d'exécution : {e}"

def _load_cv2():
    try:
        import cv2  # type: ignore
        return cv2, None
    except Exception as e:
        return None, f"OpenCV introuvable : {e}. Ajoute 'opencv-python-headless' dans requirements.txt."

# =========================
# Fonctions "opticalflow" intégrées
# =========================

def lire_frame_a(cv2, chemin: Path):
    """Lit une image disque en BGR et retourne (ok, bgr)."""
    arr = cv2.imread(str(chemin), cv2.IMREAD_COLOR)
    if arr is None:
        return False, None
    return True, arr

def farneback_pair(cv2, prev_gray: np.ndarray, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Flux optique Farneback entre deux images grises. Retourne (flow, magnitude)."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray,
        None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0
    )
    mag = np.linalg.norm(flow, axis=2)
    return flow, mag

def heatmap(cv2, mag: np.ndarray) -> np.ndarray:
    """Génère une heatmap RGB à partir d'une magnitude."""
    m = mag.copy()
    if not np.isfinite(m).all():
        m[np.isnan(m)] = 0.0
    vmax = np.percentile(m, 99) if np.any(m > 0) else 1.0
    if vmax <= 0:
        vmax = 1.0
    norm = np.clip((m / vmax) * 255.0, 0, 255).astype(np.uint8)
    hm_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    hm_rgb = cv2.cvtColor(hm_bgr, cv2.COLOR_BGR2RGB)
    return hm_rgb

def vectors_overlay(cv2, frame_rgb: np.ndarray, flow: np.ndarray, step: int = 16) -> np.ndarray:
    """Trace un champ de vecteurs échantillonné sur l'image RGB."""
    h, w = frame_rgb.shape[:2]
    overlay = frame_rgb.copy()
    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]
            x2 = int(round(x + fx))
            y2 = int(round(y + fy))
            cv2.arrowedLine(overlay, (x, y), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
    return overlay

def serie_magnitude(cv2, images_gray: List[np.ndarray], seuil_pix: float):
    """Calcule, pour une série d'images grises, les métriques par pas et l'énergie."""
    metriques = []
    energies = []
    for i in range(1, len(images_gray)):
        flow, mag = farneback_pair(cv2, images_gray[i-1], images_gray[i])
        m = {
            "magnitude_moyenne": float(np.mean(mag)),
            "magnitude_ecart_type": float(np.std(mag)),
            "magnitude_p95": float(np.percentile(mag, 95)),
            "ratio_pixels_mobiles": float(np.mean(mag > seuil_pix)),
            "energie_mouvement": float(np.sum(mag)),
        }
        metriques.append(m)
        energies.append(m["energie_mouvement"])
    return metriques, np.array(energies, dtype=float)

# =========================
# Extraction et chargement des frames
# =========================

def extraire_frames_ffmpeg(ff: str, video: Path, dossier: Path, fps_ech: float, largeur: int) -> Tuple[bool, str]:
    """Extrait des JPG à cadence régulière, redimensionnés pour l'analyse."""
    if dossier.exists():
        try:
            shutil.rmtree(dossier)
        except Exception:
            pass
    dossier.mkdir(parents=True, exist_ok=True)
    motif = str(dossier / "frame_%06d.jpg")
    filtre = f"fps={fps_ech},scale={largeur}:-2"
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video), "-vf", filtre, "-q:v", "2", motif]
    return _run(cmd)

def charger_images_gris_et_rgb(cv2, dossier: Path) -> Tuple[List[np.ndarray], List[np.ndarray], List[Path]]:
    """Charge les images JPG en gris et en RGB, retourne aussi la liste des chemins."""
    fichiers = sorted(dossier.glob("frame_*.jpg"))
    grays, rgbs = [], []
    for f in fichiers:
        ok, bgr = lire_frame_a(cv2, f)
        if not ok:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        grays.append(gray)
        rgbs.append(rgb)
    return grays, rgbs, fichiers

# =========================
# Page Streamlit
# =========================

BASE_DIR, REP_SORTIE, REP_TMP = initialiser_repertoires()

st.set_page_config(page_title="Analyse des mouvements (flux optique)", layout="wide")
st.title("Analyse des mouvements (flux optique)")
st.markdown("**www.codeandcortex.fr**")

ff = _ffmpeg_path()
if not ff:
    st.error("FFmpeg introuvable. Fournis un binaire ./bin/ffmpeg ou vérifie /usr/bin/ffmpeg.")
    st.stop()

cv2, cv_err = _load_cv2()
if cv2 is None:
    st.error(cv_err)
    st.stop()

st.subheader("Source de la vidéo")
source = st.radio("Choisir la source", ["Importer un MP4", "Utiliser la vidéo préparée"], index=0, horizontal=True)

video_path: Optional[Path] = None
if source == "Importer un MP4":
    up = st.file_uploader("Importer une vidéo (.mp4)", type=["mp4"], key="analyse_upload")
    if up is not None:
        video_path = REP_TMP / f"analyse_{up.name}"
        with open(video_path, "wb") as g:
            g.write(up.read())
        st.success(f"Fichier uploadé : {video_path.name}")
else:
    if st.session_state.get("video_base"):
        p = Path(st.session_state["video_base"])
        if p.exists():
            video_path = p
            st.info(f"Vidéo préparée utilisée : {p.name}")
        else:
            st.warning("La vidéo préparée est introuvable sur le disque. Importez un MP4.")
    else:
        st.warning("Aucune vidéo préparée en mémoire. Importez un MP4.")

st.subheader("Paramètres")
c1, c2, c3 = st.columns(3)
with c1:
    fps_ech = st.number_input("Cadence d’échantillonnage (images/s)", min_value=1, max_value=30, value=5, step=1)
with c2:
    largeur_det = st.selectbox("Largeur d’analyse (px)", [480, 640, 960], index=1)
with c3:
    seuil_pix = st.number_input("Seuil pixel 'mobile' (px/frame)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)

c4, c5, c6 = st.columns(3)
with c4:
    nb_vignettes = st.number_input("Nombre de vignettes", min_value=8, max_value=200, value=40, step=4)
with c5:
    mode_vignettes = st.selectbox("Style vignette", ["RGB", "Heatmap", "Vecteurs"], index=0)
with c6:
    afficher_log = st.checkbox("Afficher le journal FFmpeg", value=False)

if st.button("Lancer l’analyse", type="primary"):
    if video_path is None:
        st.error("Aucune source vidéo. Importez un MP4 ou sélectionnez la vidéo préparée.")
        st.stop()

    frames_dir = (BASE_DIR / "frames_analysis" / video_path.stem).resolve()
    ok_ext, log_ext = extraire_frames_ffmpeg(ff, video_path, frames_dir, float(fps_ech), int(largeur_det))
    if not ok_ext:
        st.error("Échec extraction des images avec FFmpeg.")
        if afficher_log:
            st.code(log_ext or "(log vide)", language="bash")
        st.stop()

    imgs_gray, imgs_rgb, chemins = charger_images_gris_et_rgb(cv2, frames_dir)
    if len(imgs_gray) < 2:
        st.error("Trop peu d’images extraites. La page resterait noire.")
        if afficher_log:
            st.code(log_ext or "(log vide)", language="bash")
        st.stop()

    metriques, energies = serie_magnitude(cv2, imgs_gray, float(seuil_pix))
    moy = float(np.mean(energies))
    std = float(np.std(energies))
    seuil_pic = moy + 2.0 * std
    pics_idx = [i+1 for i, e in enumerate(energies) if e >= seuil_pic]

    st.subheader("Métriques globales")
    st.write(f"Énergie moyenne : {moy:.2f}")
    st.write(f"Écart-type : {std:.2f}")
    st.write(f"Seuil de pic (moy + 2σ) : {seuil_pic:.2f}")
    if pics_idx:
        temps_pics = [idx / float(fps_ech) for idx in pics_idx]
        st.write("Pics détectés : " + ", ".join([f"t≈{t:.1f}s (#{idx})" for t, idx in zip(temps_pics, pics_idx)]))
    else:
        st.write("Aucun pic détecté au seuil courant.")

    import pandas as pd
    lignes = []
    for i, m in enumerate(metriques, start=1):
        lignes.append({
            "index_sequence": i,
            "temps_s_approx": i / float(fps_ech),
            "magnitude_moyenne": m["magnitude_moyenne"],
            "magnitude_ecart_type": m["magnitude_ecart_type"],
            "magnitude_p95": m["magnitude_p95"],
            "ratio_pixels_mobiles": m["ratio_pixels_mobiles"],
            "energie_mouvement": m["energie_mouvement"],
        })
    df = pd.DataFrame(lignes)
    st.download_button("Télécharger les métriques (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="metriques_flux_optique.csv", mime="text/csv")

    st.subheader("Vignettes réparties sur la vidéo")
    N = len(imgs_rgb)
    idxs = np.linspace(0, N-1, num=int(nb_vignettes), dtype=int)
    cols_par_ligne = 8
    lignes = math.ceil(len(idxs) / cols_par_ligne)
    k = 0

    if mode_vignettes == "RGB":
        for _ in range(lignes):
            cols = st.columns(cols_par_ligne)
            for c in cols:
                if k >= len(idxs):
                    break
                i = int(idxs[k])
                c.image(imgs_rgb[i], caption=f"#{i} • t≈{i/float(fps_ech):.1f}s", use_container_width=False)
                k += 1

    elif mode_vignettes == "Heatmap":
        for _ in range(lignes):
            cols = st.columns(cols_par_ligne)
            for c in cols:
                if k >= len(idxs):
                    break
                i = int(idxs[k])
                if i == 0:
                    img = np.zeros_like(imgs_rgb[i])
                else:
                    _, mag = farneback_pair(cv2, imgs_gray[i-1], imgs_gray[i])
                    img = heatmap(cv2, mag)
                c.image(img, caption=f"HM #{i}", use_container_width=False)
                k += 1

    else:  # Vecteurs
        for _ in range(lignes):
            cols = st.columns(cols_par_ligne)
            for c in cols:
                if k >= len(idxs):
                    break
                i = int(idxs[k])
                if i == 0:
                    img = imgs_rgb[i]
                else:
                    flow, _ = farneback_pair(cv2, imgs_gray[i-1], imgs_gray[i])
                    img = vectors_overlay(cv2, imgs_rgb[i], flow, step=16)
                c.image(img, caption=f"Vec #{i}", use_container_width=False)
                k += 1

    if afficher_log:
        with st.expander("Journal FFmpeg"):
            st.code(log_ext or "(log vide)", language="bash")
