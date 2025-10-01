# pages/analyse_mouvement.py
# Analyse des mouvements par flux optique (OpenCV) avec métriques de variation
# et affichage de vignettes uniformément réparties sur toute la vidéo.
# Hypothèse : main.py a déjà préparé une vidéo et stocké son chemin dans st.session_state["video_base"].

import io
import math
from pathlib import Path

import numpy as np
import streamlit as st

from core_media import initialiser_repertoires, SEUIL_APERCU_OCTETS

# =========================
# Utilitaires de lecture
# =========================

def _charger_cv2():
    """Import différé d'OpenCV (opencv-python-headless recommandé sur Streamlit Cloud)."""
    try:
        import cv2  # type: ignore
        return cv2, None
    except Exception as e:
        return None, f"OpenCV introuvable : {e}. Ajoute 'opencv-python-headless' dans requirements.txt."

def _ouvrir_video(cv2, chemin: Path):
    """Ouvre la vidéo et retourne (cap, nb_frames, fps, largeur, hauteur) ou (None, ...)."""
    cap = cv2.VideoCapture(str(chemin))
    if not cap.isOpened():
        return None, 0, 0.0, 0, 0
    nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    return cap, nb, fps, w, h

def _lire_frame(cv2, cap, index: int):
    """Positionne la vidéo sur index et lit 1 frame. Retourne (ok, frame_bgr)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    return ok, frame

# =========================
# Flux optique et métriques
# =========================

def _calculer_flux_optique_farneback(cv2, prev_gray: np.ndarray, gray: np.ndarray):
    """Calcule le flux optique dense de Farneback entre prev_gray et gray."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray,
        None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0
    )
    return flow  # shape (H, W, 2)

def _magnitude(flow: np.ndarray):
    """Retourne la magnitude du flux (norme L2)."""
    mag = np.linalg.norm(flow, axis=2)
    return mag

def _metriques_mouvement(mag: np.ndarray, seuil_pix: float):
    """Calcule les métriques de variation à partir d'une carte de magnitude."""
    m = float(np.mean(mag))
    s = float(np.std(mag))
    p95 = float(np.percentile(mag, 95))
    ratio_mobile = float(np.mean(mag > seuil_pix))  # proportion de pixels au-dessus du seuil
    energie = float(np.sum(mag))
    return {
        "magnitude_moyenne": m,
        "magnitude_ecart_type": s,
        "magnitude_p95": p95,
        "ratio_pixels_mobiles": ratio_mobile,
        "energie_mouvement": energie,
    }

# =========================
# Vignettes
# =========================

def _fabriquer_vignette(cv2, frame_bgr: np.ndarray, target_largeur: int = 200):
    """Redimensionne la frame et convertit en RGB pour affichage Streamlit."""
    h, w = frame_bgr.shape[:2]
    if w <= 0 or h <= 0:
        return None
    scale = target_largeur / float(w)
    new_w = target_largeur
    new_h = max(1, int(round(h * scale)))
    vignette = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    vignette_rgb = cv2.cvtColor(vignette, cv2.COLOR_BGR2RGB)
    return vignette_rgb

# =========================
# Page Streamlit
# =========================

BASE_DIR, REP_SORTIE, REP_TMP = initialiser_repertoires()

st.set_page_config(page_title="Analyse des mouvements (flux optique)", layout="wide")
st.title("Analyse des mouvements (flux optique)")
st.markdown("**www.codeandcortex.fr**")

# Vérification de la vidéo préparée
if not st.session_state.get("video_base"):
    st.warning("Aucune vidéo préparée. Va d’abord sur la page d’accueil pour préparer la source.")
    st.stop()

video_path = Path(st.session_state["video_base"])
if not video_path.exists():
    st.error("La vidéo préparée est introuvable sur le disque.")
    st.stop()

cv2, err = _charger_cv2()
if cv2 is None:
    st.error(err)
    st.stop()

# Paramètres d'analyse
st.subheader("Paramètres d'analyse")
c1, c2, c3 = st.columns(3)
with c1:
    nb_vignettes = st.number_input("Nombre de vignettes sur toute la vidéo", min_value=8, max_value=200, value=40, step=4)
with c2:
    pas_frames = st.number_input("Pas d'analyse (1 = toutes les frames)", min_value=1, max_value=50, value=5, step=1)
with c3:
    seuil_mouvement = st.number_input("Seuil pixel 'mobile' (px/frame)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)

# Lecture vidéo
cap, nb_frames, fps, W, H = _ouvrir_video(cv2, video_path)
if cap is None or nb_frames <= 1:
    st.error("Impossible d'ouvrir la vidéo ou nombre de frames insuffisant.")
    st.stop()

duree_s = nb_frames / fps if fps > 0 else 0.0
st.caption(f"Vidéo : {W}×{H} • {fps:.2f} i/s • {nb_frames} frames • ~{duree_s:.1f} s")

# Sélection des indices pour vignettes (répartition uniforme)
indices_vignettes = np.linspace(0, nb_frames - 1, num=int(nb_vignettes), dtype=int)

# Boucle d'analyse
metriques_par_frame = []
vignettes = []
times = []

st.info("Analyse en cours...")

# Lire la première frame grise (référence)
ok0, frame0 = _lire_frame(cv2, cap, 0)
if not ok0:
    st.error("Impossible de lire la première image de la vidéo.")
    st.stop()
prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

for idx in range(pas_frames, nb_frames, pas_frames):
    ok, frame = _lire_frame(cv2, cap, idx)
    if not ok or frame is None:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Flux optique Farneback
    flow = _calculer_flux_optique_farneback(cv2, prev_gray, gray)
    mag = _magnitude(flow)

    # Métriques
    met = _metriques_mouvement(mag, seuil_mouvement)
    metriques_par_frame.append((idx, met))
    t = idx / fps if fps > 0 else 0.0
    times.append(t)

    # Conserver la référence
    prev_gray = gray

    # Vignette à cet index si demandé
    # (on profite de la boucle : si idx est proche d'un des indices uniformes, on prend la vignette)
    # On utilise une tolérance de ±pas_frames/2
    if np.any(np.abs(indices_vignettes - idx) <= (pas_frames // 2 + 1)):
        v = _fabriquer_vignette(cv2, frame, target_largeur=200)
        if v is not None:
            vignettes.append((idx, t, v))

cap.release()

# Résumé des métriques globales
if metriques_par_frame:
    # Série sur l'énergie (utile pour repérer des pics)
    energies = np.array([m["energie_mouvement"] for _, m in metriques_par_frame], dtype=float)
    moy = float(np.mean(energies))
    std = float(np.std(energies))
    seuil_pic = moy + 2.0 * std
    pics = [(idx, t) for (idx, _), t, e in zip(metriques_par_frame, times, energies) if e >= seuil_pic]

    st.subheader("Métriques globales")
    st.write(f"Énergie moyenne du mouvement : {moy:.2f}")
    st.write(f"Écart-type de l’énergie : {std:.2f}")
    st.write(f"Seuil de détection de pics (moy + 2σ) : {seuil_pic:.2f}")
    if pics:
        premiers_pics = ", ".join([f"t≈{t:.1f}s (frame {idx})" for idx, t in pics[:10]])
        st.write(f"Pics détectés : {premiers_pics}")
    else:
        st.write("Aucun pic détecté au seuil courant.")

    # Export CSV des métriques par frame
    import pandas as pd
    lignes = []
    for (idx, met), t in zip(metriques_par_frame, times):
        lignes.append({
            "frame": idx,
            "temps_s": t,
            "magnitude_moyenne": met["magnitude_moyenne"],
            "magnitude_ecart_type": met["magnitude_ecart_type"],
            "magnitude_p95": met["magnitude_p95"],
            "ratio_pixels_mobiles": met["ratio_pixels_mobiles"],
            "energie_mouvement": met["energie_mouvement"],
        })
    df = pd.DataFrame(lignes)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger les métriques (CSV)", data=csv_bytes, file_name="metriques_flux_optique.csv", mime="text/csv")

# Grille de vignettes
st.subheader("Vignettes uniformément réparties sur la vidéo")
if not vignettes:
    st.info("Aucune vignette n’a pu être générée. Vérifie la lecture vidéo et les paramètres.")
else:
    # Affiche en grille (8 colonnes par défaut)
    cols_par_ligne = 8
    lignes = math.ceil(len(vignettes) / cols_par_ligne)
    k = 0
    for _ in range(lignes):
        cols = st.columns(cols_par_ligne)
        for c in cols:
            if k >= len(vignettes):
                break
            idx, t, vignette_rgb = vignettes[k]
            c.image(vignette_rgb, caption=f"t≈{t:.1f}s (#{idx})", use_container_width=False)
            k += 1

# Note d’affichage
st.caption(
    "Remarques : les vignettes sont en RGB (conversion depuis BGR OpenCV). "
    "Le nombre de vignettes est limité pour préserver la fluidité de la page. "
    "Augmentez 'Nombre de vignettes' pour un maillage plus fin."
)
