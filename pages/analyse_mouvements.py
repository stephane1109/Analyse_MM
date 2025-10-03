# pages/analyse_mouvements.py
# Analyse des mouvements avec choix du pas d’analyse (intervalle entre frames analysées),
# extraction "toutes les frames (timelapse/natif)" ou "cadence fixe (i/s)", et
# visualisation des anomalies (encadrées en rouge) sur un APERÇU GLOBAL basé sur TOUTES les images extraites.

import math
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st

from core_media import initialiser_repertoires, info_ffmpeg

# =============================
# Utilitaires système
# =============================

def trouver_ffmpeg() -> Optional[str]:
    """Retourne le chemin de ffmpeg si disponible, sinon None."""
    p, _ = info_ffmpeg()
    return p

def executer(cmd: List[str]) -> Tuple[bool, str]:
    """Exécute une commande système et retourne (ok, log)."""
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

def importer_cv2():
    """Import différé d'OpenCV (opencv-python-headless recommandé sur Streamlit Cloud)."""
    try:
        import cv2  # type: ignore
        return cv2, None
    except Exception as e:
        return None, f"OpenCV introuvable : {e}. Ajoute 'opencv-python-headless' dans requirements.txt."

# =============================
# Extraction d’images (FFmpeg)
# =============================

def extraire_frames_1080p(
    ffmpeg: str,
    video: Path,
    dossier: Path,
    mode_extraction: str,
    fps_ech: int = 4,
) -> Tuple[bool, str]:
    """
    Extrait des images JPG en 1080p (largeur 1920).
    mode_extraction = "natifs" -> toutes les frames sources (timelapse, VFR), sans filtre fps (vsync vfr).
    mode_extraction = "fixe"   -> fps_ech images/seconde (uniforme).
    Les fichiers sont nommés frame_%06d.jpg.
    """
    if dossier.exists():
        try:
            shutil.rmtree(dossier)
        except Exception:
            pass
    dossier.mkdir(parents=True, exist_ok=True)

    motif = str(dossier / "frame_%06d.jpg")
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video)]

    if mode_extraction == "natifs":
        cmd += ["-vf", "scale=1920:-2", "-vsync", "vfr", "-q:v", "2", motif]
    else:
        filtre = f"fps={fps_ech},scale=1920:-2"
        cmd += ["-vf", filtre, "-q:v", "2", motif]

    return executer(cmd)

# =============================
# Chargement et utilitaires image
# =============================

def charger_images_gris_et_rgb(cv2, dossier: Path) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Charge les images extraites en niveaux de gris et en RGB (pour vignettes)."""
    fichiers = sorted(dossier.glob("frame_*.jpg"))
    grays, rgbs = [], []
    for f in fichiers:
        bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        grays.append(gray)
        rgbs.append(rgb)
    return grays, rgbs

def encadrer_rouge(cv2, img_rgb: np.ndarray, epaisseur: int = 8) -> np.ndarray:
    """Dessine un cadre rouge autour d'une image RGB pour signaler une anomalie."""
    vis = img_rgb.copy()
    h, w = vis.shape[:2]
    cv2.rectangle(vis, (0, 0), (w-1, h-1), (255, 0, 0), thickness=epaisseur)
    return vis

# =============================
# Flux optique + métriques
# =============================

def farneback(cv2, prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Flux optique dense Farneback (retourne (H,W,2) vecteurs (dx,dy))."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray,
        None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0
    )
    return flow

def stats_circulaires(flow: np.ndarray) -> Tuple[float, float]:
    """Direction dominante (degrés) et dispersion (1-R) des directions de mouvement."""
    dx = flow[..., 0].astype(np.float32)
    dy = flow[..., 1].astype(np.float32)
    angle = np.arctan2(dy, dx)
    ux, uy = np.cos(angle), np.sin(angle)
    R_x, R_y = float(np.mean(ux)), float(np.mean(uy))
    R = float(np.sqrt(R_x**2 + R_y**2))
    direction_deg = float(np.degrees(np.arctan2(R_y, R_x)))
    dispersion = float(1.0 - R)
    return direction_deg, dispersion

def metriques_par_pas(flow: np.ndarray) -> Dict[str, float]:
    """Métriques par pas à partir du champ de flux."""
    mag = np.linalg.norm(flow, axis=2).astype(np.float32)
    direction, dispersion = stats_circulaires(flow)
    return {
        "magnitude_moyenne": float(np.mean(mag)),
        "magnitude_ecart_type": float(np.std(mag)),
        "magnitude_p95": float(np.percentile(mag, 95)),
        "energie_mouvement": float(np.sum(mag)),
        "direction_dominante_deg": direction,
        "dispersion_direction": dispersion,
    }

def zscore(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Standardise x → z = (x - mu) / sigma. Retourne (z, mu, sigma)."""
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma < 1e-12:
        sigma = 1.0
    return (x - mu) / sigma, mu, sigma

# =============================
# Page Streamlit
# =============================

BASE_DIR, REP_SORTIE, REP_TMP = initialiser_repertoires()

st.set_page_config(page_title="Analyse des mouvements (pas d’analyse + anomalies)", layout="wide")
st.title("Analyse des mouvements avec pas d’analyse et anomalies")
st.markdown("**www.codeandcortex.fr**")

st.markdown(
    "Principe général. Le flux optique estime les déplacements de pixels entre deux images successives. "
    "Les images sont extraites en 1080p soit à **cadence fixe**, soit en **conservant toutes les frames natives** (idéal pour les timelapses). "
    "Pour chaque pas d’analyse, on calcule : magnitude moyenne, écart-type, 95e percentile, énergie, direction dominante et dispersion. "
    "Un score composite est formé en combinant les z-scores de la magnitude moyenne et de l’énergie. "
    "Les pas dont le score composite dépasse nettement la moyenne (z ≥ 2.5) sont considérés comme des anomalies et **encadrés en rouge**."
)

ff = trouver_ffmpeg()
if not ff:
    st.error("FFmpeg introuvable. Fournis un binaire ./bin/ffmpeg ou vérifie /usr/bin/ffmpeg.")
    st.stop()

cv2, cv_err = importer_cv2()
if cv2 is None:
    st.error(cv_err)
    st.stop()

# Source vidéo
st.subheader("Source vidéo")
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

# Paramètres d’extraction et d’analyse
st.subheader("Paramètres d’extraction et d’analyse")
col1, col2, col3 = st.columns(3)
with col1:
    mode_extraction = st.radio("Mode d’extraction d’images", ["Toutes les frames (timelapse/natif)", "Cadence fixe (i/s)"], index=0)
with col2:
    fps_ech = st.number_input("Cadence fixe (si sélectionnée)", min_value=1, max_value=60, value=4, step=1)
with col3:
    pas_analyse = st.number_input("Pas d’analyse (1 = chaque image)", min_value=1, max_value=100, value=1, step=1)

st.caption(
    "• **Pas d’analyse** : intervalle entre images utilisées pour le calcul (1,2,3...). "
    "En timelapse, mets 1 pour exploiter chaque image extraite. "
    "• **Mode d’extraction** : « frames natives » (toutes) ou « cadence fixe ». "
    "Si tu ne vois que quelques vignettes, passe en *frames natives* ou augmente la cadence fixe."
)

lancer = st.button("Analyser", type="primary")

if lancer:
    if video_path is None:
        st.error("Aucune source vidéo. Importez un MP4 ou sélectionnez la vidéo préparée.")
        st.stop()

    frames_dir = (BASE_DIR / "frames_analysis" / video_path.stem).resolve()
    mode = "natifs" if mode_extraction.startswith("Toutes") else "fixe"
    ok_ext, log_ext = extraire_frames_1080p(ff, video_path, frames_dir, mode_extraction=mode, fps_ech=int(fps_ech))
    if not ok_ext:
        st.error("Échec extraction des images avec FFmpeg.")
        with st.expander("Journal FFmpeg"):
            st.code(log_ext or "(vide)", language="bash")
        st.stop()

    imgs_gray, imgs_rgb = charger_images_gris_et_rgb(cv2, frames_dir)
    total_frames = len(imgs_gray)
    if total_frames < 2:
        st.error("Trop peu d’images extraites pour analyser.")
        with st.expander("Journal FFmpeg"):
            st.code(log_ext or "(vide)", language="bash")
        st.stop()

    # Info explicative sur le nombre d'images extraites
    mode_txt = "frames natives" if mode == "natifs" else f"cadence fixe = {int(fps_ech)} i/s"
    st.info(f"Images extraites : {total_frames} ({mode_txt}). "
            f"Si ce nombre est trop faible, choisis « frames natives » ou augmente la cadence fixe.")

    # Sous-échantillonnage pour le pas d'analyse (1 = chaque image, 2 = une sur deux, etc.)
    indices = list(range(0, total_frames, int(pas_analyse)))
    if len(indices) < 2:
        st.error("Le pas d’analyse est trop grand pour la longueur de la séquence.")
        st.stop()

    # Calcul du flux optique et des métriques entre images espacées par le pas choisi
    lignes: List[Dict[str, float]] = []
    for k in range(1, len(indices)):
        i_prev = indices[k-1]
        i_curr = indices[k]
        flow = farneback(cv2, imgs_gray[i_prev], imgs_gray[i_curr])
        met = metriques_par_pas(flow)
        lignes.append({
            "etape": k,
            "frame_prev": i_prev,
            "frame_curr": i_curr,
            "magnitude_moyenne": met["magnitude_moyenne"],
            "magnitude_ecart_type": met["magnitude_ecart_type"],
            "magnitude_p95": met["magnitude_p95"],
            "energie_mouvement": met["energie_mouvement"],
            "direction_dominante_deg": met["direction_dominante_deg"],
            "dispersion_direction": met["dispersion_direction"],
        })

    df = pd.DataFrame(lignes)

    # Baseline globale
    moyennes_globales = df[[
        "magnitude_moyenne", "magnitude_ecart_type", "magnitude_p95",
        "energie_mouvement", "direction_dominante_deg", "dispersion_direction"
    ]].mean(numeric_only=True).to_dict()

    # Score composite et anomalies
    zM, _, _ = zscore(df["magnitude_moyenne"].to_numpy(dtype=np.float64))
    zE, _, _ = zscore(df["energie_mouvement"].to_numpy(dtype=np.float64))
    df["score_composite_z"] = (zM + zE) / 2.0
    seuil_z = 2.5
    df["anomalie"] = df["score_composite_z"] >= seuil_z

    # Affichages essentiels
    st.subheader("Moyennes globales (baseline)")
    st.dataframe(pd.DataFrame([moyennes_globales]).T.rename(columns={0: "valeur"}))

    st.subheader("Anomalies détectées (encadrées en rouge)")
    nb_ano = int(df["anomalie"].sum())
    st.write(f"Nombre d’anomalies détectées : {nb_ano} (seuil z ≥ {seuil_z:.1f})")

    # Vignettes des anomalies (top 16 par z décroissant), avec encadrement rouge
    if nb_ano > 0:
        ord_ano = df[df["anomalie"]].sort_values("score_composite_z", ascending=False)
        top = ord_ano.head(16)
        cols_par_ligne = 8
        k = 0
        for _ in range(math.ceil(len(top) / cols_par_ligne)):
            cols = st.columns(cols_par_ligne)
            for c in cols:
                if k >= len(top):
                    break
                row = top.iloc[k]
                idx = int(row["frame_curr"])
                z_here = float(row["score_composite_z"])
                if 0 <= idx < len(imgs_rgb):
                    vis = encadrer_rouge(cv2, imgs_rgb[idx], epaisseur=8)
                    c.image(vis, caption=f"frame #{idx} • z={z_here:.2f}", use_container_width=False)
                k += 1
    else:
        st.info("Aucune anomalie forte détectée.")

    # Téléchargement CSV
    st.subheader("Téléchargement des indices")
    st.download_button(
        "Télécharger les indices et anomalies (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="indices_mouvement_et_anomalies.csv",
        mime="text/csv"
    )

    # ======= APERÇU GLOBAL corrigé (sur TOUTES les images extraites) =======
    st.subheader("Aperçu global (vignettes réparties, anomalies encadrées)")
    # On base l’aperçu sur le nombre TOTAL d’images extraites (imgs_rgb), pas sur le nombre d’étapes analysées.
    N = len(imgs_rgb)
    nb_vignettes = min(96, N)  # on monte à 96 pour mieux couvrir
    idxs = np.linspace(0, N - 1, num=nb_vignettes, dtype=int)

    # Pour savoir si une frame est marquée anomalie, on cherche si elle est un 'frame_curr' anormal
    df_ano = df[df["anomalie"]]
    frames_anormales = set(df_ano["frame_curr"].astype(int).tolist())
    z_map = {int(r["frame_curr"]): float(r["score_composite_z"]) for _, r in df.iterrows()}

    cols_par_ligne = 8
    k = 0
    for _ in range(math.ceil(len(idxs) / cols_par_ligne)):
        cols = st.columns(cols_par_ligne)
        for c in cols:
            if k >= len(idxs):
                break
            fr = int(idxs[k])
            img = encadrer_rouge(cv2, imgs_rgb[fr], epaisseur=6) if fr in frames_anormales else imgs_rgb[fr]
            z_here = z_map.get(fr, None)
            cap = f"frame #{fr}" + (f" • z={z_here:.2f}" if z_here is not None else "")
            c.image(img, caption=cap, use_container_width=False)
            k += 1

    # Explications pédagogiques (rappel)
    with st.expander("Pourquoi je ne vois que peu de vignettes parfois ?"):
        st.markdown(
            "- Si tu choisis **Cadence fixe** avec une petite valeur (ex. 1–4 i/s) et que la vidéo est courte, FFmpeg extraira peu d’images. "
            "Passe en **Toutes les frames (timelapse/natif)** pour récupérer chaque image distincte.\n"
            "- Le **pas d’analyse** n’affecte plus l’aperçu global : il n’influence que le calcul. "
            "L’aperçu utilise désormais **toutes les images extraites**, donc augmente la cadence fixe ou change de mode si tu veux plus de vignettes.\n"
            "- En timelapse, préfère **Toutes les frames** + **pas d’analyse = 1**."
        )
