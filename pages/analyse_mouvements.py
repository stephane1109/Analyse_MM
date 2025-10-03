# pages/analyse_mouvements.py
# Analyse d'anomalies visuelles par différence à l'image moyenne, avec explications intégrées.
# Source prioritaire : MP4 importé sur cette page. Fallback : vidéo préparée (st.session_state["video_base"]).
# Extraction d'images en 1080p via FFmpeg, analyse à cadence réglable (4 i/s par défaut).
# Détection : score MAE par frame (écart moyen absolu à l'image moyenne), standardisé en z-score.
# Anomalie si z-score >= seuil (sensibilité choisie).

import math
import shutil
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from core_media import initialiser_repertoires, info_ffmpeg

# ----------------------------
# Utilitaires système
# ----------------------------

def _ffmpeg_path() -> Optional[str]:
    """Retourne le chemin de ffmpeg si disponible, sinon None."""
    p, _ = info_ffmpeg()
    return p

def _run(cmd: List[str]) -> Tuple[bool, str]:
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

def _load_cv2():
    """Import différé d'OpenCV (opencv-python-headless recommandé)."""
    try:
        import cv2  # type: ignore
        return cv2, None
    except Exception as e:
        return None, f"OpenCV introuvable : {e}. Ajoute 'opencv-python-headless' dans requirements.txt."

# ----------------------------
# Extraction d'images
# ----------------------------

def extraire_frames_ffmpeg(ff: str, video: Path, dossier: Path, fps_ech: int, largeur: int = 1920) -> Tuple[bool, str]:
    """Extrait des images JPG uniformément, en 1080p (largeur 1920), à fps_ech images/s."""
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

# ----------------------------
# Chargement des images
# ----------------------------

def charger_images_gris_et_rgb(cv2, dossier: Path) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Charge toutes les images JPG en niveaux de gris et en RGB pour affichage."""
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

# ----------------------------
# Anomalie = différence à l'image moyenne
# ----------------------------

def calculer_image_moyenne(imgs_gray: List[np.ndarray]) -> np.ndarray:
    """Calcule l'image moyenne (float32) sur toutes les frames grises."""
    acc = None
    n = 0
    for g in imgs_gray:
        g32 = g.astype(np.float32)
        if acc is None:
            acc = g32
        else:
            acc += g32
        n += 1
    if acc is None or n == 0:
        raise ValueError("Aucune image pour calculer la moyenne.")
    return acc / float(n)

def score_mae_par_frame(imgs_gray: List[np.ndarray], mean_img: np.ndarray) -> np.ndarray:
    """Retourne, pour chaque frame, le MAE à l'image moyenne (moyenne des |diff| par pixel)."""
    scores = []
    for g in imgs_gray:
        diff = np.abs(g.astype(np.float32) - mean_img)
        mae = float(np.mean(diff))
        scores.append(mae)
    return np.array(scores, dtype=np.float32)

def zscore(scores: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Standardise les scores : z = (score - moyenne) / écart-type. Retourne (z, moyenne, std)."""
    m = float(np.mean(scores))
    s = float(np.std(scores)) if float(np.std(scores)) > 1e-12 else 1.0
    z = (scores - m) / s
    return z, m, s

# ----------------------------
# Page Streamlit
# ----------------------------

BASE_DIR, REP_SORTIE, REP_TMP = initialiser_repertoires()

st.set_page_config(page_title="Analyse d'anomalies (différence à la moyenne)", layout="wide")
st.title("Analyse d'anomalies (différence à la moyenne)")
st.markdown("**www.codeandcortex.fr**")

# Explications intégrées
st.markdown(
    "Cette analyse détecte des images « atypiques » en comparant chaque image à l'image moyenne "
    "calculée sur toute la séquence. Pour chaque image i, on mesure l'écart moyen absolu (MAE) entre i et la moyenne. "
    "Ces scores sont ensuite standardisés en z-score. Une image est dite « anormale » si son z-score dépasse un seuil."
)

with st.expander("Pourquoi 4 images/seconde par défaut ?"):
    st.markdown(
        "Le choix par défaut de 4 images/s équilibre vitesse et pertinence : assez d'images pour repérer des changements "
        "visibles, sans saturer la mémoire ni ralentir l'application. Tu peux augmenter si la vidéo change très vite, "
        "ou diminuer si elle est longue et assez statique."
    )

with st.expander("Définitions affichées dans la page"):
    st.markdown(
        "MAE (Mean Absolute Error) : moyenne des valeurs absolues de la différence pixel à pixel entre une image et "
        "l'image moyenne. Plus le MAE est grand, plus l'image s'éloigne du contenu moyen.\n\n"
        "z-score : normalisation des scores pour les rendre comparables. "
        "z = (score − moyenne_des_scores) / écart_type_des_scores. "
        "Un z-score élevé signifie que la frame est très différente des autres."
    )

ff = _ffmpeg_path()
if not ff:
    st.error("FFmpeg introuvable. Fournis un binaire ./bin/ffmpeg ou vérifie /usr/bin/ffmpeg.")
    st.stop()

cv2, cv_err = _load_cv2()
if cv2 is None:
    st.error(cv_err)
    st.stop()

# Source : upload prioritaire, sinon vidéo préparée
st.subheader("Source vidéo pour l'analyse")
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

# Paramètres simplifiés
st.subheader("Paramètres")
c1, c2, c3 = st.columns(3)
with c1:
    fps_ech = st.number_input("Cadence d'échantillonnage (images/s)", min_value=1, max_value=30, value=4, step=1)
with c2:
    sensibilite = st.selectbox("Sensibilité des anomalies", ["Faible", "Normale", "Forte"], index=1)
with c3:
    nb_vignettes = st.number_input("Nombre de vignettes globales", min_value=12, max_value=200, value=48, step=12)

seuils = {"Forte": 2.0, "Normale": 2.5, "Faible": 3.0}
seuil_z = seuils[sensibilite]

montrer_log = st.checkbox("Afficher le journal FFmpeg", value=False)
lancer = st.button("Lancer l'analyse", type="primary")

if lancer:
    if video_path is None:
        st.error("Aucune source vidéo. Importez un MP4 ou sélectionnez la vidéo préparée.")
        st.stop()

    # 1) Extraction d'images en 1080p à fps_ech
    frames_dir = (BASE_DIR / "frames_analysis" / video_path.stem).resolve()
    ok_ext, log_ext = extraire_frames_ffmpeg(ff, video_path, frames_dir, int(fps_ech), largeur=1920)
    if not ok_ext:
        st.error("Échec extraction des images avec FFmpeg.")
        if montrer_log:
            st.code(log_ext or "(log vide)", language="bash")
        st.stop()

    # 2) Chargement
    imgs_gray, imgs_rgb = charger_images_gris_et_rgb(cv2, frames_dir)
    if len(imgs_gray) < 2:
        st.error("Aucune image ou trop peu d'images extraites. Impossible d'analyser.")
        if montrer_log:
            st.code(log_ext or "(log vide)", language="bash")
        st.stop()

    # 3) Image moyenne et MAE par frame
    mean_img = calculer_image_moyenne(imgs_gray)
    scores_mae = score_mae_par_frame(imgs_gray, mean_img)
    z, m_mae, s_mae = zscore(scores_mae)

    # 4) Détection d'anomalies
    anomalies_idx = np.where(z >= seuil_z)[0].tolist()

    # 5) Résultats chiffrés et explications
    st.subheader("Résultats et définitions")
    st.markdown(
        f"MAE moyen : {m_mae:.2f}  |  Écart-type MAE : {s_mae:.2f}  |  Seuil d’anomalie z ≥ {seuil_z:.1f}\n\n"
        "Une frame est marquée « anomalie » si son z-score dépasse le seuil. "
        "Cela signifie qu’elle diffère fortement de l’image moyenne relativement aux autres frames."
    )
    st.write(f"Nombre d’images analysées : {len(imgs_gray)}  |  Anomalies détectées : {len(anomalies_idx)}")

    # 6) Export CSV
    lignes = []
    for i, (s, zi) in enumerate(zip(scores_mae, z)):
        lignes.append({
            "index_frame": i,
            "temps_s_approx": i / float(fps_ech),
            "mae_diff_moyenne": float(s),
            "zscore_mae": float(zi),
            "anomalie": bool(zi >= seuil_z),
        })
    df = pd.DataFrame(lignes)
    st.download_button(
        "Télécharger les scores (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="scores_anomalies_diff_moyenne.csv",
        mime="text/csv"
    )

    # 7) Vignettes anormales (triées)
    st.subheader("Vignettes anormales (si présentes)")
    if anomalies_idx:
        anomalies_sorted = sorted(anomalies_idx, key=lambda i: float(z[i]), reverse=True)
        to_show = anomalies_sorted[:32]
        cols_par_ligne = 8
        lignes_nb = math.ceil(len(to_show) / cols_par_ligne)
        k = 0
        for _ in range(lignes_nb):
            cols = st.columns(cols_par_ligne)
            for c in cols:
                if k >= len(to_show):
                    break
                i = int(to_show[k])
                c.image(imgs_rgb[i], caption=f"#{i} • z={z[i]:.2f}", use_container_width=False)
                k += 1
    else:
        st.info("Aucune frame n’a franchi le seuil d’anomalie pour la sensibilité choisie.")

    # 8) Aperçu global réparti
    st.subheader("Aperçu global (vignettes réparties sur la vidéo)")
    N = len(imgs_rgb)
    idxs = np.linspace(0, N - 1, num=int(nb_vignettes), dtype=int)
    cols_par_ligne = 8
    lignes_nb = math.ceil(len(idxs) / cols_par_ligne)
    k = 0
    for _ in range(lignes_nb):
        cols = st.columns(cols_par_ligne)
        for c in cols:
            if k >= len(idxs):
                break
            i = int(idxs[k])
            c.image(imgs_rgb[i], caption=f"#{i} • z={z[i]:.2f}", use_container_width=False)
            k += 1

    # 9) Journal FFmpeg optionnel
    if montrer_log:
        with st.expander("Journal FFmpeg"):
            st.code(log_ext or "(log vide)", language="bash")
