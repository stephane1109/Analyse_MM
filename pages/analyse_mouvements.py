# pages/analyse_mouvements.py
# Analyse visuelle avec deux modes :
# 1) MAE (différence à l'image moyenne) pour anomalies globales de contenu.
# 2) Flux optique (magnitude Farneback) pour indices/mesures de mouvement + heatmap + vecteurs.
# Source prioritaire : MP4 uploadé. Fallback : vidéo préparée (st.session_state["video_base"]).
# Extraction d'images en 1080p via FFmpeg, puis calculs avec OpenCV.

import math
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

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
# Extraction d'images via FFmpeg
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
# Mode 1 : MAE (différence à l'image moyenne)
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
# Mode 2 : Flux optique (Farneback)
# ----------------------------

def farneback_flow(cv2, prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Calcule le flux optique dense Farneback, retourne un tableau (H, W, 2) de vecteurs (dx, dy)."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray,
        None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0
    )
    return flow

def metriques_flux_optique(flow: np.ndarray, seuil_pix: float) -> dict:
    """Calcule les indices/métriques à partir du flux (magnitude)."""
    mag = np.linalg.norm(flow, axis=2)
    m = float(np.mean(mag))
    s = float(np.std(mag))
    p95 = float(np.percentile(mag, 95))
    ratio_mobile = float(np.mean(mag > seuil_pix))
    energie = float(np.sum(mag))
    return {
        "magnitude_moyenne": m,
        "magnitude_ecart_type": s,
        "magnitude_p95": p95,
        "ratio_pixels_mobiles": ratio_mobile,
        "energie_mouvement": energie,
    }

def heatmap_depuis_magnitude(cv2, mag: np.ndarray) -> np.ndarray:
    """Crée une heatmap RGB à partir d'une magnitude (palette INFERNO, normalisation robuste)."""
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

def vecteurs_sur_image(cv2, frame_rgb: np.ndarray, flow: np.ndarray, pas: int = 16) -> np.ndarray:
    """Dessine un champ de vecteurs échantillonné sur l'image RGB."""
    h, w = frame_rgb.shape[:2]
    vis = frame_rgb.copy()
    for y in range(0, h, pas):
        for x in range(0, w, pas):
            fx, fy = flow[y, x]
            x2 = int(round(x + fx))
            y2 = int(round(y + fy))
            cv2.arrowedLine(vis, (x, y), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
    return vis

# ----------------------------
# Page Streamlit
# ----------------------------

BASE_DIR, REP_SORTIE, REP_TMP = initialiser_repertoires()

st.set_page_config(page_title="Analyse des mouvements et anomalies", layout="wide")
st.title("Analyse des mouvements et anomalies")
st.markdown("**www.codeandcortex.fr**")

# Explications intégrées
st.markdown(
    "Deux modes complémentaires :\n"
    "• **Différence à la moyenne (MAE)** : détecte des images globalement atypiques (flash, écran noir, scène très différente).\n"
    "• **Flux optique (magnitude)** : mesure les déplacements entre images consécutives et fournit des **indices de mouvement** "
    "(magnitude moyenne, écart-type, P95, ratio de pixels mobiles, énergie). Heatmap et vecteurs permettent de visualiser ce mouvement."
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

# Paramètres
st.subheader("Paramètres")
c0, c1, c2, c3 = st.columns(4)
with c0:
    mode = st.radio("Mode d'analyse", ["Différence à la moyenne (MAE)", "Flux optique (magnitude)"], index=1)
with c1:
    fps_ech = st.number_input("Cadence extraction (images/s)", min_value=1, max_value=30, value=4, step=1)
with c2:
    nb_vignettes = st.number_input("Vignettes globales", min_value=12, max_value=200, value=48, step=12)
with c3:
    montrer_log = st.checkbox("Afficher le journal FFmpeg", value=False)

c4, c5 = st.columns(2)
if mode.startswith("Différence"):
    with c4:
        sensibilite = st.selectbox("Sensibilité anomalies (seuil z-score)", ["Forte (z≥2.0)", "Normale (z≥2.5)", "Faible (z≥3.0)"], index=1)
    seuil_z = 2.0 if sensibilite.startswith("Forte") else (2.5 if sensibilite.startswith("Normale") else 3.0)
else:
    with c4:
        seuil_pix = st.number_input("Seuil pixel « mobile » (px/frame)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
    with c5:
        top_k = st.number_input("Nombre de frames les plus mobiles à illustrer", min_value=6, max_value=60, value=24, step=6)

lancer = st.button("Lancer l'analyse", type="primary")

if lancer:
    if video_path is None:
        st.error("Aucune source vidéo. Importez un MP4 ou sélectionnez la vidéo préparée.")
        st.stop()

    # 1) Extraction d'images en 1080p
    frames_dir = (BASE_DIR / "frames_analysis" / video_path.stem).resolve()
    ok_ext, log_ext = extraire_frames_ffmpeg(ff, video_path, frames_dir, int(fps_ech), largeur=1920)
    if not ok_ext:
        st.error("Échec extraction des images avec FFmpeg.")
        if montrer_log:
            st.code(log_ext or "(log vide)", language="bash")
        st.stop()

    # 2) Chargement des images
    imgs_gray, imgs_rgb = charger_images_gris_et_rgb(cv2, frames_dir)
    if len(imgs_gray) < 2:
        st.error("Aucune image ou trop peu d'images extraites. Impossible d'analyser.")
        if montrer_log:
            st.code(log_ext or "(log vide)", language="bash")
        st.stop()

    # 3) Mode MAE
    if mode.startswith("Différence"):
        mean_img = calculer_image_moyenne(imgs_gray)
        scores_mae = score_mae_par_frame(imgs_gray, mean_img)
        z, m_mae, s_mae = zscore(scores_mae)
        anomalies_idx = np.where(z >= seuil_z)[0].tolist()

        st.subheader("Résultats (MAE vs image moyenne)")
        st.markdown(
            f"MAE moyen : {m_mae:.2f}  |  Écart-type MAE : {s_mae:.2f}  |  Seuil d’anomalie z ≥ {seuil_z:.1f}\n\n"
            "MAE (Mean Absolute Error) = moyenne des valeurs absolues de la différence pixel-à-pixel "
            "entre l’image i et l’image moyenne de la séquence.\n"
            "z-score = (MAE − moyenne_des_MAE) / écart-type_des_MAE. Un z élevé = image très atypique."
        )
        st.write(f"Images analysées : {len(imgs_gray)}  |  Anomalies détectées : {len(anomalies_idx)}")

        # Export CSV
        df = pd.DataFrame({
            "index_frame": np.arange(len(scores_mae)),
            "temps_s_approx": np.arange(len(scores_mae)) / float(fps_ech),
            "mae_diff_moyenne": scores_mae,
            "zscore_mae": z,
            "anomalie": (z >= seuil_z),
        })
        st.download_button("Télécharger les scores (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="scores_anomalies_mae.csv", mime="text/csv")

        # Vignettes anormales triées
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
            st.info("Aucune frame n’a dépassé le seuil d’anomalie.")

    # 4) Mode Flux optique
    else:
        st.subheader("Résultats (flux optique Farneback)")
        # Calcul des métriques par pas
        metriques_liste = []
        energies = []
        mags_for_heat = []   # stocker quelques magnitudes pour heatmap
        flows_for_vec = []   # stocker quelques flows pour vecteurs (échantillonnage léger)

        for i in range(1, len(imgs_gray)):
            flow = farneback_flow(cv2, imgs_gray[i-1], imgs_gray[i])
            met = metriques_flux_optique(flow, float(seuil_pix))
            metriques_liste.append(met)
            energies.append(met["energie_mouvement"])
            if i % max(1, len(imgs_gray)//min(len(imgs_gray), 60)) == 0:
                mag = np.linalg.norm(flow, axis=2)
                mags_for_heat.append((i, mag))
                flows_for_vec.append((i, flow))

        energies = np.array(energies, dtype=float)
        mE = float(np.mean(energies))
        sE = float(np.std(energies))
        p95E = float(np.percentile(energies, 95)) if len(energies) else 0.0

        st.markdown(
            "Définitions des indices de mouvement :\n"
            "- **Magnitude moyenne** : intensité moyenne du déplacement par pixel entre deux images.\n"
            "- **Écart-type de la magnitude** : variabilité du mouvement.\n"
            "- **P95 de la magnitude** : niveau de « pic » de mouvement.\n"
            "- **Ratio de pixels mobiles** : part des pixels dont la magnitude dépasse le seuil choisi.\n"
            "- **Énergie du mouvement** : somme des magnitudes (mesure globale par pas)."
        )
        st.write(f"Énergie moyenne : {mE:.2f}  |  Écart-type énergie : {sE:.2f}  |  P95 énergie : {p95E:.2f}")

        # Export CSV des métriques par pas
        df = pd.DataFrame(metriques_liste)
        df.insert(0, "pas_index", np.arange(1, len(metriques_liste)+1))
        df.insert(1, "temps_s_approx", df["pas_index"] / float(fps_ech))
        st.download_button("Télécharger les indices (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="indices_flux_optique.csv", mime="text/csv")

        # Sélection des frames les plus « mobiles » par énergie
        if len(energies):
            ordre = np.argsort(-energies)  # décroissant
            top = ordre[:int(top_k)]
            st.subheader("Vignettes des frames les plus mobiles (heatmap et vecteurs)")
            cols_par_ligne = 6
            k = 0
            # Heatmaps
            st.markdown("Heatmap de la magnitude (couleurs proportionnelles à l'intensité du mouvement)")
            for _ in range(math.ceil(len(top) / cols_par_ligne)):
                cols = st.columns(cols_par_ligne)
                for c in cols:
                    if k >= len(top):
                        break
                    idx = int(top[k])
                    # flow/mag correspond au pas entre frame idx-1 -> idx
                    # on affiche la frame idx en fond
                    flow = farneback_flow(cv2, imgs_gray[idx-1], imgs_gray[idx])
                    mag = np.linalg.norm(flow, axis=2)
                    hm = heatmap_depuis_magnitude(cv2, mag)
                    c.image(hm, caption=f"pas #{idx} • énergie={energies[idx-1]:.0f}", use_container_width=False)
                    k += 1

            # Vecteurs
            st.markdown("Champ de vecteurs (échantillonné)")
            k = 0
            for _ in range(math.ceil(len(top) / cols_par_ligne)):
                cols = st.columns(cols_par_ligne)
                for c in cols:
                    if k >= len(top):
                        break
                    idx = int(top[k])
                    flow = farneback_flow(cv2, imgs_gray[idx-1], imgs_gray[idx])
                    vis = vecteurs_sur_image(cv2, imgs_rgb[idx], flow, pas=16)
                    c.image(vis, caption=f"pas #{idx} • énergie={energies[idx-1]:.0f}", use_container_width=False)
                    k += 1
        else:
            st.info("Impossible de calculer l’ordre des frames les plus mobiles (énergie vide).")

    # 5) Aperçu global réparti (quel que soit le mode)
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
            c.image(imgs_rgb[i], caption=f"#{i}", use_container_width=False)
            k += 1

    # 6) Journal FFmpeg optionnel
    if montrer_log:
        with st.expander("Journal FFmpeg"):
            st.code(log_ext or "(log vide)", language="bash")
