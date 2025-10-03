# pages/analyse_mouvements.py
# Analyse automatique et simplifiée des mouvements par flux optique.
# - AUCUN réglage requis : upload MP4 (ou vidéo préparée), bouton "Analyser".
# - Extraction d’images en 1080p (1920 de large) à 4 i/s via FFmpeg.
# - Calcul du flux optique (Farneback) entre images consécutives.
# - Baseline = moyenne globale des métriques.
# - Anomalies = pas (frames consécutives) dont le score composite de mouvement s’écarte fortement (z-score).
# - Sorties : résumé global, liste d’anomalies (top 16 avec vignettes), export CSV.

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
    """Import différé d'OpenCV (opencv-python-headless recommandé)."""
    try:
        import cv2  # type: ignore
        return cv2, None
    except Exception as e:
        return None, f"OpenCV introuvable : {e}. Ajoute 'opencv-python-headless' dans requirements.txt."

# =============================
# Extraction image (FFmpeg)
# =============================

def extraire_frames_1080p(ffmpeg: str, video: Path, dossier: Path, fps_ech: int = 4) -> Tuple[bool, str]:
    """Extrait des images JPG en 1080p (largeur 1920) à fps_ech i/s."""
    if dossier.exists():
        try:
            shutil.rmtree(dossier)
        except Exception:
            pass
    dossier.mkdir(parents=True, exist_ok=True)
    motif = str(dossier / "frame_%06d.jpg")
    filtre = f"fps={fps_ech},scale=1920:-2"
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video), "-vf", filtre, "-q:v", "2", motif]
    return executer(cmd)

# =============================
# Chargement images
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
# Page Streamlit (ultra simple)
# =============================

BASE_DIR, REP_SORTIE, REP_TMP = initialiser_repertoires()

st.set_page_config(page_title="Analyse mouvements (auto)", layout="wide")
st.title("Analyse des mouvements (moyenne & anomalies automatiques)")
st.markdown("**www.codeandcortex.fr**")

st.markdown(
    "Cette page calcule automatiquement :\n"
    "- une **moyenne globale** des mouvements (sur toute la vidéo),\n"
    "- des **anomalies de mouvement** = passages qui s’écartent fortement de la moyenne.\n\n"
    "Principes : on estime le **flux optique** entre images successives (4 images/s, 1080p). "
    "Pour chaque pas, on calcule : magnitude moyenne, écart-type, P95, énergie, direction dominante, dispersion. "
    "On forme un **score composite** (magnitude moyenne + énergie, standardisées) et on marque en **anomalies** les pas dont le z-score est élevé."
)

ff = trouver_ffmpeg()
if not ff:
    st.error("FFmpeg introuvable. Fournis un binaire ./bin/ffmpeg ou vérifie /usr/bin/ffmpeg.")
    st.stop()

cv2, cv_err = importer_cv2()
if cv2 is None:
    st.error(cv_err)
    st.stop()

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

lancer = st.button("Analyser", type="primary")

if lancer:
    if video_path is None:
        st.error("Aucune source vidéo. Importez un MP4 ou sélectionnez la vidéo préparée.")
        st.stop()

    # 1) Extraction auto en 1080p @ 4 i/s
    frames_dir = (BASE_DIR / "frames_analysis" / video_path.stem).resolve()
    ok_ext, log_ext = extraire_frames_1080p(ff, video_path, frames_dir, fps_ech=4)
    if not ok_ext:
        st.error("Échec extraction des images avec FFmpeg.")
        with st.expander("Journal FFmpeg"):
            st.code(log_ext or "(vide)", language="bash")
        st.stop()

    # 2) Chargement images
    imgs_gray, imgs_rgb = charger_images_gris_et_rgb(cv2, frames_dir)
    if len(imgs_gray) < 2:
        st.error("Trop peu d’images extraites pour analyser.")
        st.stop()

    # 3) Flux optique et métriques par pas
    lignes = []
    for i in range(1, len(imgs_gray)):
        flow = farneback(cv2, imgs_gray[i-1], imgs_gray[i])
        met = metriques_par_pas(flow)
        lignes.append({
            "pas_index": i,
            "temps_s_approx": i / 4.0,
            **met
        })
    df = pd.DataFrame(lignes)

    # 4) Baseline globale (moyenne sur tous les pas)
    moyennes_globales = df[[
        "magnitude_moyenne", "magnitude_ecart_type", "magnitude_p95",
        "energie_mouvement", "direction_dominante_deg", "dispersion_direction"
    ]].mean(numeric_only=True).to_dict()

    # 5) Score composite + anomalies
    #    - Standardiser magnitude_moyenne et energie_mouvement
    zM, muM, sM = zscore(df["magnitude_moyenne"].to_numpy(dtype=np.float64))
    zE, muE, sE = zscore(df["energie_mouvement"].to_numpy(dtype=np.float64))
    score_composite = (zM + zE) / 2.0
    df["score_composite_z"] = score_composite

    # Seuil auto : z >= 2.5 (déviation forte) ; ajusté automatiquement sans saisie utilisateur
    seuil_z = 2.5
    df["anomalie"] = df["score_composite_z"] >= seuil_z

    # 6) Résumés visibles simples
    st.subheader("Moyennes globales (baseline)")
    st.dataframe(pd.DataFrame([moyennes_globales]).T.rename(columns={0: "valeur"}))

    st.subheader("Anomalies détectées")
    nb_ano = int(df["anomalie"].sum())
    st.write(f"Nombre d’anomalies : {nb_ano} (seuil z ≥ {seuil_z:.1f})")
    if nb_ano == 0:
        st.info("Aucune anomalie forte détectée sur cette vidéo.")
    else:
        # Top 16 anomalies (z le plus élevé)
        top_idx = df.sort_values("score_composite_z", ascending=False).head(16)["pas_index"].tolist()
        cols_par_ligne = 8
        k = 0
        for _ in range(math.ceil(len(top_idx) / cols_par_ligne)):
            cols = st.columns(cols_par_ligne)
            for c in cols:
                if k >= len(top_idx):
                    break
                idx = int(top_idx[k])
                # Afficher la frame cible (idx) comme vignette
                if 0 <= idx < len(imgs_rgb):
                    c.image(imgs_rgb[idx], caption=f"#{idx} • z={df.loc[df['pas_index']==idx,'score_composite_z'].values[0]:.2f}", use_container_width=False)
                k += 1

    # 7) Export CSV (indices par pas + marquage anomalies)
    st.subheader("Téléchargement")
    st.download_button(
        "Télécharger les indices & anomalies (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="indices_mouvement_et_anomalies.csv",
        mime="text/csv"
    )

    # 8) Aperçu global réparti (vignettes) — simple, sans réglage
    st.subheader("Aperçu global (vignettes réparties)")
    N = len(imgs_rgb)
    nb_vignettes = min(48, N)
    idxs = np.linspace(0, N - 1, num=nb_vignettes, dtype=int)
    cols_par_ligne = 8
    k = 0
    for _ in range(math.ceil(len(idxs) / cols_par_ligne)):
        cols = st.columns(cols_par_ligne)
        for c in cols:
            if k >= len(idxs):
                break
            i = int(idxs[k])
            z_here = df.loc[df["pas_index"] == i, "score_composite_z"]
            cap = f"#{i}" + (f" • z={float(z_here.values[0]):.2f}" if len(z_here) else "")
            c.image(imgs_rgb[i], caption=cap, use_container_width=False)
            k += 1

    # 9) Explications succinctes affichées
    with st.expander("Explications (que calcule-t-on ?)"):
        st.markdown(
            "- **Flux optique** : champ de vecteurs (dx,dy) décrivant le déplacement des pixels entre deux images successives.\n"
            "- **Magnitude moyenne** : intensité moyenne du mouvement au pas considéré.\n"
            "- **Énergie du mouvement** : somme des magnitudes (poids global du mouvement).\n"
            "- **Score composite z** : moyenne des z-scores de la magnitude moyenne et de l’énergie. "
            "Un z élevé signifie « beaucoup plus de mouvement qu’à l’habitude » → **anomalie**.\n"
            "- **Direction dominante & dispersion** : orientation moyenne et stabilité des directions (pistes d’interprétation)."
        )
