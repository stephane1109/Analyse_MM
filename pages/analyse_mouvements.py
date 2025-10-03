# pages/analyse_mouvements.py
# Analyse des mouvements complète avec :
# - Extraction FFmpeg 1080p (frames natives timelapse OU cadence fixe)
# - Choix du PAS D’ANALYSE (intervalle entre frames analysées)
# - Affichage robuste via PIL (Pillow)
# - Calculs flux optique (OpenCV Farneback) et MÉTRIQUES par pas
# - Score composite standardisé et ANOMALIES encadrées en rouge
# - Aperçu global sur TOUTES les images extraites
# - Tableaux, graphiques, téléchargements, explications détaillées

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
# Chargement et affichage images (PIL)
# =============================

def charger_images_pil(dossier: Path) -> Tuple[List[np.ndarray], List[np.ndarray], List[Path]]:
    """
    Charge les images via PIL pour fiabiliser l'affichage Streamlit.
    - rgbs_pil : liste d'images RGB (numpy uint8) pour affichage
    - grays    : liste d'images grises (numpy uint8) pour calculs
    - chemins  : fichiers correspondants
    """
    from PIL import Image
    fichiers = sorted(dossier.glob("frame_*.jpg"))
    rgbs, grays = [], []
    for f in fichiers:
        try:
            with Image.open(f) as im:
                im = im.convert("RGB")
                rgb = np.array(im)  # H x W x 3, uint8
                gray = np.mean(rgb, axis=2).astype(np.uint8)  # luminance simple pour robustesse
                rgbs.append(rgb)
                grays.append(gray)
        except Exception:
            continue
    return rgbs, grays, fichiers

def encadrer_rouge_pil(img_rgb: np.ndarray, epaisseur: int = 8) -> np.ndarray:
    """Encadre une image RGB avec un cadre rouge (PIL-like, mais en numpy direct)."""
    vis = img_rgb.copy()
    h, w = vis.shape[:2]
    e = max(1, int(epaisseur))
    vis[:e, :, :] = [255, 0, 0]
    vis[-e:, :, :] = [255, 0, 0]
    vis[:, :e, :] = [255, 0, 0]
    vis[:, -e:, :] = [255, 0, 0]
    return vis

# =============================
# Flux optique + métriques (OpenCV)
# =============================

def farneback(cv2, prev_gray: np.ndarray, gray: np.ndarray) -> Optional[np.ndarray]:
    """Flux optique dense Farneback (retourne (H,W,2) ou None en cas d'échec local)."""
    try:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )
        return flow
    except Exception:
        return None

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
    "Principe général : estimation du **flux optique** entre deux images successives extraites en 1080p. "
    "Extraction possible en **frames natives** (timelapse) ou à **cadence fixe**. "
    "Pour chaque pas, l’application calcule la **magnitude moyenne**, l’**écart-type**, le **P95**, l’**énergie**, la **direction dominante** et la **dispersion**. "
    "On forme un **score composite** en combinant les z-scores de la magnitude moyenne et de l’énergie. "
    "Les pas dont ce score dépasse 2.5 écart-types sont marqués comme **anomalies** et leurs vignettes sont **encadrées en rouge**."
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
    "Pas d’analyse = intervalle entre frames analysées (1, 2, 3…). "
    "En timelapse, choisis 1. Si tu vois peu de vignettes, passe en « frames natives » ou augmente la cadence fixe."
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

    # Chargement et APERÇU IMMÉDIAT (PIL)
    imgs_rgb, imgs_gray, fichiers = charger_images_pil(frames_dir)
    total_frames = len(imgs_rgb)
    if total_frames < 2:
        st.error("Trop peu d’images extraites pour analyser (ou échec de lecture des JPG).")
        with st.expander("Diagnostic extraction"):
            st.write(f"Dossier : {frames_dir}")
            st.write(f"Fichiers JPG détectés : {len(list(frames_dir.glob('frame_*.jpg')))}")
            st.code(log_ext or "(vide)", language="bash")
        st.stop()

    st.info(f"Images extraites : {total_frames} ({'frames natives' if mode=='natifs' else f'cadence fixe = {int(fps_ech)} i/s'}). "
            f"Si ce nombre est trop faible, choisis « frames natives » ou augmente la cadence fixe.")

    st.subheader("Aperçu immédiat des images extraites")
    Nprev = len(imgs_rgb)
    nb_prev = min(24, Nprev)
    idx_prev = np.linspace(0, Nprev - 1, num=nb_prev, dtype=int)
    cols = st.columns(6)
    for i, idx in enumerate(idx_prev):
        cols[i % 6].image(imgs_rgb[int(idx)], caption=f"frame #{int(idx)}", use_container_width=False)

    # Construction des indices pour le PAS D’ANALYSE
    pas = int(pas_analyse)
    indices = list(range(0, total_frames, pas))
    if len(indices) < 2:
        st.warning("Pas d’analyse trop grand pour la longueur de la séquence. Utilisation automatique de pas=1.")
        pas = 1
        indices = list(range(0, total_frames, pas))

    # Calcul du flux optique et des métriques entre frames espacées par 'pas'
    lignes: List[Dict[str, float]] = []
    echecs_pairs = 0

    for k in range(1, len(indices)):
        i_prev = indices[k-1]
        i_curr = indices[k]
        flow = farneback(cv2, imgs_gray[i_prev], imgs_gray[i_curr])
        if flow is None:
            echecs_pairs += 1
            continue
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

    if not lignes:
        st.error("Aucune paire valide pour le flux optique. Essaie pas=1 et « frames natives ».")
        st.stop()

    if echecs_pairs > 0:
        st.warning(f"{echecs_pairs} paire(s) ont échoué au calcul du flux optique et ont été ignorées.")

    df = pd.DataFrame(lignes)

    # =========================
    # RÉSUMÉS, SCORES, ANOMALIES
    # =========================

    # Baseline globale (moyennes sur tous les pas)
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

    st.subheader("Moyennes globales (baseline)")
    st.dataframe(pd.DataFrame([moyennes_globales]).T.rename(columns={0: "valeur"}))

    st.subheader("Scores et anomalies")
    st.write(f"Seuil d’anomalie : z ≥ {seuil_z:.1f}")
    st.line_chart(df.set_index("etape")[["magnitude_moyenne", "energie_mouvement", "score_composite_z"]])

    nb_ano = int(df["anomalie"].sum())
    st.write(f"Nombre d’anomalies détectées : {nb_ano}")

    st.subheader("Vignettes des anomalies (encadrées en rouge)")
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
                    vis = encadrer_rouge_pil(imgs_rgb[idx], epaisseur=8)
                    c.image(vis, caption=f"frame #{idx} • z={z_here:.2f}", use_container_width=False)
                k += 1
    else:
        st.info("Aucune anomalie forte détectée.")

    # =========================
    # TABLEAUX et EXPORTS
    # =========================

    st.subheader("Tableau des indices par pas")
    st.dataframe(df)

    st.subheader("Téléchargements (CSV)")
    st.download_button(
        "Indices et anomalies par pas",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="indices_mouvement_et_anomalies.csv",
        mime="text/csv"
    )

    # =========================
    # APERÇU GLOBAL sur TOUTES les images extraites
    # =========================

    st.subheader("Aperçu global (vignettes réparties, anomalies encadrées)")
    N = len(imgs_rgb)
    nb_vignettes = min(96, N)
    idxs = np.linspace(0, N - 1, num=nb_vignettes, dtype=int)

    frames_anormales = set(df[df["anomalie"]]["frame_curr"].astype(int).tolist())
    z_map = {int(r["frame_curr"]): float(r["score_composite_z"]) for _, r in df.iterrows()}

    cols_par_ligne = 8
    k = 0
    for _ in range(math.ceil(len(idxs) / cols_par_ligne)):
        cols = st.columns(cols_par_ligne)
        for c in cols:
            if k >= len(idxs):
                break
            fr = int(idxs[k])
            img = encadrer_rouge_pil(imgs_rgb[fr], epaisseur=6) if fr in frames_anormales else imgs_rgb[fr]
            z_here = z_map.get(fr, None)
            cap = f"frame #{fr}" + (f" • z={z_here:.2f}" if z_here is not None else "")
            c.image(img, caption=cap, use_container_width=False)
            k += 1

    # =========================
    # EXPLICATIONS DÉTAILLÉES
    # =========================

    st.subheader("Explications détaillées")
    with st.expander("Pas d’analyse (intervalle entre frames analysées)"):
        st.markdown(
            "Le pas d’analyse détermine l’intervalle entre les images utilisées pour estimer le flux optique. "
            "Un pas de 1 signifie que chaque image extraite est utilisée, ce qui est idéal pour un timelapse où l’espacement réel est déjà important. "
            "Augmenter le pas (2, 3, …) accélère l’analyse en sautant des images, mais accentue les écarts entre deux états de la scène. "
            "Un pas trop grand peut rater des micro-événements situés entre deux images retenues. "
            "La règle pratique est de garder 1 en timelapse et d’ajuster seulement si le volume d’images est très élevé."
        )
    with st.expander("Mode d’extraction (frames natives vs cadence fixe)"):
        st.markdown(
            "Le mode « frames natives » conserve chaque image distincte fournie par la vidéo, ce qui est adapté aux timelapses et aux cadences variables. "
            "Le mode « cadence fixe » force un échantillonnage régulier en images par seconde pour uniformiser la résolution temporelle. "
            "En cadence fixe trop basse, des événements courts peuvent être lissés ; en cadence trop haute, le volume d’images explose. "
            "Si tu vois peu de vignettes, choisis « frames natives » ou augmente la cadence fixe. "
            "Dans tous les cas, les images sont redimensionnées en 1080p pour des calculs cohérents."
        )
    with st.expander("Indices de mouvement (magnitude, écart-type, P95, énergie)"):
        st.markdown(
            "La magnitude moyenne quantifie l’intensité moyenne du déplacement des pixels entre deux images. "
            "L’écart-type décrit la variabilité interne des vitesses de mouvement : élevé = hétérogène, faible = plus homogène. "
            "Le 95e percentile (P95) met en évidence les parties les plus rapides du mouvement sans être dominé par des outliers. "
            "L’énergie du mouvement est la somme des magnitudes et favorise les événements qui couvrent une grande surface de l’image. "
            "Pris ensemble, ces indicateurs permettent de caractériser des passages calmes, dynamiques, homogènes ou très contrastés."
        )
    with st.expander("Direction dominante et dispersion"):
        st.markdown(
            "La direction dominante est l’angle moyen des vecteurs du flux optique, calculé en statistiques circulaires. "
            "Elle signale une orientation préférentielle des mouvements (ex. déplacement global, geste orienté). "
            "La dispersion mesure la cohérence des directions : faible dispersion = directions alignées ; forte dispersion = agitation, désordre. "
            "Ces deux mesures sont utiles pour qualifier la nature des gestes, au-delà de leur simple intensité. "
            "Elles complètent les mesures de magnitude pour une lecture plus riche du comportement corporel."
        )
    with st.expander("Score composite et anomalies (z-score)"):
        st.markdown(
            "Le score composite combine la magnitude moyenne et l’énergie du mouvement après standardisation (z-score). "
            "Standardiser signifie recentrer par la moyenne et mettre à l’échelle par l’écart-type afin de rendre comparables les unités. "
            "On prend ensuite la moyenne des deux z-scores pour capturer à la fois l’intensité typique et le volume global du mouvement. "
            "Un pas est dit « anomalie » si ce score dépasse 2.5 écart-types, ce qui isole des déviations marquées de la norme. "
            "Les vignettes correspondantes sont encadrées en rouge pour un repérage visuel immédiat."
        )
