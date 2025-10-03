# pages/analyse_mouvements.py
# Analyse automatique des mouvements par flux optique avec intervalle optionnel et explications détaillées.
# Principe : extraction d’images en 1080p à 4 i/s (FFmpeg) sur toute la vidéo ou un intervalle,
# calcul du flux optique Farneback entre images consécutives, métriques par pas, baseline globale,
# score composite standardisé (magnitude moyenne + énergie) et détection d’anomalies par z-score.

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
    fps_ech: int = 4,
    debut_s: Optional[float] = None,
    fin_s: Optional[float] = None
) -> Tuple[bool, str]:
    """
    Extrait des images JPG en 1080p (largeur 1920) à fps_ech i/s.
    Si un intervalle [debut_s, fin_s] est fourni, on le passe à FFmpeg pour ne traiter que cette portion.
    Les fichiers sont nommés frame_%06d.jpg.
    """
    if dossier.exists():
        try:
            shutil.rmtree(dossier)
        except Exception:
            pass
    dossier.mkdir(parents=True, exist_ok=True)
    motif = str(dossier / "frame_%06d.jpg")
    filtre = f"fps={fps_ech},scale=1920:-2"

    # Construction de la commande avec éventuel intervalle.
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    if debut_s is not None and debut_s > 0:
        cmd += ["-ss", str(float(debut_s))]
    if fin_s is not None and fin_s > 0 and (debut_s is None or fin_s > debut_s):
        cmd += ["-to", str(float(fin_s))]
    cmd += ["-i", str(video), "-vf", filtre, "-q:v", "2", motif]

    return executer(cmd)

# =============================
# Chargement des images
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
# Page Streamlit
# =============================

BASE_DIR, REP_SORTIE, REP_TMP = initialiser_repertoires()

st.set_page_config(page_title="Analyse des mouvements (intervalle et anomalies)", layout="wide")
st.title("Analyse des mouvements (moyenne globale et anomalies)")
st.markdown("**www.codeandcortex.fr**")

# Explication concise des principes en tête de page.
st.markdown(
    "On estime le flux optique entre images successives extraites en 1080p à quatre images par seconde. "
    "Pour chaque pas temporel, on calcule la magnitude moyenne, l’écart-type, le 95e percentile, l’énergie de mouvement, "
    "ainsi que la direction dominante et sa dispersion. On construit ensuite un score composite à partir de la magnitude "
    "moyenne et de l’énergie, standardisées par z-score pour les rendre comparables. Les pas dont le z-score composite est "
    "nettement supérieur à la moyenne globale sont marqués comme anomalies."
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

# Intervalle optionnel
st.subheader("Intervalle d’analyse (optionnel)")
activer_intervalle = st.checkbox("Analyser uniquement un intervalle de la vidéo", value=False)
debut_s = None
fin_s = None
if activer_intervalle:
    c1, c2 = st.columns(2)
    with c1:
        debut_s = st.number_input("Début (secondes)", min_value=0.0, value=0.0, step=0.5)
    with c2:
        fin_s = st.number_input("Fin (secondes)", min_value=0.5, value=10.0, step=0.5)
    if fin_s <= debut_s:
        st.warning("La fin doit être strictement supérieure au début.")

# Lancement
if st.button("Analyser", type="primary"):
    if video_path is None:
        st.error("Aucune source vidéo. Importez un MP4 ou sélectionnez la vidéo préparée.")
        st.stop()

    frames_dir = (BASE_DIR / "frames_analysis" / video_path.stem).resolve()
    ok_ext, log_ext = extraire_frames_1080p(ff, video_path, frames_dir, fps_ech=4, debut_s=debut_s, fin_s=fin_s)
    if not ok_ext:
        st.error("Échec extraction des images avec FFmpeg.")
        with st.expander("Journal FFmpeg"):
            st.code(log_ext or "(vide)", language="bash")
        st.stop()

    imgs_gray, imgs_rgb = charger_images_gris_et_rgb(cv2, frames_dir)
    if len(imgs_gray) < 2:
        st.error("Trop peu d’images extraites pour analyser.")
        with st.expander("Journal FFmpeg"):
            st.code(log_ext or "(vide)", language="bash")
        st.stop()

    # Calcul flux optique + métriques
    lignes: List[Dict[str, float]] = []
    for i in range(1, len(imgs_gray)):
        flow = farneback(cv2, imgs_gray[i-1], imgs_gray[i])
        met = metriques_par_pas(flow)
        lignes.append({
            "pas_index": i,
            "temps_s_approx": i / 4.0,
            **met
        })
    df = pd.DataFrame(lignes)

    # Baseline globale
    moyennes_globales = df[[
        "magnitude_moyenne", "magnitude_ecart_type", "magnitude_p95",
        "energie_mouvement", "direction_dominante_deg", "dispersion_direction"
    ]].mean(numeric_only=True).to_dict()

    # Score composite et anomalies (seuil fixe, clair et robuste)
    zM, _, _ = zscore(df["magnitude_moyenne"].to_numpy(dtype=np.float64))
    zE, _, _ = zscore(df["energie_mouvement"].to_numpy(dtype=np.float64))
    df["score_composite_z"] = (zM + zE) / 2.0
    seuil_z = 2.5
    df["anomalie"] = df["score_composite_z"] >= seuil_z

    # Affichages essentiels
    st.subheader("Moyennes globales (baseline)")
    st.dataframe(pd.DataFrame([moyennes_globales]).T.rename(columns={0: "valeur"}))

    st.subheader("Anomalies détectées")
    nb_ano = int(df["anomalie"].sum())
    st.write(f"Nombre d’anomalies détectées : {nb_ano} (seuil z ≥ {seuil_z:.1f})")
    if nb_ano > 0:
        top_idx = df.sort_values("score_composite_z", ascending=False).head(16)["pas_index"].tolist()
        cols_par_ligne = 8
        k = 0
        for _ in range(math.ceil(len(top_idx) / cols_par_ligne)):
            cols = st.columns(cols_par_ligne)
            for c in cols:
                if k >= len(top_idx):
                    break
                idx = int(top_idx[k])
                if 0 <= idx < len(imgs_rgb):
                    z_here = df.loc[df["pas_index"] == idx, "score_composite_z"].values[0]
                    c.image(imgs_rgb[idx], caption=f"#{idx} • z={z_here:.2f}", use_container_width=False)
                k += 1
    else:
        st.info("Aucune anomalie forte détectée sur la période analysée.")

    st.subheader("Téléchargement des indices")
    st.download_button(
        "Télécharger les indices et anomalies (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="indices_mouvement_et_anomalies.csv",
        mime="text/csv"
    )

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

    # Explications détaillées (minimum 5 phrases par paramètre)
    st.subheader("Explications détaillées des paramètres et du test")
    st.markdown(
        "La magnitude moyenne mesure l’intensité moyenne des déplacements estimés par le flux optique entre deux images consécutives. "
        "Elle agrège les modules des vecteurs de mouvement sur l’ensemble des pixels, puis en prend la moyenne pour obtenir une valeur simple et robuste. "
        "Lorsque cette magnitude est élevée sur un pas, cela signifie qu’une part importante de l’image a changé de position de manière notable. "
        "Cette mesure est utile pour comparer l’activité visible d’un passage par rapport au comportement général de la vidéo. "
        "Dans le cadre d’un timelapse où les mouvements sont saccadés, la magnitude moyenne reste pertinente car elle résume la variation apparente même si les intervalles temporels ne sont pas réguliers."
    )
    st.markdown(
        "L’écart-type de la magnitude décrit la variabilité des vitesses de mouvement à l’intérieur d’un même pas. "
        "Une valeur élevée indique que certaines zones se déplacent fortement tandis que d’autres bougent peu, ce qui traduit une hétérogénéité des gestes ou de la scène. "
        "À l’inverse, un écart-type faible signale des mouvements plus homogènes, souvent associés à une action coordonnée ou à une stabilité visuelle. "
        "Cette mesure complète naturellement la magnitude moyenne en révélant la dispersion interne des vitesses. "
        "Elle est utile pour distinguer des pics localisés de mouvement de véritables changements globaux dans l’image."
    )
    st.markdown(
        "Le 95e percentile (P95) de la magnitude représente un seuil au-delà duquel se situent les vitesses les plus élevées observées sur un pas. "
        "Il est moins sensible aux valeurs extrêmes isolées qu’un maximum brut, tout en capturant l’existence de mouvements rapides. "
        "Une augmentation du P95 par rapport à la moyenne globale suggère la présence de micro-événements intenses, même si la scène reste globalement modérée. "
        "Dans l’analyse comparative, un P95 plus élevé sur un passage signale des instants de mouvement rapide plus fréquents ou plus marqués. "
        "Cette métrique aide à repérer des changements brusques qui ne suffisent pas à augmenter la moyenne mais qui restent significatifs."
    )
    st.markdown(
        "L’énergie du mouvement est la somme de toutes les magnitudes sur un pas, ce qui en fait un indicateur volumique du déplacement total. "
        "Elle cumule l’intensité des vecteurs de tous les pixels et met en évidence les instants où l’activité globale est maximale. "
        "Dans des vidéos longues, l’énergie permet de repérer des segments particulièrement dynamiques sans examiner image par image. "
        "Cette mesure est proche de la magnitude moyenne mais pondère davantage les scènes couvrant de grandes surfaces en mouvement. "
        "Combinée à la magnitude moyenne dans un score composite, elle renforce la stabilité de la détection d’événements inhabituels."
    )
    st.markdown(
        "La direction dominante correspond à l’orientation moyenne des vecteurs du flux optique, projetée sur le cercle trigonométrique. "
        "Elle révèle si les mouvements tendent majoritairement vers une même direction, ce qui peut signaler un geste orienté ou un déplacement global de la scène. "
        "Cette information est précieuse lorsqu’on étudie des comportements structurés, comme une personne qui se tourne ou un cadrage qui glisse. "
        "Elle se complète par la dispersion, afin de ne pas sur-interpréter une moyenne de directions lorsque les mouvements sont contradictoires. "
        "Dans une perspective d’analyse des interactions, une direction dominante stable peut caractériser des séquences de geste répétitif ou une posture dirigée."
    )
    st.markdown(
        "La dispersion de direction mesure la concentration des orientations autour de la moyenne, via la longueur résultante normalisée des vecteurs unitaires. "
        "Une dispersion faible indique des orientations cohérentes entre elles, tandis qu’une dispersion forte traduit de l’agitation ou des mouvements désordonnés. "
        "Cette mesure aide à distinguer les passages où l’attention corporelle est focalisée de ceux où les déplacements sont erratiques. "
        "Dans une analyse comparative, une dispersion plus faible sur un passage peut être associée à une tâche précise ou à une activité dirigée. "
        "Inversement, une dispersion élevée peut coïncider avec des phases de transition, d’exploration ou de réorganisation de la scène."
    )
    st.markdown(
        "Le score composite combine la magnitude moyenne et l’énergie du mouvement après standardisation, afin d’obtenir un indicateur synthétique et comparable. "
        "La standardisation par z-score recentre chaque métrique autour de sa moyenne et la met à l’échelle de son écart-type, ce qui évite qu’une unité domine arbitrairement l’autre. "
        "La moyenne des deux z-scores capture à la fois l’intensité typique et le volume global des mouvements sur chaque pas. "
        "Ce score est ensuite comparé à un seuil fixe de déviation importante pour marquer les anomalies, ce qui rend la décision lisible et reproductible. "
        "Cette construction stabilise la détection dans des contextes variés, y compris lorsque l’échantillonnage temporel provient d’un timelapse."
    )
    st.markdown(
        "L’intervalle d’analyse optionnel permet de cibler automatiquement une portion de la vidéo sans intervention supplémentaire. "
        "Lorsque l’intervalle est activé, la découpe est effectuée au moment de l’extraction d’images par FFmpeg, de sorte que seuls les pas pertinents sont calculés. "
        "Cette approche est adaptée aux timelapses, où l’espacement entre images originales peut être important sans empêcher l’estimation du mouvement apparent. "
        "Le choix d’un intervalle n’influence pas la définition des images ni la cadence d’extraction, qui demeurent constantes pour garantir la comparabilité. "
        "En pratique, cette focalisation accélère l’analyse lorsque l’on souhaite étudier précisément un passage critique sans bruit contextuel."
    )
