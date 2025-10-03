# pages/analyse_mouvements.py
# Analyse des mouvements avec choix de l'intervalle entre frames (pas d'analyse)
# et visualisation des anomalies (encadrement rouge).
# Source prioritaire : MP4 importé, sinon vidéo préparée.
# Extraction d'images : mode "toutes les frames (timelapse/natif)" ou "cadence fixe (i/s)" en 1080p.
# Calcul : flux optique Farneback entre paires consécutives du sous-échantillon choisi,
# métriques par pas (magnitude moyenne, écart-type, p95, énergie, direction dominante, dispersion),
# score composite (z(magnitude_moyenne) + z(énergie))/2, anomalies si z >= 2.5.

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
    debut_s: Optional[float] = None,
    fin_s: Optional[float] = None
) -> Tuple[bool, str]:
    """
    Extrait des images JPG en 1080p (largeur 1920).
    mode_extraction = "natifs" -> toutes les frames sources (timelapse, VFR), sans filtre fps.
    mode_extraction = "fixe"   -> fps_ech images/seconde (uniforme).
    Un intervalle temporel [debut_s, fin_s] peut être fourni (optionnel).
    Les fichiers sont nommés frame_%06d.jpg.
    """
    if dossier.exists():
        try:
            shutil.rmtree(dossier)
        except Exception:
            pass
    dossier.mkdir(parents=True, exist_ok=True)

    motif = str(dossier / "frame_%06d.jpg")

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    if debut_s is not None and debut_s > 0:
        cmd += ["-ss", str(float(debut_s))]
    if fin_s is not None and fin_s > 0 and (debut_s is None or fin_s > debut_s):
        cmd += ["-to", str(float(fin_s))]
    cmd += ["-i", str(video)]

    if mode_extraction == "natifs":
        # Toutes les frames en 1080p (sans imposer un fps). -vsync vfr conserve le rythme variable.
        cmd += ["-vf", "scale=1920:-2", "-vsync", "vfr", "-q:v", "2", motif]
    else:
        # Cadence fixe fps_ech en 1080p.
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
    "Les images sont extraites en 1080p, soit à cadence fixe, soit en conservant toutes les frames natives, ce qui convient aux timelapses. "
    "Pour chaque pas temporel, on calcule la magnitude moyenne, l’écart-type, le 95e percentile, l’énergie, la direction dominante et la dispersion. "
    "Un score composite est formé en combinant les versions standardisées de la magnitude moyenne et de l’énergie. "
    "Les pas dont le score composite dépasse nettement la moyenne (z ≥ 2.5) sont considérés comme des anomalies et sont encadrés en rouge."
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
    fps_ech = st.number_input("Cadence fixe (si sélectionnée)", min_value=1, max_value=30, value=4, step=1)
with col3:
    pas_analyse = st.number_input("Pas d’analyse (1 = chaque image)", min_value=1, max_value=50, value=1, step=1)

st.caption(
    "Le pas d’analyse contrôle l’intervalle entre images analysées après extraction. "
    "Par exemple, un pas de 5 analysera les paires (frame 0 → 5), (5 → 10), etc. "
    "En timelapse, un pas de 1 est généralement adapté puisque l’espacement est déjà grand. "
    "En vidéo classique, augmenter le pas accélère l’analyse en sautant des frames intermédiaires. "
    "Le mode d’extraction « toutes les frames » utilise le rythme natif ; la cadence fixe impose un sous-échantillonnage régulier."
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
    if len(imgs_gray) < 2:
        st.error("Trop peu d’images extraites pour analyser.")
        with st.expander("Journal FFmpeg"):
            st.code(log_ext or "(vide)", language="bash")
        st.stop()

    # Sous-échantillonnage pour le pas d'analyse (1 = chaque image, 2 = une sur deux, etc.)
    # On construit des indices 0, pas, 2*pas, ...
    indices = list(range(0, len(imgs_gray), int(pas_analyse)))
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

    # Aperçu global réparti, avec encadrement si anomalie
    st.subheader("Aperçu global (vignettes réparties, anomalies encadrées)")
    N = len(indices)
    nb_vignettes = min(48, N)
    sel = np.linspace(0, N - 1, num=nb_vignettes, dtype=int)
    cols_par_ligne = 8
    kk = 0
    for _ in range(math.ceil(len(sel) / cols_par_ligne)):
        cols = st.columns(cols_par_ligne)
        for c in cols:
            if kk >= len(sel):
                break
            kidx = int(sel[kk])
            fr = indices[kidx]
            # Chercher si cette frame est la destination d'un pas marqué anomalie
            z_here = None
            encadre = False
            hit = df[df["frame_curr"] == fr]
            if not hit.empty:
                z_here = float(hit["score_composite_z"].iloc[0])
                encadre = bool(hit["anomalie"].iloc[0])
            if 0 <= fr < len(imgs_rgb):
                img = encadrer_rouge(cv2, imgs_rgb[fr], epaisseur=6) if encadre else imgs_rgb[fr]
                cap = f"frame #{fr}" + (f" • z={z_here:.2f}" if z_here is not None else "")
                c.image(img, caption=cap, use_container_width=False)
            kk += 1

    # Explications pédagogiques des paramètres (au moins cinq phrases par paramètre)
    st.subheader("Explications détaillées")
    st.markdown(
        "Le pas d’analyse représente l’intervalle entre images successives utilisées pour le calcul du flux optique. "
        "Un pas égal à un signifie que chaque image est utilisée et que l’on évalue les déplacements entre deux images consécutives. "
        "Augmenter le pas revient à sauter des images intermédiaires, ce qui accélère l’analyse et accentue les variations apparentes entre deux états éloignés. "
        "Cette approche est particulièrement utile lorsque les frames extraites proviennent déjà d’un timelapse, car l’espacement temporel réel est important. "
        "Dans une vidéo classique, choisir un pas supérieur à un permet d’alléger la charge de calcul tout en conservant une sensibilité raisonnable aux changements visuels."
    )
    st.markdown(
        "Le mode d’extraction « toutes les frames (timelapse/natif) » tente de conserver chaque image distincte produite par la vidéo source. "
        "Cette option est adaptée aux timelapses et aux séquences à cadence variable, car elle respecte le rythme d’acquisition original. "
        "Elle peut générer davantage d’images à traiter, offrant une granularité plus fine de l’analyse au prix d’un temps de calcul plus long. "
        "La conversion en 1080p garantit une dimension cohérente pour le calcul du flux optique sans perdre trop d’information spatiale. "
        "Lorsque la vidéo est très longue, il reste possible de combiner cette option avec un pas d’analyse supérieur à un pour maîtriser la complexité."
    )
    st.markdown(
        "La cadence fixe définit un nombre d’images extraites par seconde indépendamment de la vidéo source. "
        "Cette stratégie uniformise l’échantillonnage temporel, utile pour comparer plusieurs vidéos différentes avec la même résolution temporelle. "
        "Elle réduit l’empreinte mémoire et le nombre de paires à analyser si la vidéo a un taux d’images natif élevé. "
        "Elle peut, en revanche, lisser certains micro-événements présents entre deux extractions si la cadence choisie est trop faible. "
        "Dans la pratique, une cadence modérée conjointe à un pas d’analyse ajusté permet un compromis entre fidélité et performance."
    )
    st.markdown(
        "La magnitude moyenne mesure l’intensité moyenne du déplacement des pixels entre deux images prises à l’intervalle défini. "
        "Cette grandeur synthétise la quantité de mouvement visible, indépendamment de la direction des déplacements. "
        "Elle est robuste au bruit local car elle agrège l’information sur l’ensemble de l’image. "
        "Une augmentation nette de la magnitude moyenne signale un passage plus dynamique que la tendance générale. "
        "Dans un timelapse, elle met en évidence les phases où la scène a le plus évolué d’un cliché au suivant."
    )
    st.markdown(
        "L’énergie du mouvement cumule toutes les magnitudes au sein d’un même pas, ce qui souligne les événements couvrant de larges portions de l’image. "
        "Elle complète la magnitude moyenne en pondérant davantage les zones étendues en mouvement. "
        "Dans les comparaisons, un pic d’énergie suggère une transformation globale de la scène plutôt qu’un simple détail local. "
        "Utilisée avec la magnitude moyenne dans un score composite, elle stabilise la détection des passages inhabituels. "
        "Cette combinaison réduit les faux positifs dus à de petites zones très rapides ou à des fluctuations locales."
    )
    st.markdown(
        "Le score composite est obtenu en standardisant la magnitude moyenne et l’énergie puis en prenant leur moyenne. "
        "La standardisation par z-score ramène chaque métrique autour de zéro et l’exprime en écart-type, ce qui les rend comparables. "
        "Un score composite élevé indique qu’au même pas, l’intensité moyenne et le volume total de mouvement dépassent notablement la moyenne globale. "
        "Le seuil d’anomalie fixé à 2.5 écart-types isole les déviations marquées sans noyer l’utilisateur sous des alertes mineures. "
        "Les vignettes correspondantes sont encadrées en rouge pour un repérage instantané dans la mosaïque."
    )
