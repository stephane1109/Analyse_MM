# pages/analyse_mouvements.py
# Analyse comparative des mouvements par flux optique (Farneback) avec métriques :
# - Magnitude moyenne
# - Direction dominante et dispersion (statistiques circulaires)
# - Variance (écart-type) de la magnitude
# - Histogramme de magnitude (lent / moyen / rapide)
# - Ratio de frames "immobiles"
# + Comparaison Segment d'intérêt vs Vidéo entière (différences et ratios)
#
# Source prioritaire : MP4 importé ici. Fallback : vidéo déjà préparée (st.session_state["video_base"]).
# Extraction d'images en 1080p via FFmpeg, calculs avec OpenCV (opencv-python-headless).

import math
import shutil
from pathlib import Path
from typing import Tuple, List, Optional, Dict

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
# Extraction d'images (FFmpeg)
# =============================

def extraire_frames_1080p(ffmpeg: str, video: Path, dossier: Path, fps_ech: int) -> Tuple[bool, str]:
    """
    Extrait des images JPG en 1080p (largeur 1920) à fps_ech images/s.
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
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video), "-vf", filtre, "-q:v", "2", motif]
    return executer(cmd)

# =============================
# Chargement images
# =============================

def charger_images_gris(cv2, dossier: Path) -> List[np.ndarray]:
    """Charge toutes les images JPG en niveaux de gris (liste)."""
    fichiers = sorted(dossier.glob("frame_*.jpg"))
    images = []
    for f in fichiers:
        bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        images.append(gray)
    return images

# =============================
# Flux optique + métriques
# =============================

def farneback(cv2, prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Flux optique dense Farneback (retourne un champ (H,W,2) de vecteurs (dx,dy))."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray,
        None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0
    )
    return flow

def stats_circulaires(flow: np.ndarray) -> Tuple[float, float]:
    """
    Calcule la direction dominante (angle moyen en degrés) et la dispersion circulaire.
    Méthode : on convertit (dx,dy) en angles theta = atan2(dy, dx), on moyenne sur le cercle.
    - direction_dominante_deg : angle en degrés (-180,180]
    - dispersion : 1 - R, avec R = longueur du vecteur résultant moyen (entre 0 et 1)
      0 => directions très alignées (faible dispersion) ; 1 => directions très dispersées.
    """
    dx = flow[..., 0].astype(np.float32)
    dy = flow[..., 1].astype(np.float32)
    angle = np.arctan2(dy, dx)  # radians
    # Vecteurs unitaires
    ux = np.cos(angle)
    uy = np.sin(angle)
    R_x = float(np.mean(ux))
    R_y = float(np.mean(uy))
    R = float(np.sqrt(R_x**2 + R_y**2))
    direction = float(np.degrees(np.arctan2(R_y, R_x)))  # degrés
    dispersion = float(1.0 - R)
    return direction, dispersion

def histogramme_magnitude(mag: np.ndarray, bornes: List[float]) -> np.ndarray:
    """Histogramme des magnitudes selon des bornes (en px/frame). Retourne les effectifs par bin."""
    vals = mag.flatten().astype(np.float32)
    hist, _ = np.histogram(vals, bins=np.array(bornes, dtype=np.float32))
    return hist

def metriques_par_pas(flow: np.ndarray, seuil_mobile: float, bornes_hist: List[float]) -> Dict[str, float]:
    """
    Calcule les métriques pour un pas (entre deux frames consécutives).
    - magnitude moyenne
    - écart-type magnitude
    - p95 magnitude
    - ratio de pixels "mobiles" (> seuil_mobile)
    - énergie (somme des magnitudes)
    - direction dominante (degrés) et dispersion
    - histogramme (déployé en colonnes "h_bin_i")
    """
    mag = np.linalg.norm(flow, axis=2).astype(np.float32)
    m = float(np.mean(mag))
    s = float(np.std(mag))
    p95 = float(np.percentile(mag, 95))
    ratio_mobile = float(np.mean(mag > seuil_mobile))
    energie = float(np.sum(mag))
    direction, dispersion = stats_circulaires(flow)
    hist = histogramme_magnitude(mag, bornes_hist)

    met = {
        "magnitude_moyenne": m,
        "magnitude_ecart_type": s,
        "magnitude_p95": p95,
        "ratio_pixels_mobiles": ratio_mobile,
        "energie_mouvement": energie,
        "direction_dominante_deg": direction,
        "dispersion_direction": dispersion,
    }
    # Colonnes histogramme : h_bin_0 ... h_bin_{n-2} (si N bornes => N-1 bacs)
    for i in range(len(bornes_hist) - 1):
        met[f"h_bin_{i}"] = int(hist[i])
    return met

def agreger_sur_plage(df: pd.DataFrame, colonnes_hist: List[str]) -> Dict[str, float]:
    """
    Agrège les métriques sur une plage de pas (moyennes pour les stats, sommes pour l'histogramme).
    Retourne un dict avec les mêmes clés que metriques_par_pas.
    """
    res = {}
    # Moyennes des métriques scalaires
    scalaires = [
        "magnitude_moyenne", "magnitude_ecart_type", "magnitude_p95",
        "ratio_pixels_mobiles", "energie_mouvement",
        "direction_dominante_deg", "dispersion_direction",
    ]
    for k in scalaires:
        if k in df:
            res[k] = float(df[k].mean())
    # Histogrammes : somme des effectifs
    for h in colonnes_hist:
        if h in df:
            res[h] = int(df[h].sum())
    return res

# =============================
# Page Streamlit
# =============================

BASE_DIR, REP_SORTIE, REP_TMP = initialiser_repertoires()

st.set_page_config(page_title="Analyse comparative des mouvements (flux optique)", layout="wide")
st.title("Analyse comparative des mouvements (flux optique)")
st.markdown("**www.codeandcortex.fr**")

st.markdown(
    "L’outil calcule des **indices de mouvement** validés en vision par ordinateur, "
    "et compare un **segment d’intérêt** à la **vidéo entière** :\n"
    "- Magnitude moyenne du flux optique : activité globale (vitesse moyenne des mouvements).\n"
    "- Direction dominante et dispersion : orientation moyenne des gestes et degré d’agitation.\n"
    "- Variance (écart-type) de la magnitude : hétérogénéité des mouvements.\n"
    "- Histogramme de magnitude : répartition des vitesses (lent / moyen / rapide).\n"
    "- Ratio de frames immobiles : proportion de pas où la magnitude moyenne est faible.\n\n"
    "Interprétation type en SHS : une magnitude moyenne plus basse dans un passage peut indiquer moins d’activité motrice, "
    "souvent corrélée à une activité cognitive/métacognitive accrue (écoute, réflexion)."
)

ff = trouver_ffmpeg()
if not ff:
    st.error("FFmpeg introuvable. Fournis un binaire ./bin/ffmpeg ou vérifie /usr/bin/ffmpeg.")
    st.stop()

cv2, cv_err = importer_cv2()
if cv2 is None:
    st.error(cv_err)
    st.stop()

# -----------------------------
# Choix de la source
# -----------------------------

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

# -----------------------------
# Paramètres (simples et explicites)
# -----------------------------

st.subheader("Paramètres")
c1, c2, c3, c4 = st.columns(4)
with c1:
    fps_ech = st.number_input("Cadence extraction (images/s)", min_value=1, max_value=30, value=4, step=1)
with c2:
    seuil_mobile = st.number_input("Seuil pixel mobile (px/frame)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
with c3:
    seuil_frame_immobile = st.number_input("Seuil frame immobile (magnitude moyenne)", min_value=0.0, max_value=5.0, value=0.2, step=0.05)
with c4:
    montrer_log = st.checkbox("Afficher le journal FFmpeg", value=False)

st.markdown(
    "Histogramme de magnitude (px/frame) : bornes entre lesquelles on compte les vitesses. "
    "Par défaut : [0, 0.5, 1, 2, 5, +∞)."
)
bornes_defaut = [0.0, 0.5, 1.0, 2.0, 5.0, 1e9]
bornes_affiche = st.text_input("Bornes (séparées par des virgules)", value="0,0.5,1,2,5,1e9")
try:
    bornes_hist = [float(x.strip()) for x in bornes_affiche.split(",") if x.strip() != ""]
    if len(bornes_hist) < 2:
        bornes_hist = bornes_defaut
except Exception:
    bornes_hist = bornes_defaut

st.subheader("Segment d’intérêt")
st.caption("Saisir le début et la fin en secondes (ils seront mappés sur les pas d’analyse à "
           "la cadence d’extraction choisie).")
c5, c6 = st.columns(2)
with c5:
    t_debut = st.number_input("Début (s)", min_value=0.0, value=0.0, step=0.5)
with c6:
    t_fin = st.number_input("Fin (s)", min_value=0.5, value=10.0, step=0.5)

lancer = st.button("Calculer les métriques", type="primary")

# -----------------------------
# Calculs
# -----------------------------

if lancer:
    if video_path is None:
        st.error("Aucune source vidéo. Importez un MP4 ou sélectionnez la vidéo préparée.")
        st.stop()

    frames_dir = (BASE_DIR / "frames_analysis" / video_path.stem).resolve()
    ok_ext, log_ext = extraire_frames_1080p(ff, video_path, frames_dir, int(fps_ech))
    if not ok_ext:
        st.error("Échec extraction des images avec FFmpeg.")
        if montrer_log:
            st.code(log_ext or "(log vide)", language="bash")
        st.stop()

    imgs = charger_images_gris(cv2, frames_dir)
    if len(imgs) < 2:
        st.error("Aucune image ou trop peu d’images extraites. Impossible d’analyser.")
        if montrer_log:
            st.code(log_ext or "(log vide)", language="bash")
        st.stop()

    # Flux optique pour tous les pas
    metriques_liste: List[Dict[str, float]] = []
    for i in range(1, len(imgs)):
        flow = farneback(cv2, imgs[i-1], imgs[i])
        met = metriques_par_pas(flow, float(seuil_mobile), bornes_hist)
        metriques_liste.append(met)

    # Table des pas
    df = pd.DataFrame(metriques_liste)
    df.insert(0, "pas_index", np.arange(1, len(metriques_liste) + 1))
    df.insert(1, "temps_s_approx", df["pas_index"] / float(fps_ech))

    # Ratio de frames "immobiles" (magnitude moyenne < seuil_frame_immobile)
    ratio_frames_immobiles_global = float(np.mean(df["magnitude_moyenne"] < float(seuil_frame_immobile)))

    # Définition segment (bornage dans la table des pas)
    t_debut = max(0.0, float(t_debut))
    t_fin = max(t_debut + 1.0 / float(fps_ech), float(t_fin))
    df_segment = df[(df["temps_s_approx"] >= t_debut) & (df["temps_s_approx"] <= t_fin)].copy()
    if df_segment.empty:
        st.warning("Le segment ne contient aucun pas (vérifie début/fin et la cadence d’extraction).")
        st.stop()
    ratio_frames_immobiles_segment = float(np.mean(df_segment["magnitude_moyenne"] < float(seuil_frame_immobile)))

    # Colonnes histogramme pour agrégation
    colonnes_hist = [c for c in df.columns if c.startswith("h_bin_")]

    # Agrégations
    global_agg = agreger_sur_plage(df, colonnes_hist)
    segment_agg = agreger_sur_plage(df_segment, colonnes_hist)

    # Ajout des ratios "frames immobiles"
    global_agg["ratio_frames_immobiles"] = ratio_frames_immobiles_global
    segment_agg["ratio_frames_immobiles"] = ratio_frames_immobiles_segment

    # Comparaisons segment vs global (différences et ratios)
    def diff_ratio(seg_val: float, glob_val: float) -> Tuple[float, float]:
        r = float(seg_val) / float(glob_val) if abs(glob_val) > 1e-12 else float("inf")
        d = float(seg_val) - float(glob_val)
        return d, r

    lignes_comp = []
    cles_comparees = [
        "magnitude_moyenne",
        "magnitude_ecart_type",
        "magnitude_p95",
        "ratio_pixels_mobiles",
        "energie_mouvement",
        "direction_dominante_deg",
        "dispersion_direction",
        "ratio_frames_immobiles",
    ]
    for k in cles_comparees:
        if k in segment_agg and k in global_agg:
            d, r = diff_ratio(segment_agg[k], global_agg[k])
            lignes_comp.append({
                "metrique": k,
                "global": global_agg[k],
                "segment": segment_agg[k],
                "diff_segment_moins_global": d,
                "ratio_segment_sur_global": r
            })
    df_comp = pd.DataFrame(lignes_comp)

    # Histogrammes : on rapporte les effectifs et, pour lecture, les pourcentages (normalisés)
    def hist_df(d: Dict[str, float]) -> pd.DataFrame:
        vals = []
        total = sum(int(d[h]) for h in colonnes_hist if h in d)
        for i, h in enumerate(colonnes_hist):
            v = int(d.get(h, 0))
            borne_g = bornes_hist[i]
            borne_d = bornes_hist[i+1]
            etiquette = f"[{borne_g}, {borne_d})"
            pourcent = (100.0 * v / total) if total > 0 else 0.0
            vals.append({"bin": etiquette, "effectif": v, "pourcent": pourcent})
        return pd.DataFrame(vals)

    df_hist_global = hist_df(global_agg)
    df_hist_segment = hist_df(segment_agg)

    # =========================
    # Rendus dans Streamlit
    # =========================

    st.subheader("Résumés numériques")
    colG, colS = st.columns(2)
    with colG:
        st.markdown("**Moyennes globales (toute la vidéo)**")
        st.dataframe(pd.DataFrame([global_agg]).T.rename(columns={0: "valeur"}))
    with colS:
        st.markdown(f"**Moyennes sur le segment**  (de {t_debut:.1f}s à {t_fin:.1f}s)")
        st.dataframe(pd.DataFrame([segment_agg]).T.rename(columns={0: "valeur"}))

    st.subheader("Comparaison Segment vs Global")
    st.caption("Différence = segment − global ; Ratio = segment / global.")
    st.dataframe(df_comp)

    st.subheader("Histogrammes de magnitude")
    cHG, cHS = st.columns(2)
    with cHG:
        st.markdown("**Global**")
        st.dataframe(df_hist_global)
        st.bar_chart(df_hist_global.set_index("bin")["pourcent"])
    with cHS:
        st.markdown("**Segment**")
        st.dataframe(df_hist_segment)
        st.bar_chart(df_hist_segment.set_index("bin")["pourcent"])

    st.subheader("Téléchargements")
    st.download_button(
        "Exporter les pas (indices par pas) en CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="indices_par_pas.csv",
        mime="text/csv"
    )
    st.download_button(
        "Exporter le résumé global en CSV",
        data=pd.DataFrame([global_agg]).to_csv(index=False).encode("utf-8"),
        file_name="resume_global.csv",
        mime="text/csv"
    )
    st.download_button(
        "Exporter le résumé segment en CSV",
        data=pd.DataFrame([segment_agg]).to_csv(index=False).encode("utf-8"),
        file_name="resume_segment.csv",
        mime="text/csv"
    )
    st.download_button(
        "Exporter la comparaison segment vs global en CSV",
        data=df_comp.to_csv(index=False).encode("utf-8"),
        file_name="comparaison_segment_vs_global.csv",
        mime="text/csv"
    )

    if montrer_log:
        with st.expander("Journal FFmpeg"):
            st.code(log_ext or "(log vide)", language="bash")

    # Notes pédagogiques affichées dans la page
    st.subheader("Notes d’interprétation (sciences humaines)")
    st.markdown(
        "- **Magnitude moyenne** plus basse dans le segment : activité motrice moindre, possiblement liée à une activité "
        "cognitive/métacognitive plus forte (écoute, réflexion).\n"
        "- **Direction dominante stable** et **faible dispersion** : gestes orientés, posture dirigée (ex. prise de parole cadrée).\n"
        "- **Variance** faible : mouvements plus homogènes, pouvant refléter une concentration accrue.\n"
        "- **Histogramme** décalé vers les basses vitesses : phase calme ; vers les hautes : phase dynamique.\n"
        "- **Frames immobiles** nombreuses : immobilité (écoute, attention)."
    )
