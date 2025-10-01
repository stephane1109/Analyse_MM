# pages/timelapse.py
# Page de recomposition timelapse : utilise un dossier d’images pour créer une vidéo timelapse H.264/AAC.
# Cette page suppose que la page d’extraction a pu générer un dossier d’images dans /tmp/appdata/images/<nom_base_court>.

import os
from pathlib import Path
import subprocess
import streamlit as st

from core_media import (
    initialiser_repertoires, info_ffmpeg
)

# ----------------- Utilitaires -----------------

def executer_commande(cmd: list):
    """Exécute une commande système et retourne (ok, log)."""
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

def chemin_ffmpeg():
    """Retourne le chemin de ffmpeg ou None."""
    chemin, _ = info_ffmpeg()
    return chemin

def afficher_log(titre: str, ok: bool, log: str, chemin_sortie: str | None = None):
    """Affiche un statut et le log éventuel."""
    if ok:
        st.success(f"{titre} terminé.")
        if chemin_sortie and Path(chemin_sortie).exists():
            st.caption(f"Fichier généré : {chemin_sortie}")
        if log:
            with st.expander("Journal FFmpeg"):
                st.code(log, language="bash")
    else:
        st.error(f"{titre} en échec.")
        if log:
            st.code(log, language="bash")

# ----------------- Initialisation -----------------

BASE_DIR, REP_SORTIE, REP_TMP = initialiser_repertoires()
st.set_page_config(page_title="Timelapse", layout="wide")
st.title("Timelapse")
st.markdown("**www.codeandcortex.fr**")

# Nom de base hérité de la préparation (optionnel)
nom_base_court = st.session_state.get("base_court")

# Détection des dossiers d’images disponibles
racine_images = (BASE_DIR / "images").resolve()
dossiers = []
if racine_images.exists():
    for d in sorted(racine_images.iterdir()):
        if d.is_dir():
            dossiers.append(str(d))

ff = chemin_ffmpeg()
with st.expander("Diagnostic FFmpeg"):
    st.write(f"ffmpeg : {ff or 'introuvable'}")
    if not ff:
        st.stop()

if not dossiers:
    st.info("Aucun dossier d’images détecté. Générez d’abord des images depuis la page Extraction.")
    st.stop()

# ----------------- Paramétrage timelapse -----------------

# Pré-sélection si le dossier porte le nom base court
index_defaut = 0
if nom_base_court:
    noms = [Path(d).name for d in dossiers]
    if nom_base_court in noms:
        index_defaut = noms.index(nom_base_court)

dossier_sel = st.selectbox("Dossier d’images", options=dossiers, index=index_defaut)
motif = st.text_input("Motif d’images (printf-style)", value="img_%06d.jpg")

col1, col2, col3 = st.columns(3)
with col1:
    fps_tl = st.selectbox("Fréquence timelapse (i/s)", [6, 8, 10, 12, 14], index=0)
with col2:
    largeur = st.number_input("Largeur de sortie (px)", min_value=320, max_value=3840, value=1280, step=16)
with col3:
    crf = st.slider("CRF (qualité H.264)", min_value=18, max_value=35, value=28)

nom_sortie = (nom_base_court or Path(dossier_sel).name) + f"_timelapse_{fps_tl}fps.mp4"
sortie = (BASE_DIR / "timelapse" / nom_sortie).resolve()
sortie.parent.mkdir(parents=True, exist_ok=True)

if st.button("Composer le timelapse"):
    cmd = [
        ff, "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", str(int(fps_tl)),
        "-i", str(Path(dossier_sel) / motif),
        "-vf", f"scale={int(largeur)}:-2",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(sortie)
    ]
    ok, log = executer_commande(cmd)
    afficher_log("Composition timelapse", ok, log, str(sortie))
    if ok and sortie.exists():
        st.video(sortie.read_bytes(), format="video/mp4")
        st.download_button("Télécharger le timelapse", data=sortie.read_bytes(), file_name=sortie.name, mime="video/mp4")
