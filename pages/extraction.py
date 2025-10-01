# pages/extraction.py
# Page d’extraction classique : audio MP3, images fixes (1/25/N fps), extrait par intervalle, ré-encodage H.264/AAC.
# Cette page suppose que main.py a déjà préparé une vidéo base dans st.session_state["video_base"].

import os
from pathlib import Path
import subprocess
import streamlit as st

from core_media import (
    initialiser_repertoires, info_ffmpeg, SEUIL_APERCU_OCTETS
)

# ----------------- Utilitaires locaux -----------------

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

def formater_hhmmss_ou_secondes(valeur):
    """Accepte HH:MM:SS ou un nombre de secondes; retourne toujours HH:MM:SS."""
    s = str(valeur).strip()
    if s.count(":") in (1, 2):
        return s
    try:
        x = max(0, float(s))
    except Exception:
        x = 0.0
    h = int(x // 3600)
    m = int((x % 3600) // 60)
    sec = int(x % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"

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
st.set_page_config(page_title="Extraction", layout="wide")
st.title("Extraction")
st.markdown("**www.codeandcortex.fr**")

if not st.session_state.get("video_base"):
    st.warning("Aucune vidéo préparée. Retourne sur la page d’accueil pour préparer une source.")
    st.stop()

video_base = Path(st.session_state["video_base"])
if not video_base.exists():
    st.warning("La vidéo préparée n’existe plus sur le disque. Relance la préparation.")
    st.stop()

nom_base_court = st.session_state.get("base_court") or video_base.stem

# Aperçu conditionnel de la vidéo base
with st.expander("Aperçu rapide de la vidéo de base"):
    if video_base.stat().st_size <= SEUIL_APERCU_OCTETS:
        with open(video_base, "rb") as fh:
            st.video(fh.read(), format="video/mp4")
    else:
        st.info("Aperçu désactivé car le fichier est volumineux.")

# Diagnostic FFmpeg
with st.expander("Diagnostic FFmpeg"):
    ff = chemin_ffmpeg()
    st.write(f"ffmpeg : {ff or 'introuvable'}")
    if not ff:
        st.stop()

# ----------------- Onglets d’extraction -----------------

ong1, ong2, ong3, ong4 = st.tabs([
    "Audio MP3",
    "Images fixes",
    "Extrait par intervalle",
    "Ré-encodage H.264/AAC"
])

# Onglet Audio MP3
with ong1:
    st.subheader("Extraction audio MP3")
    col1, col2 = st.columns(2)
    with col1:
        bitrate = st.selectbox("Débit audio", ["96k", "128k", "160k", "192k", "256k"], index=1)
    with col2:
        mono = st.checkbox("Forcer mono", value=False)

    mp3_path = REP_SORTIE / f"{nom_base_court}.mp3"
    if st.button("Extraire MP3"):
        cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_base)]
        if mono:
            cmd += ["-ac", "1"]
        cmd += ["-vn", "-c:a", "libmp3lame", "-b:a", bitrate, str(mp3_path)]
        ok, log = executer_commande(cmd)
        afficher_log("Extraction MP3", ok, log, str(mp3_path))
        if ok and mp3_path.exists():
            st.audio(mp3_path.read_bytes())
            st.download_button("Télécharger le MP3", data=mp3_path.read_bytes(), file_name=mp3_path.name, mime="audio/mpeg")

# Onglet Images fixes
with ong2:
    st.subheader("Extraction d’images fixes")
    c1, c2, c3 = st.columns(3)
    with c1:
        mode = st.selectbox("Fréquence", ["1 image/s", "25 images/s", "N images/s"], index=0)
    with c2:
        n_fps = st.number_input("N images/s (si mode N)", min_value=1, max_value=120, value=10, step=1)
    with c3:
        largeur = st.number_input("Largeur des images (px)", min_value=160, max_value=3840, value=1280, step=16)

    fps = 1 if mode == "1 image/s" else (25 if mode == "25 images/s" else int(n_fps))
    dossier_images = (BASE_DIR / "images" / f"{nom_base_court}").resolve()
    dossier_images.mkdir(parents=True, exist_ok=True)
    motif = str(dossier_images / "img_%06d.jpg")

    if st.button("Extraire les images"):
        cmd = [
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_base),
            "-vf", f"fps={fps},scale={int(largeur)}:-2",
            "-q:v", "2",
            motif
        ]
        ok, log = executer_commande(cmd)
        afficher_log(f"Extraction d’images à {fps} i/s", ok, log, str(dossier_images))
        if ok:
            st.caption(f"Motif de fichiers : {motif}")

# Onglet Extrait par intervalle
with ong3:
    st.subheader("Extraction d’un intervalle vidéo")
    c1, c2, c3 = st.columns(3)
    with c1:
        debut = st.text_input("Début (HH:MM:SS ou secondes)", value="00:00:05")
    with c2:
        fin = st.text_input("Fin (HH:MM:SS ou secondes)", value="00:00:25")
    with c3:
        copy = st.checkbox("Copie directe (-c copy, sans ré-encodage)", value=False)

    c4, c5 = st.columns(2)
    with c4:
        largeur_iv = st.number_input("Largeur si ré-encodage (px)", min_value=320, max_value=3840, value=1280, step=16)
    with c5:
        crf_iv = st.slider("CRF si ré-encodage", min_value=18, max_value=35, value=28)

    interval_path = REP_SORTIE / f"{nom_base_court}_intervalle.mp4"
    if st.button("Extraire l’intervalle"):
        t0 = formater_hhmmss_ou_secondes(debut)
        t1 = formater_hhmmss_ou_secondes(fin)
        if copy:
            cmd = [
                ff, "-y", "-hide_banner", "-loglevel", "error",
                "-ss", t0, "-to", t1,
                "-i", str(video_base),
                "-c", "copy",
                "-movflags", "+faststart",
                str(interval_path)
            ]
        else:
            cmd = [
                ff, "-y", "-hide_banner", "-loglevel", "error",
                "-ss", t0, "-to", t1,
                "-i", str(video_base),
                "-vf", f"scale={int(largeur_iv)}:-2",
                "-c:v", "libx264",
                "-crf", str(crf_iv),
                "-preset", "slow",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "128k",
                "-ac", "2",
                "-movflags", "+faststart",
                str(interval_path)
            ]
        ok, log = executer_commande(cmd)
        afficher_log("Extraction par intervalle", ok, log, str(interval_path))
        if ok and interval_path.exists():
            st.video(interval_path.read_bytes(), format="video/mp4")
            st.download_button("Télécharger l’extrait", data=interval_path.read_bytes(), file_name=interval_path.name, mime="video/mp4")

# Onglet Ré-encodage
with ong4:
    st.subheader("Ré-encodage H.264/AAC de la vidéo base")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        largeur_full = st.number_input("Largeur cible (px)", min_value=320, max_value=3840, value=1280, step=16)
    with c2:
        crf_full = st.slider("CRF (plus bas = meilleure qualité)", min_value=18, max_value=35, value=28)
    with c3:
        preset = st.selectbox("Preset x264", ["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"], index=6)
    with c4:
        audio_kbps = st.selectbox("Débit audio", ["64k","96k","128k","160k","192k"], index=1)

    sortie_full = REP_SORTIE / f"{nom_base_court}_full.mp4"
    if st.button("Ré-encoder la vidéo base"):
        cmd = [
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_base),
            "-vf", f"scale={int(largeur_full)}:-2",
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf_full),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", audio_kbps,
            "-ac", "2",
            "-movflags", "+faststart",
            str(sortie_full)
        ]
        ok, log = executer_commande(cmd)
        afficher_log("Ré-encodage H.264/AAC", ok, log, str(sortie_full))
        if ok and sortie_full.exists():
            st.video(sortie_full.read_bytes(), format="video/mp4")
            st.download_button("Télécharger la vidéo ré-encodée", data=sortie_full.read_bytes(), file_name=sortie_full.name, mime="video/mp4")
