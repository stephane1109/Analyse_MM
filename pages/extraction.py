# pages/extraction.py
# Extraction et téléchargement : MP4 (HD / basse def), MP3, WAV, images (1/25/N fps), extrait par intervalle.
# Aucune option technique superflue exposée : uniquement "basse def" (1280p) ou "HD".
# La vidéo source préparée est attendue dans st.session_state["video_base"] (définie par main.py).

import io
import shutil
import subprocess
from pathlib import Path
import streamlit as st

from core_media import initialiser_repertoires, info_ffmpeg, SEUIL_APERCU_OCTETS

# ---------- utilitaires simples ----------

def _run(cmd: list):
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

def _ffmpeg():
    p, _ = info_ffmpeg()
    return p

def _to_hhmmss(v):
    s = str(v).strip()
    if s.count(":") in (1, 2):
        return s
    try:
        x = max(0, float(s))
    except Exception:
        x = 0.0
    h = int(x // 3600); m = int((x % 3600) // 60); sec = int(x % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"

def _zip_dir_as_bytes(dir_path: Path) -> bytes:
    tmp_zip = dir_path.parent / (dir_path.name + ".zip")
    if tmp_zip.exists():
        try:
            tmp_zip.unlink()
        except Exception:
            pass
    shutil.make_archive(str(tmp_zip.with_suffix("")), "zip", root_dir=dir_path)
    data = tmp_zip.read_bytes()
    try:
        tmp_zip.unlink()
    except Exception:
        pass
    return data

# ---------- page ----------

BASE_DIR, REP_SORTIE, REP_TMP = initialiser_repertoires()

st.set_page_config(page_title="Extraction", layout="wide")
st.title("Extraction")
st.markdown("**www.codeandcortex.fr**")

if not st.session_state.get("video_base"):
    st.warning("Aucune vidéo préparée. Va d’abord sur la page d’accueil pour préparer la source.")
    st.stop()

video_base = Path(st.session_state["video_base"])
if not video_base.exists():
    st.warning("La vidéo préparée n’existe plus. Relance la préparation.")
    st.stop()

base_court = st.session_state.get("base_court") or video_base.stem
ff = _ffmpeg()
if not ff:
    st.error("FFmpeg introuvable.")
    st.stop()

with st.expander("Aperçu rapide"):
    if video_base.stat().st_size <= SEUIL_APERCU_OCTETS:
        with open(video_base, "rb") as fh:
            st.video(fh.read(), format="video/mp4")
    else:
        st.info("Aperçu désactivé car le fichier est volumineux.")

ong1, ong2, ong3, ong4 = st.tabs([
    "Vidéo MP4",
    "Audio (MP3 / WAV)",
    "Images fixes",
    "Extrait par intervalle"
])

# ---------- Vidéo MP4 (HD / Basse def) ----------
with ong1:
    st.subheader("Exporter la vidéo")
    choix = st.radio("Qualité", ["Basse définition (1280p)", "HD"], index=0, horizontal=True)

    if st.button("Générer et télécharger"):
        if choix.startswith("Basse"):
            sortie = REP_SORTIE / f"{base_court}_basse.mp4"
            cmd = [
                ff, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(video_base),
                "-vf", "scale=1280:-2",
                "-c:v", "libx264",
                "-preset", "slow",
                "-crf", "28",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "96k", "-ac", "2",
                "-movflags", "+faststart",
                str(sortie)
            ]
        else:
            sortie = REP_SORTIE / f"{base_court}_hd.mp4"
            cmd = [
                ff, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(video_base),
                "-c:v", "libx264",
                "-preset", "slow",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "160k", "-ac", "2",
                "-movflags", "+faststart",
                str(sortie)
            ]

        ok, log = _run(cmd)
        if ok and sortie.exists():
            data = sortie.read_bytes()
            st.success(f"Fichier prêt : {sortie.name}")
            st.video(data, format="video/mp4")
            st.download_button("Télécharger le MP4", data=data, file_name=sortie.name, mime="video/mp4")
            with st.expander("Journal FFmpeg"):
                st.code(log, language="bash")
        else:
            st.error("Échec de l’export vidéo.")
            st.code(log or "Aucun log", language="bash")

# ---------- Audio MP3 / WAV ----------
with ong2:
    st.subheader("Exporter l’audio")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Extraire en MP3"):
            sortie = REP_SORTIE / f"{base_court}.mp3"
            cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
                   "-i", str(video_base),
                   "-vn", "-c:a", "libmp3lame", "-b:a", "128k",
                   str(sortie)]
            ok, log = _run(cmd)
            if ok and sortie.exists():
                data = sortie.read_bytes()
                st.success(f"Fichier prêt : {sortie.name}")
                st.audio(data)
                st.download_button("Télécharger le MP3", data=data, file_name=sortie.name, mime="audio/mpeg")
                with st.expander("Journal FFmpeg"):
                    st.code(log, language="bash")
            else:
                st.error("Échec extraction MP3.")
                st.code(log or "Aucun log", language="bash")
    with col2:
        if st.button("Extraire en WAV"):
            sortie = REP_SORTIE / f"{base_court}.wav"
            cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
                   "-i", str(video_base),
                   "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                   str(sortie)]
            ok, log = _run(cmd)
            if ok and sortie.exists():
                data = sortie.read_bytes()
                st.success(f"Fichier prêt : {sortie.name}")
                st.audio(data)
                st.download_button("Télécharger le WAV", data=data, file_name=sortie.name, mime="audio/wav")
                with st.expander("Journal FFmpeg"):
                    st.code(log, language="bash")
            else:
                st.error("Échec extraction WAV.")
                st.code(log or "Aucun log", language="bash")

# ---------- Images fixes ----------
with ong3:
    st.subheader("Extraire des images")
    c1, c2, c3 = st.columns(3)
    with c1:
        mode = st.selectbox("Fréquence", ["1 image/s", "25 images/s", "N images/s"], index=0)
    with c2:
        n_fps = st.number_input("N images/s (si mode N)", min_value=1, max_value=120, value=10, step=1)
    with c3:
        largeur = st.radio("Définition des images", ["Basse (1280p)", "HD (1920p)"], index=0, horizontal=True)

    fps = 1 if mode == "1 image/s" else (25 if mode == "25 images/s" else int(n_fps))
    w = 1280 if largeur.startswith("Basse") else 1920

    dossier = (BASE_DIR / "images" / f"{base_court}").resolve()
    dossier.mkdir(parents=True, exist_ok=True)
    motif = str(dossier / "img_%06d.jpg")

    if st.button("Extraire les images"):
        cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
               "-i", str(video_base),
               "-vf", f"fps={fps},scale={w}:-2",
               "-q:v", "2",
               motif]
        ok, log = _run(cmd)
        if ok and any(dossier.glob("img_*.jpg")):
            st.success(f"Images enregistrées dans : {dossier}")
            st.code(f"Motif : {motif}")
            try:
                data = _zip_dir_as_bytes(dossier)
                st.download_button("Télécharger les images (zip)", data=data, file_name=f"{dossier.name}.zip", mime="application/zip")
            except Exception as e:
                st.info(f"Impossible de zipper automatiquement : {e}")
            with st.expander("Journal FFmpeg"):
                st.code(log, language="bash")
        else:
            st.error("Échec extraction d’images.")
            st.code(log or "Aucun log", language="bash")

# ---------- Extrait par intervalle ----------
with ong4:
    st.subheader("Exporter un extrait de la vidéo")
    c1, c2, c3 = st.columns(3)
    with c1:
        debut = st.text_input("Début (HH:MM:SS ou secondes)", value="00:00:05")
    with c2:
        fin = st.text_input("Fin (HH:MM:SS ou secondes)", value="00:00:25")
    with c3:
        qual = st.radio("Qualité de sortie", ["Copie directe (rapide)", "Basse def (1280p)", "HD"], index=0, horizontal=False)

    if st.button("Générer l’extrait"):
        t0 = _to_hhmmss(debut); t1 = _to_hhmmss(fin)
        sortie = REP_SORTIE / f"{base_court}_intervalle.mp4"
        if qual.startswith("Copie"):
            cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
                   "-ss", t0, "-to", t1,
                   "-i", str(video_base),
                   "-c", "copy",
                   "-movflags", "+faststart",
                   str(sortie)]
        elif "Basse" in qual:
            cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
                   "-ss", t0, "-to", t1,
                   "-i", str(video_base),
                   "-vf", "scale=1280:-2",
                   "-c:v", "libx264", "-preset", "slow", "-crf", "28", "-pix_fmt", "yuv420p",
                   "-c:a", "aac", "-b:a", "96k", "-ac", "2",
                   "-movflags", "+faststart",
                   str(sortie)]
        else:
            cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
                   "-ss", t0, "-to", t1,
                   "-i", str(video_base),
                   "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
                   "-c:a", "aac", "-b:a", "160k", "-ac", "2",
                   "-movflags", "+faststart",
                   str(sortie)]

        ok, log = _run(cmd)
        if ok and sortie.exists():
            data = sortie.read_bytes()
            st.success(f"Extrait prêt : {sortie.name}")
            st.video(data, format="video/mp4")
            st.download_button("Télécharger l’extrait", data=data, file_name=sortie.name, mime="video/mp4")
            with st.expander("Journal FFmpeg"):
                st.code(log, language="bash")
        else:
            st.error("Échec extraction d’intervalle.")
            st.code(log or "Aucun log", language="bash")
