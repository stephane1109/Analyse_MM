# _1_extraction.py
# Bibliothèque de fonctions d'extraction/encodage basées sur FFmpeg pour être utilisées dans main.py.
# Conçu pour Streamlit Cloud : pas d'installation système nécessaire si un binaire statique est fourni dans ./bin/ffmpeg.
# Toutes les fonctions et commentaires sont en français, les erreurs de FFmpeg sont capturées et renvoyées.

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional

# =========================================
# Utilitaires système
# =========================================

def assurer_dossier(chemin: str) -> None:
    """Crée un dossier s'il n'existe pas, de façon idempotente."""
    os.makedirs(chemin, exist_ok=True)

def trouver_ffmpeg() -> Optional[str]:
    """Retourne le chemin exécutable vers ffmpeg. Priorité à ./bin/ffmpeg, sinon /usr/bin/ffmpeg, sinon PATH."""
    local = os.path.abspath("./bin/ffmpeg")
    if os.path.isfile(local) and os.access(local, os.X_OK):
        return local
    for cand in ["/usr/bin/ffmpeg", shutil.which("ffmpeg")]:
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None

def _run_cmd(cmd: list) -> Tuple[bool, str]:
    """Exécute une commande et retourne (ok, log). Capture stdout et stderr, renvoie le texte le plus utile."""
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out = (res.stdout or "").strip()
        err = (res.stderr or "").strip()
        log = "\n".join([s for s in [out, err] if s]).strip()
        return True, log
    except subprocess.CalledProcessError as e:
        out = (e.stdout or "").strip()
        err = (e.stderr or "").strip()
        log = "\n".join([s for s in [out, err] if s]).strip()
        if not log:
            log = str(e)
        return False, log
    except Exception as e:
        return False, f"Erreur d'exécution : {e}"

def verifier_encodeurs_cles(ffmpeg: str) -> Tuple[Dict[str, bool], str]:
    """Vérifie la présence de libx264 et aac dans les encodeurs FFmpeg."""
    try:
        res = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=True)
        txt = res.stdout
    except Exception as e:
        return {"libx264": False, "aac": False}, f"Impossible de lister les encodeurs : {e}"

    rep = {"libx264": ("libx264" in txt), "aac": ("aac" in txt)}
    warn = ""
    if not rep["libx264"]:
        warn += "libx264 introuvable dans ce binaire FFmpeg. "
    if not rep["aac"]:
        warn += "aac introuvable dans ce binaire FFmpeg. "
    return rep, warn.strip()

# =========================================
# Fonctions d'encodage et d'extraction
# =========================================

def encoder_video_h264_aac(
    ffmpeg: str,
    chemin_entree: str,
    chemin_sortie: str,
    largeur: int = 1280,
    crf: int = 28,
    preset: str = "slow",
    audio_bitrate: str = "96k",
) -> Tuple[bool, str]:
    """Encode la vidéo en H.264 + AAC, redimensionnée à 'largeur', yuv420p, et faststart."""
    assurer_dossier(str(Path(chemin_sortie).parent))
    filtre_scale = f"scale={int(largeur)}:-2"
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", chemin_entree,
        "-vf", filtre_scale,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-ac", "2",
        "-movflags", "+faststart",
        chemin_sortie
    ]
    return _run_cmd(cmd)

def extraire_audio_mp3(
    ffmpeg: str,
    chemin_entree: str,
    chemin_sortie: str,
    bitrate_audio: str = "128k",
    mono: bool = False,
) -> Tuple[bool, str]:
    """Extrait l'audio en MP3 à partir de la vidéo source."""
    assurer_dossier(str(Path(chemin_sortie).parent))
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", chemin_entree,
    ]
    if mono:
        cmd += ["-ac", "1"]
    cmd += ["-vn", "-c:a", "libmp3lame", "-b:a", bitrate_audio, chemin_sortie]
    return _run_cmd(cmd)

def extraire_images_par_fps(
    ffmpeg: str,
    chemin_entree: str,
    dossier_sortie: str,
    fps: int = 1,
    largeur: int = 1280,
) -> Tuple[bool, str, str]:
    """Extrait des images fixes à fréquence donnée en i/s, redimensionnées à la largeur souhaitée."""
    assurer_dossier(dossier_sortie)
    motif = os.path.join(dossier_sortie, "img_%06d.jpg")
    filtre = f"fps={int(fps)},scale={int(largeur)}:-2"
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", chemin_entree,
        "-vf", filtre,
        "-q:v", "2",
        motif
    ]
    ok, log = _run_cmd(cmd)
    return ok, log, motif

def _parse_temps(t: str) -> str:
    """Accepte soit 'HH:MM:SS' soit un nombre de secondes, et retourne un format accepté par FFmpeg."""
    t = str(t).strip()
    if t.count(":") in (1, 2):
        return t
    try:
        # seconds -> HH:MM:SS
        s = float(t)
        s = max(0, s)
        heures = int(s // 3600)
        minutes = int((s % 3600) // 60)
        secondes = int(s % 60)
        return f"{heures:02d}:{minutes:02d}:{secondes:02d}"
    except Exception:
        return t  # laisser tel quel, ffmpeg renverra une erreur si invalide

def extraire_images_intervalle(
    ffmpeg: str,
    chemin_entree: str,
    chemin_sortie: str,
    debut: str,
    fin: str,
    copy: bool = False,
    largeur: int = 1280,
    crf: int = 28,
) -> Tuple[bool, str]:
    """Extrait un intervalle de la vidéo, du temps 'debut' à 'fin' inclusivement approximatif. Peut copier sans ré-encodage ou ré-encoder."""
    assurer_dossier(str(Path(chemin_sortie).parent))
    t0 = _parse_temps(debut)
    t1 = _parse_temps(fin)

    if copy:
        # Copie sans ré-encodage. Plus rapide, mais uniquement aux frontières de keyframes.
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", t0, "-to", t1,
            "-i", chemin_entree,
            "-c", "copy",
            "-movflags", "+faststart",
            chemin_sortie
        ]
        return _run_cmd(cmd)
    else:
        # Ré-encodage avec redimensionnement et H.264/AAC.
        filtre_scale = f"scale={int(largeur)}:-2"
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", t0, "-to", t1,
            "-i", chemin_entree,
            "-vf", filtre_scale,
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", "slow",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-ac", "2",
            "-movflags", "+faststart",
            chemin_sortie
        ]
        return _run_cmd(cmd)

def composer_timelapse_depuis_images(
    ffmpeg: str,
    dossier_images: str,
    motif: str = "img_%06d.jpg",
    fps: int = 10,
    largeur: int = 1280,
    crf: int = 28,
) -> Tuple[bool, str]:
    """Compose une vidéo timelapse depuis une séquence d'images numérotées. Le motif utilise la numérotation printf."""
    dossier = Path(dossier_images)
    if not dossier.is_dir():
        return False, f"Dossier images introuvable : {dossier_images}"

    # Sortie à côté du dossier images si non précisé par l'appelant
    sortie = str(dossier.parent / "timelapse.mp4")

    # -framerate définit la cadence d'entrée des images; on peut garder fps aussi en sortie si besoin.
    filtre_scale = f"scale={int(largeur)}:-2"
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", str(int(fps)),
        "-i", os.path.join(dossier_images, motif),
        "-vf", filtre_scale,
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        sortie
    ]
    ok, log = _run_cmd(cmd)
    return ok, log
