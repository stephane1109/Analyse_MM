# core_media.py
# Fonctions utilitaires pour la page "Source & Préparation".
# Implémente : initialiser_repertoires, info_ffmpeg, afficher_message_cookies,
# preparer_depuis_url, preparer_depuis_fichier, SEUIL_APERCU_OCTETS.
# Toutes les fonctions et commentaires sont en français.

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

# Seuil d’aperçu en octets (par défaut ~80 Mo)
SEUIL_APERCU_OCTETS = 80 * 1024 * 1024

# ----------------- Utilitaires de base -----------------

def initialiser_repertoires() -> Tuple[Path, Path, Path]:
    """Crée l’arborescence de travail sous /tmp pour Streamlit Cloud et retourne (BASE_DIR, REP_SORTIE, REP_TMP)."""
    base_dir = Path("/tmp/appdata").resolve()
    rep_sortie = base_dir / "sortie"
    rep_tmp = base_dir / "tmp"
    for d in (base_dir, rep_sortie, rep_tmp):
        d.mkdir(parents=True, exist_ok=True)
    return base_dir, rep_sortie, rep_tmp

def _trouver_ffmpeg() -> Optional[str]:
    """Retourne le chemin de ffmpeg : ./bin/ffmpeg prioritaire, sinon /usr/bin/ffmpeg, sinon PATH."""
    local = Path("./bin/ffmpeg")
    if local.is_file() and os.access(str(local), os.X_OK):
        return str(local.resolve())
    for cand in ("/usr/bin/ffmpeg", shutil.which("ffmpeg")):
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return cand
    return None

def info_ffmpeg() -> Tuple[Optional[str], Optional[str]]:
    """Retourne (chemin_ffmpeg, ligne_version_1) ou (None, None) si introuvable."""
    ff = _trouver_ffmpeg()
    if not ff:
        return None, None
    try:
        out = subprocess.run([ff, "-version"], capture_output=True, text=True, check=True).stdout.strip()
        first = out.splitlines()[0] if out else ""
        return ff, first
    except Exception:
        return ff, None

def _run(cmd: list) -> Tuple[bool, str]:
    """Exécute une commande, capture stdout/stderr, retourne (ok, log)."""
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

def _format_hhmmss_or_seconds(val: float) -> str:
    """Formate un nombre de secondes en HH:MM:SS."""
    s = max(0, float(val))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"

# ----------------- Gestion des cookies -----------------

def afficher_message_cookies(rep_sortie: Path) -> Optional[str]:
    """Affiche un uploader de cookies.txt et retourne le chemin où il est enregistré, ou None.
    Cette fonction ne dépend pas de Streamlit directement pour rester testable ; le main l’utilise comme une API simple.
    Ici, on se contente d'exposer une convention de chemin. Le main gère l'uploader et passe le fichier ? Non :
    Pour rester fidèle à l’appel existant, on implémente un mécanisme simple :
    - Si un fichier 'cookies.txt' existe déjà dans rep_sortie, on retourne son chemin.
    - Sinon, on ne crée rien et retourne None. L'utilisateur peut déposer le fichier via l'interface de la page dédiée si besoin.
    """
    p = rep_sortie / "cookies.txt"
    return str(p) if p.exists() else None

# ----------------- Préparation à partir d'un fichier local -----------------

def preparer_depuis_fichier(
    chemin_local: Path,
    nom_base: str,
    qualite: str,
    utiliser_intervalle: bool,
    debut_secs: int,
    fin_secs: int,
) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """Prépare la vidéo à partir d’un fichier local.
    Retourne (True, (chemin_video_base, nom_base_court)) ou (False, message_erreur)."""
    ff = _trouver_ffmpeg()
    if not ff:
        return False, "FFmpeg introuvable."

    # Répertoires de sortie
    base_dir, rep_sortie, rep_tmp = initialiser_repertoires()

    # Copie dans le tmp pour traitement
    src = Path(chemin_local)
    if not src.exists():
        return False, f"Fichier local introuvable : {src}"

    travail = rep_tmp / f"{nom_base}.mp4"
    try:
        shutil.copyfile(src, travail)
    except Exception as e:
        return False, f"Impossible de copier le fichier local : {e}"

    # Encodage selon la qualité demandée et, si besoin, l’intervalle
    out_path = rep_sortie / f"{nom_base}_base.mp4"
    if utiliser_intervalle:
        t0 = _format_hhmmss_or_seconds(debut_secs)
        t1 = _format_hhmmss_or_seconds(fin_secs)
    else:
        t0, t1 = None, None

    if qualite.startswith("Compressée"):
        # 1280px, CRF 28
        filtre = ["-vf", "scale=1280:-2"]
        video_args = ["-c:v", "libx264", "-crf", "28", "-preset", "slow", "-pix_fmt", "yuv420p"]
        audio_args = ["-c:a", "aac", "-b:a", "96k", "-ac", "2"]
    else:
        # HD max disponible (copie vidéo si déjà H.264, sinon ré-encode soft)
        # Pour rester simple et robuste, on ré-encode léger en libx264 CRF 20 si pas d’intervalle,
        # et CRF 23 si intervalle pour limiter la taille.
        filtre = []
        video_args = ["-c:v", "libx264", "-crf", "20", "-preset", "slow", "-pix_fmt", "yuv420p"]
        audio_args = ["-c:a", "aac", "-b:a", "160k", "-ac", "2"]

    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error"]
    if t0 and t1:
        cmd += ["-ss", t0, "-to", t1]
    cmd += ["-i", str(travail)]
    cmd += filtre + video_args + audio_args + ["-movflags", "+faststart", str(out_path)]

    ok, log = _run(cmd)
    if not ok:
        return False, f"Préparation échouée (local) :\n{log}"

    return True, (str(out_path), nom_base)

# ----------------- Préparation à partir d’une URL YouTube -----------------

def _trouver_ytdlp() -> Optional[str]:
    """Trouve yt-dlp (recommandé) ou youtube-dl. Retourne le chemin ou None."""
    for nom in ("yt-dlp", "youtube-dl"):
        p = shutil.which(nom)
        if p:
            return p
    # Copie locale éventuelle
    local = Path("./bin/yt-dlp")
    if local.is_file() and os.access(str(local), os.X_OK):
        return str(local.resolve())
    return None

def _telecharger_url(url: str, cookies_path: Optional[str], rep_tmp: Path, verbose: bool) -> Tuple[bool, Optional[Path], str]:
    """Télécharge une URL via yt-dlp/youtube-dl dans rep_tmp. Retourne (ok, chemin_fichier, log)."""
    outil = _trouver_ytdlp()
    if not outil:
        return False, None, "yt-dlp/youtube-dl introuvable. Ajoute-le aux dépendances ou fournis un binaire dans ./bin/yt-dlp."

    # On demande le meilleur mp4 possible (vidéo+audio), sinon mp4 converti.
    sortie = rep_tmp / "source.%(ext)s"
    cmd = [outil, "-f", "mp4/bestvideo+bestaudio/best", "-o", str(sortie), url]

    if cookies_path and Path(cookies_path).exists():
        cmd += ["--cookies", cookies_path]

    if not verbose:
        cmd += ["-q"]

    ok, log = _run(cmd)
    if not ok:
        return False, None, f"Téléchargement échoué :\n{log}"

    # Chercher le fichier téléchargé (mp4 prioritaire)
    cand_mp4 = list(rep_tmp.glob("source.mp4"))
    if cand_mp4:
        return True, cand_mp4[0], log
    # Sinon, prendre le premier fichier 'source.*' téléchargé
    cand_any = sorted(rep_tmp.glob("source.*"))
    if cand_any:
        return True, cand_any[0], log
    return False, None, "Téléchargement terminé mais fichier introuvable."

def preparer_depuis_url(
    url: str,
    cookies_path: Optional[str],
    qualite: str,
    verbose: bool,
    utiliser_intervalle: bool,
    debut_secs: int,
    fin_secs: int,
) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """Prépare la vidéo à partir d’une URL YouTube.
    Retourne (True, (chemin_video_base, nom_base_court)) ou (False, message_erreur)."""
    ff = _trouver_ffmpeg()
    if not ff:
        return False, "FFmpeg introuvable."

    base_dir, rep_sortie, rep_tmp = initialiser_repertoires()

    ok, chemin_dl, log = _telecharger_url(url, cookies_path, rep_tmp, verbose)
    if not ok or not chemin_dl:
        return False, log

    nom_base = "url_video"
    travail = rep_tmp / f"{nom_base}.mp4"
    try:
        # Si pas mp4, on remux/reencode en MP4 avant de poursuivre
        if chemin_dl.suffix.lower() != ".mp4":
            cmd_remux = [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(chemin_dl), "-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart", str(travail)]
            ok2, log2 = _run(cmd_remux)
            if not ok2:
                return False, f"Remux en MP4 échoué :\n{log2}"
        else:
            shutil.copyfile(chemin_dl, travail)
    except Exception as e:
        return False, f"Impossible de préparer le fichier téléchargé : {e}"

    out_path = rep_sortie / f"{nom_base}_base.mp4"
    if utiliser_intervalle:
        t0 = _format_hhmmss_or_seconds(debut_secs)
        t1 = _format_hhmmss_or_seconds(fin_secs)
    else:
        t0, t1 = None, None

    if qualite.startswith("Compressée"):
        filtre = ["-vf", "scale=1280:-2"]
        video_args = ["-c:v", "libx264", "-crf", "28", "-preset", "slow", "-pix_fmt", "yuv420p"]
        audio_args = ["-c:a", "aac", "-b:a", "96k", "-ac", "2"]
    else:
        filtre = []
        video_args = ["-c:v", "libx264", "-crf", "20", "-preset", "slow", "-pix_fmt", "yuv420p"]
        audio_args = ["-c:a", "aac", "-b:a", "160k", "-ac", "2"]

    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error"]
    if t0 and t1:
        cmd += ["-ss", t0, "-to", t1]
    cmd += ["-i", str(travail)]
    cmd += filtre + video_args + audio_args + ["-movflags", "+faststart", str(out_path)]

    ok3, log3 = _run(cmd)
    if not ok3:
        return False, f"Préparation échouée (URL) :\n{log3}"

    return True, (str(out_path), nom_base)
