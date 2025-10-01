# core_media.py
# Fonctions utilitaires pour la page "Source & Préparation".
# Implémente : initialiser_repertoires, info_ffmpeg, afficher_message_cookies (avec uploader),
# preparer_depuis_url, preparer_depuis_fichier, SEUIL_APERCU_OCTETS.
# Ajout : _telecharger_url gère les cas SABR en testant plusieurs clients YouTube (android/ios/tv/web),
#         une sélection de formats explicite mp4, et un parallélisme réduit pour éviter les fragments vides.
# Toutes les fonctions et commentaires sont en français.

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st

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

# ----------------- Cookies : uploader intégré -----------------

def afficher_message_cookies(rep_sortie: Path) -> Optional[str]:
    """Affiche un uploader Streamlit pour déposer un fichier cookies.txt (optionnel).
    - Si un fichier est déposé, il est sauvegardé sous rep_sortie / 'cookies.txt' et son chemin est renvoyé.
    - Si aucun fichier n’est déposé mais qu’un cookies.txt existe déjà, on renvoie ce chemin.
    - Sinon, on renvoie None.
    """
    st.markdown("### Cookies (optionnel)")
    st.caption("Pour les vidéos restreintes (403), exporte tes cookies depuis Firefox avec l’extension cookies.txt, puis importe le fichier ici.")
    cookies_file = st.file_uploader("Importer cookies.txt", type=["txt"], key="cookies_uploader")

    cible = rep_sortie / "cookies.txt"
    if cookies_file is not None:
        try:
            rep_sortie.mkdir(parents=True, exist_ok=True)
            with open(cible, "wb") as f:
                f.write(cookies_file.read())
            st.success(f"Fichier cookies enregistré : {cible}")
            return str(cible)
        except Exception as e:
            st.error(f"Impossible d’enregistrer cookies.txt : {e}")
            return None

    if cible.exists():
        st.info(f"Un fichier cookies.txt existe déjà : {cible}")
        return str(cible)

    st.caption("Aucun fichier cookies.txt importé.")
    return None

# ----------------- yt-dlp / youtube-dl -----------------

def _trouver_ytdlp() -> Optional[str]:
    """Trouve yt-dlp (recommandé) ou youtube-dl. Retourne le chemin ou None."""
    for nom in ("yt-dlp", "youtube-dl"):
        p = shutil.which(nom)
        if p:
            return p
    local = Path("./bin/yt-dlp")
    if local.is_file() and os.access(str(local), os.X_OK):
        return str(local.resolve())
    return None

def _fich_dl_vraiment_non_vide(rep_tmp: Path) -> Optional[Path]:
    """Retourne un fichier téléchargé 'source.*' non vide s'il existe, priorité au .mp4."""
    cand_mp4 = list(rep_tmp.glob("source.mp4"))
    for p in cand_mp4:
        if p.stat().st_size > 0:
            return p
    cand_any = sorted(rep_tmp.glob("source.*"))
    for p in cand_any:
        if p.stat().st_size > 0:
            return p
    return None

def _telecharger_url(url: str, cookies_path: Optional[str], rep_tmp: Path, verbose: bool) -> Tuple[bool, Optional[Path], str]:
    """Télécharge une URL via yt-dlp/youtube-dl dans rep_tmp. Gère les cas SABR en testant plusieurs stratégies.
    Retourne (ok, chemin_fichier, log détaillé)."""

    outil = _trouver_ytdlp()
    if not outil:
        return False, None, "yt-dlp/youtube-dl introuvable. Ajoute-le aux dépendances ou fournis un binaire dans ./bin/yt-dlp."

    rep_tmp.mkdir(parents=True, exist_ok=True)
    sortie = rep_tmp / "source.%(ext)s"

    # Sélection de format prioritaire mp4 ; fallback sur best si besoin
    base_fmt = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best"

    # UA « desktop » explicite pour écarter certains profils à pub serveur
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

    # Variantes à essayer pour contourner SABR / formats sans URL
    variantes = [
        ["-f", base_fmt],
        ["-f", base_fmt, "--concurrent-fragments", "1"],
        ["-f", base_fmt, "--extractor-args", "youtube:player_client=android", "--concurrent-fragments", "1"],
        ["-f", base_fmt, "--extractor-args", "youtube:player_client=ios", "--concurrent-fragments", "1"],
        ["-f", base_fmt, "--extractor-args", "youtube:player_client=tv", "--concurrent-fragments", "1"],
        ["-f", base_fmt, "--user-agent", ua, "--add-header", "Accept-Language:en-US,en;q=0.9", "--concurrent-fragments", "1"],
    ]

    logs_cumul = []
    for idx, extra in enumerate(variantes, start=1):
        cmd = [outil, "-o", str(sortie)]
        cmd += extra
        # Merge/remux en mp4 si possible via ffmpeg
        cmd += ["--merge-output-format", "mp4", "--restrict-filenames", "--ignore-no-formats-error"]
        # Cookies si fournis
        if cookies_path and Path(cookies_path).exists():
            cmd += ["--cookies", cookies_path]
        # Verbosité
        if not verbose:
            cmd += ["-q"]
        cmd += [url]

        ok, log = _run(cmd)
        logs_cumul.append(f"\n=== Tentative {idx}/{len(variantes)} ===\n$ {' '.join(cmd)}\n{log}\n")

        f = _fich_dl_vraiment_non_vide(rep_tmp)
        if ok and f is not None and f.exists() and f.stat().st_size > 0:
            return True, f, "".join(logs_cumul)

        # Nettoyage entre tentatives si fichier vide créé
        for p in rep_tmp.glob("source.*"):
            try:
                if p.is_file() and p.stat().st_size == 0:
                    p.unlink(missing_ok=True)
            except Exception:
                pass

    return False, None, "Téléchargement échoué après plusieurs stratégies.\n" + "".join(logs_cumul)

# ----------------- Préparation à partir d’un fichier local -----------------

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

    base_dir, rep_sortie, rep_tmp = initialiser_repertoires()

    src = Path(chemin_local)
    if not src.exists():
        return False, f"Fichier local introuvable : {src}"

    travail = rep_tmp / f"{nom_base}.mp4"
    try:
        shutil.copyfile(src, travail)
    except Exception as e:
        return False, f"Impossible de copier le fichier local : {e}"

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

    ok, log = _run(cmd)
    if not ok:
        return False, f"Préparation échouée (local) :\n{log}"

    return True, (str(out_path), nom_base)

# ----------------- Préparation à partir d’une URL YouTube -----------------

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

    ok, chemin_dl, log_dl = _telecharger_url(url, cookies_path, rep_tmp, verbose)
    if not ok or not chemin_dl:
        return False, f"Téléchargement échoué : {log_dl}"

    nom_base = "url_video"
    travail = rep_tmp / f"{nom_base}.mp4"
    try:
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
        return False, f"Préparation échouée (URL) :\n{log3}\n\nLog téléchargement :\n{log_dl}"

    return True, (str(out_path), "url_video")
