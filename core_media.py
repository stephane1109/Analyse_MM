# core_media.py
# Utilitaires partagés : répertoires, ffmpeg, yt-dlp, préparation vidéo, extraction, zip.

import os
from pathlib import Path
import unicodedata, re, shutil, glob, zipfile, subprocess
import cv2
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

SEUIL_APERCU_OCTETS = 160 * 1024 * 1024
LONGUEUR_TITRE_MAX = 24
LONGUEUR_PREFIX_ID = 8

# -------- Répertoires --------
def initialiser_repertoires():
    base = Path("/tmp/appdata")
    rep_sortie = base / "fichiers"
    rep_tmp = base / "tmp"
    rep_sortie.mkdir(parents=True, exist_ok=True)
    rep_tmp.mkdir(parents=True, exist_ok=True)
    return base, rep_sortie, rep_tmp

# -------- FFmpeg --------
def info_ffmpeg():
    try:
        # timelapse.chemin_ffmpeg est robuste, mais on évite l’import croisé ici
        from timelapse import chemin_ffmpeg
        p = chemin_ffmpeg()
        ver = subprocess.run([p, "-version"], capture_output=True, text=True, check=False)
        line1 = ver.stdout.splitlines()[0] if ver.stdout else ""
        return p, line1
    except Exception:
        return None, None

# -------- Cookies UI --------
import streamlit as st

def afficher_message_cookies(rep_sortie: Path):
    st.markdown(
        "Si la vidéo est restreinte (403), exportez vos cookies avec l’extension Firefox : "
        "[cookies.txt](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/)."
    )
    up = st.file_uploader("Importer un fichier cookies.txt (optionnel)", type=["txt"])
    if up:
        dest = rep_sortie / "cookies.txt"
        with open(dest, "wb") as f:
            f.write(up.read())
        st.success("Cookies chargés.")
        return str(dest)
    return None

# -------- Noms sûrs --------
def _nettoyer_titre(titre: str) -> str:
    if not titre:
        titre = "video"
    titre = titre.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    rem = {'«':'','»':'','“':'','”':'','’':'','‘':'','„':'','"':'',"'":'',
           ':':'-','/':'-','\\':'-','|':'-','?':'','*':'','<':'','>':'','\u00A0':' '}
    for k, v in rem.items():
        titre = titre.replace(k, v)
    titre = unicodedata.normalize('NFKD', titre)
    titre = ''.join(c for c in titre if not unicodedata.combining(c))
    titre = re.sub(r'[^\w\s-]', '', titre)
    titre = re.sub(r'\s+', '_', titre.strip())
    return (titre or "video")[:LONGUEUR_TITRE_MAX]

def _nom_base(video_id: str, titre: str) -> str:
    vid = (video_id or "vid")[:LONGUEUR_PREFIX_ID]
    tit = _nettoyer_titre(titre)
    return f"{vid}_{tit}"

def _renommer_unique(src: Path, dest_base: Path, ext: str) -> Path:
    cand = Path(f"{dest_base}{ext}")
    i = 1
    while cand.exists():
        cand = Path(f"{dest_base}_{i}{ext}")
        i += 1
    shutil.move(str(src), str(cand))
    return cand

# -------- Préparation vidéo --------
def _telecharger_ytdlp(url: str, cookies_path: str|None, verbose: bool, sections: list|None, rep_sortie: Path):
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0"
    http_headers = {'User-Agent': user_agent, 'Accept':'*/*','Accept-Language':'en-US,en;q=0.5','Referer':'https://www.youtube.com/'}
    base_opts = {
        'paths': {'home': str(rep_sortie)},
        'outtmpl': {'default': '%(id)s.%(ext)s'},
        'noplaylist': True,
        'quiet': not verbose,
        'no_warnings': not verbose,
        'merge_output_format': 'mp4',
        'retries': 10,
        'fragment_retries': 10,
        'continuedl': True,
        'concurrent_fragment_downloads': 1,
        'http_headers': http_headers,
        'geo_bypass': True,
        'nocheckcertificate': True,
        'restrictfilenames': True,
        'trim_file_name': 80,
        'extractor_args': {'youtube': {'player_client': ['android','ios','mweb','web']}},
    }
    if sections:
        base_opts['download_sections'] = sections
        base_opts['force_keyframes_at_cuts'] = True
    if cookies_path:
        base_opts['cookiefile'] = cookies_path

    fallbacks = ["bv*[ext=mp4][height<=2160]+ba[ext=m4a]/b[ext=mp4]/b", "bv*+ba/b"]
    derniere = None
    info, out = None, None
    for fmt in fallbacks:
        opts = base_opts.copy(); opts['format'] = fmt
        try:
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                _ = ydl.prepare_filename(info)
            cand = []
            for ext in ['mp4', 'mkv', 'webm', 'm4a', 'mp3']:
                cand.extend(Path(rep_sortie).glob(f"*.{ext}"))
            if not cand:
                raise DownloadError("Téléchargement terminé mais aucun fichier détecté (download is empty).")
            cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            out = cand[0]
            break
        except Exception as e:
            derniere = e
            msg = str(e) or repr(e)
            if "403" in msg or "Forbidden" in msg:
                if not cookies_path:
                    raise RuntimeError("HTTP 403 : fournissez un cookies.txt puis relancez.")
                raise RuntimeError("HTTP 403 persistant : cookies.txt invalide ou expiré.")
            continue
    if out is None:
        raise RuntimeError(str(derniere) if derniere else "Echec inconnu au téléchargement.")
    return info, out

def _ffmpeg_path():
    from timelapse import chemin_ffmpeg
    return chemin_ffmpeg()

def _transcoder(src: Path, dst: Path, compress: bool, interval: tuple[int,int]|None):
    ffmpeg = _ffmpeg_path()
    args = [ffmpeg, "-y"]
    if interval:
        args += ["-ss", str(interval[0]), "-to", str(interval[1])]
    args += ["-i", str(src)]
    if compress:
        args += ["-vf", "scale=1280:-2", "-c:v", "libx264", "-preset", "slow", "-crf", "28",
                 "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(dst)]
    else:
        try:
            args_copy = args + ["-c", "copy", "-movflags", "+faststart", str(dst)]
            subprocess.run(args_copy, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception:
            args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                     "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(dst)]
            subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def preparer_depuis_url(url: str, cookies_path: str|None, qualite: str, verbose: bool,
                        use_interval: bool, debut: int, fin: int):
    base, rep_sortie, _ = initialiser_repertoires()
    sections = [{'section': f"*{debut}-{fin}"}] if use_interval else None
    try:
        info, fichier = _telecharger_ytdlp(url, cookies_path, verbose, sections, rep_sortie)
    except Exception as e:
        return False, f"Echec téléchargement : {e}"
    video_id = (info.get('id') if info else "vid") or "vid"
    titre = (info.get('title') if info else fichier.stem) or "video"
    base_court = _nom_base(video_id, titre)
    src_propre = _renommer_unique(fichier, rep_sortie / f"{base_court}_src", fichier.suffix)
    dst = rep_sortie / f"{base_court}_video.mp4"
    try:
        _transcoder(src_propre, dst, qualite.startswith("Compressée"), (debut, fin) if use_interval else None)
    except Exception as e:
        return False, f"Echec préparation vidéo : {e}"
    try:
        if src_propre.exists():
            src_propre.unlink()
    except Exception:
        pass
    return True, (str(dst), base_court)

def preparer_depuis_fichier(f_local: Path, base_nom_local: str, qualite: str,
                            use_interval: bool, debut: int, fin: int):
    _, rep_sortie, _ = initialiser_repertoires()
    base_court = _nom_base("local", base_nom_local)
    dst = rep_sortie / f"{base_court}_video.mp4"
    try:
        _transcoder(f_local, dst, qualite.startswith("Compressée"), (debut, fin) if use_interval else None)
    except Exception as e:
        return False, f"Echec préparation (local) : {e}"
    return True, (str(dst), base_court)

# -------- Extraction ressources --------
def _cmds_extraction(video_path: str, debut: int, fin: int, base_court: str, options: dict, interval: bool, rep_sortie: Path):
    ffmpeg = _ffmpeg_path()
    def run(args):
        subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    def seg_cmd(outp: Path):
        if interval:
            return [ffmpeg,"-y","-ss",str(debut),"-to",str(fin),"-i",video_path,
                    "-vf","scale=1280:-2","-c:v","libx264","-preset","slow","-crf","28",
                    "-c:a","aac","-b:a","96k","-movflags","+faststart",str(outp)]
        else:
            return [ffmpeg,"-y","-i",video_path,
                    "-vf","scale=1280:-2","-c:v","libx264","-preset","slow","-crf","28",
                    "-c:a","aac","-b:a","96k","-movflags","+faststart",str(outp)]

    def aud_cmd(outp: Path, codec):
        base = [ffmpeg,"-y"]
        if interval:
            base += ["-ss",str(debut),"-to",str(fin)]
        base += ["-i",video_path] + codec + ["-movflags","+faststart",str(outp)]
        return base

    def img_cmd(pattern: str, fps: int):
        vf = f"fps={fps},scale=1920:1080"
        if interval:
            return [ffmpeg,"-y","-ss",str(debut),"-to",str(fin),"-i",video_path,"-vf",vf,"-q:v","1",pattern]
        else:
            return [ffmpeg,"-y","-i",video_path,"-vf",vf,"-q:v","1",pattern]

    produits = []

    if options.get("mp4"):
        nom = f"{base_court}_seg.mp4" if interval else f"{base_court}_full.mp4"
        p = (rep_sortie / nom)
        run(seg_cmd(p)); produits.append(p)

    if options.get("mp3"):
        nom = f"{base_court}_seg.mp3" if interval else f"{base_court}_full.mp3"
        p = (rep_sortie / nom)
        run(aud_cmd(p, ["-vn","-acodec","libmp3lame","-q:a","5"])); produits.append(p)

    if options.get("wav"):
        nom = f"{base_court}_seg.wav" if interval else f"{base_court}_full.wav"
        p = (rep_sortie / nom)
        run(aud_cmd(p, ["-vn","-acodec","adpcm_ima_wav"])); produits.append(p)

    if options.get("img1") or options.get("img25"):
        for fps in [1, 25]:
            if (fps == 1 and options.get("img1")) or (fps == 25 and options.get("img25")):
                dossier = f"img{fps}_{base_court}" if interval else f"img{fps}_full_{base_court}"
                rep = rep_sortie / dossier
                rep.mkdir(parents=True, exist_ok=True)
                tmp_pattern = str(rep / "tmp_%06d.jpg")
                run(img_cmd(tmp_pattern, fps))
                images = sorted(rep.glob("tmp_*.jpg"))
                start_offset = debut if interval else 0
                for i, src in enumerate(images):
                    t = start_offset + (i / float(fps))
                    sec = int(t)
                    if fps == 1:
                        nom_img = f"i_{sec}s_1fps.jpg"
                    else:
                        f_in_s = int(round((t - sec) * fps))
                        if f_in_s >= fps:
                            f_in_s = fps - 1
                        nom_img = f"i_{sec}s_{fps}fps_{f_in_s:02d}.jpg"
                    dst = rep / nom_img
                    j = 1
                    base_dst = dst.with_suffix("")
                    ext = dst.suffix
                    while dst.exists():
                        dst = Path(f"{base_dst}_{j}{ext}"); j += 1
                    os.replace(str(src), str(dst))
                produits.append(rep)

    return produits

def extraire_ressources(video_path: str, debut: int, fin: int, base_court: str, options: dict, interval: bool):
    try:
        _, rep_sortie, _ = initialiser_repertoires()
        return True, _cmds_extraction(video_path, debut, fin, base_court, options, interval, rep_sortie)
    except Exception as e:
        return False, f"Echec extraction : {e}"

def zipper(fichiers: list[Path], zip_path: Path):
    with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in fichiers:
            f = Path(f)
            if f.is_dir():
                for sub in f.rglob("*"):
                    if sub.is_file():
                        zf.write(str(sub), arcname=str(Path(f.name) / sub.relative_to(f)))
            elif f.is_file():
                zf.write(str(f), arcname=f.name)
    return zip_path
