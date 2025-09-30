# main.py
# Application Streamlit pour extraction vidéo/son/images et timelapse via FFmpeg.
# Toutes les fonctions et commentaires sont en français. Compatible Streamlit Cloud.

import os
import sys
import time
import importlib
import importlib.util
from pathlib import Path
from typing import Optional

import streamlit as st

# =========================================
# Chargement robuste de extraction.py (racine OU pages/extraction.py)
# =========================================

def _charger_module_extraction():
    """Charge le module extraction, même s'il est placé dans ./pages/extraction.py.
    Ordre des tentatives :
      1) import extraction
      2) import pages.extraction
      3) ajout de chemins au sys.path puis 1) et 2) à nouveau
      4) chargement direct par chemin : ./extraction.py puis ./pages/extraction.py
    """
    # Tentative 1 : import direct
    try:
        return importlib.import_module("extraction")
    except ModuleNotFoundError:
        pass

    # Tentative 2 : import sous-package pages
    try:
        return importlib.import_module("pages.extraction")
    except ModuleNotFoundError:
        pass

    # Tentative 3 : forcer les chemins
    racine = Path(__file__).parent.resolve()
    pages_dir = racine / "pages"

    chemins_a_ajouter = [str(racine)]
    if pages_dir.exists():
        chemins_a_ajouter.append(str(pages_dir))

    for p in chemins_a_ajouter:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Réessayer l'import après injection des chemins
    for nom in ("extraction", "pages.extraction"):
        try:
            return importlib.import_module(nom)
        except ModuleNotFoundError:
            continue

    # Tentative 4 : chargement direct par chemin de fichier
    candidats = [racine / "extraction.py", pages_dir / "extraction.py"]
    for chemin in candidats:
        if chemin.exists():
            spec = importlib.util.spec_from_file_location("extraction", str(chemin))
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            sys.modules["extraction"] = module
            return module

    return None

_mod = _charger_module_extraction()
if _mod is None:
    st.set_page_config(page_title="Extraction vidéo/son/images et timelapse", layout="wide")
    st.title("Erreur d’import du module extraction.py")
    st.error(
        "Impossible de charger le module local 'extraction.py'.\n"
        "- Si ton fichier est dans 'pages/extraction.py', laisse-le là (c’est géré).\n"
        "- Sinon, place 'extraction.py' à côté de 'main.py'.\n"
        "Vérifie aussi la casse exacte du nom de fichier."
    )
    st.stop()

# Import des symboles depuis le module chargé
trouver_ffmpeg                   = _mod.trouver_ffmpeg
verifier_encodeurs_cles          = _mod.verifier_encodeurs_cles
assurer_dossier                  = _mod.assurer_dossier
encoder_video_h264_aac           = _mod.encoder_video_h264_aac
extraire_audio_mp3               = _mod.extraire_audio_mp3
extraire_images_par_fps          = _mod.extraire_images_par_fps
extraire_images_intervalle       = _mod.extraire_images_intervalle
composer_timelapse_depuis_images = _mod.composer_timelapse_depuis_images

# =========================================
# Configuration de l'application
# =========================================

st.set_page_config(page_title="Extraction vidéo, audio, images et timelapse", layout="wide")

BASE_DIR = Path("/tmp/appdata")
FICHIERS_DIR = BASE_DIR / "fichiers"
IMAGES_DIR = BASE_DIR / "images"
TIMELAPSE_DIR = BASE_DIR / "timelapse"

assurer_dossier(str(FICHIERS_DIR))
assurer_dossier(str(IMAGES_DIR))
assurer_dossier(str(TIMELAPSE_DIR))

# =========================================
# Outils UI
# =========================================

def nom_sans_extension(nom_fichier: str) -> str:
    """Retourne le nom de fichier sans extension, nettoyé des caractères spéciaux."""
    base = Path(nom_fichier).stem
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base)

def afficher_logs_taches(titre: str, ok: bool, log: str, chemin_sortie: Optional[str] = None):
    """Affiche un bloc de résultat standardisé avec le log détaillé."""
    if ok:
        st.success(f"{titre} terminé.")
        if chemin_sortie and os.path.isfile(chemin_sortie):
            st.caption(f"Fichier généré : {chemin_sortie}")
        if log:
            with st.expander("Journal FFmpeg (stdout/stderr)"):
                st.code(log, language="bash")
    else:
        st.error(f"{titre} en échec.")
        if log:
            st.code(log, language="bash")

def lire_fichier_binaire(chemin: str) -> bytes:
    """Lit un fichier binaire en mémoire pour affichage ou téléchargement."""
    with open(chemin, "rb") as f:
        return f.read()

# =========================================
# Corps de l'app
# =========================================

st.title("Extraction vidéo, audio, images et timelapse (FFmpeg)")

ffmpeg_path = trouver_ffmpeg()
if not ffmpeg_path:
    st.error("FFmpeg introuvable. Dépose un binaire statique dans ./bin/ffmpeg ou vérifie /usr/bin/ffmpeg.")
    st.stop()

st.caption(f"FFmpeg utilisé : {ffmpeg_path}")

# Diagnostic rapide des encodeurs
encod_ok, encod_warn = verifier_encodeurs_cles(ffmpeg_path)
col_a, col_b = st.columns(2)
with col_a:
    st.write("Encodeurs requis détectés")
    st.json({"libx264": encod_ok.get("libx264", False), "aac": encod_ok.get("aac", False)})
with col_b:
    if encod_warn:
        st.info(encod_warn)

# Téléversement de la vidéo source
fichier = st.file_uploader("Déposer une vidéo (mp4, mov, m4v, mkv…)", type=["mp4", "mov", "m4v", "mkv"])
if fichier is not None:
    base_name = nom_sans_extension(fichier.name)
    src_path = str(FICHIERS_DIR / f"{base_name}.mp4")
    with open(src_path, "wb") as f:
        f.write(fichier.read())
    st.success(f"Fichier importé : {src_path}")

    # Prévisualisation
    with open(src_path, "rb") as vf:
        st.video(vf.read())

    onglet1, onglet2, onglet3, onglet4, onglet5 = st.tabs([
        "Encodage MP4 (H.264 + AAC)",
        "Extraction audio MP3",
        "Extraction images (1 fps / 25 fps / N fps)",
        "Extraction par intervalle",
        "Recomposition timelapse",
    ])

    # Encodage vidéo standard
    with onglet1:
        st.subheader("Encodage et normalisation de la vidéo")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            largeur = st.number_input("Largeur cible (px)", min_value=320, max_value=3840, value=1280, step=16)
        with col2:
            crf = st.slider("CRF (qualité, plus bas = meilleure)", min_value=18, max_value=35, value=28)
        with col3:
            preset = st.selectbox("Preset x264", ["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"], index=6)
        with col4:
            audio_kbps = st.selectbox("Débit audio", ["64k","96k","128k","160k","192k"], index=1)

        if st.button("Lancer l'encodage", type="primary"):
            dst_path = str(FICHIERS_DIR / f"{base_name}_full.mp4")
            ok, log = encoder_video_h264_aac(
                ffmpeg=ffmpeg_path,
                chemin_entree=src_path,
                chemin_sortie=dst_path,
                largeur=int(largeur),
                crf=int(crf),
                preset=preset,
                audio_bitrate=audio_kbps,
            )
            afficher_logs_taches("Encodage H.264/AAC", ok, log, dst_path)
            if ok and os.path.isfile(dst_path):
                st.video(lire_fichier_binaire(dst_path))
                st.download_button("Télécharger la vidéo encodée", data=lire_fichier_binaire(dst_path), file_name=os.path.basename(dst_path), mime="video/mp4")

    # Extraction audio MP3
    with onglet2:
        st.subheader("Extraction audio MP3")
        col1, col2 = st.columns(2)
        with col1:
            mp3_bitrate = st.selectbox("Débit MP3", ["96k", "128k", "160k", "192k", "256k"], index=1)
        with col2:
            forcer_mono = st.checkbox("Forcer mono", value=False)

        if st.button("Extraire MP3"):
            mp3_path = str(FICHIERS_DIR / f"{base_name}.mp3")
            ok, log = extraire_audio_mp3(
                ffmpeg=ffmpeg_path,
                chemin_entree=src_path,
                chemin_sortie=mp3_path,
                bitrate_audio=mp3_bitrate,
                mono=forcer_mono,
            )
            afficher_logs_taches("Extraction audio MP3", ok, log, mp3_path)
            if ok and os.path.isfile(mp3_path):
                st.audio(lire_fichier_binaire(mp3_path))
                st.download_button("Télécharger le MP3", data=lire_fichier_binaire(mp3_path), file_name=os.path.basename(mp3_path), mime="audio/mpeg")

    # Extraction d'images
    with onglet3:
        st.subheader("Extraction d'images")
        col1, col2, col3 = st.columns(3)
        with col1:
            choix_fps = st.selectbox("Choix du mode", ["1 image/s", "25 images/s", "N images/s"], index=0)
        with col2:
            n_fps = st.number_input("N images/s (si mode N)", min_value=1, max_value=120, value=10, step=1)
        with col3:
            largeur_img = st.number_input("Largeur des images (px)", min_value=160, max_value=3840, value=1280, step=16)

        dossier_images = str(IMAGES_DIR / f"{base_name}_{int(time.time())}")
        if st.button("Extraire les images"):
            fps = 1 if choix_fps == "1 image/s" else (25 if choix_fps == "25 images/s" else int(n_fps))
            ok, log, motif = extraire_images_par_fps(
                ffmpeg=ffmpeg_path,
                chemin_entree=src_path,
                dossier_sortie=dossier_images,
                fps=fps,
                largeur=int(largeur_img),
            )
            afficher_logs_taches(f"Extraction d'images à {fps} i/s", ok, log, dossier_images)
            if ok:
                st.caption(f"Images écrites dans : {dossier_images}")
                st.code(f"Motif des fichiers : {motif}")

    # Extraction par intervalle
    with onglet4:
        st.subheader("Extraction par intervalle")
        col1, col2, col3 = st.columns(3)
        with col1:
            debut = st.text_input("Début (HH:MM:SS ou secondes)", value="00:00:10")
        with col2:
            fin = st.text_input("Fin (HH:MM:SS ou secondes)", value="00:00:40")
        with col3:
            copie_sans_reencodage = st.checkbox("Copie directe (sans ré-encoder)", value=False, help="Utilise -c copy, plus rapide mais calé sur les keyframes.")

        col4, col5 = st.columns(2)
        with col4:
            largeur_intervalle = st.number_input("Largeur si ré-encodage (px)", min_value=320, max_value=3840, value=1280, step=16)
        with col5:
            crf_intervalle = st.slider("CRF intervalle", min_value=18, max_value=35, value=28)

        if st.button("Extraire la séquence"):
            interval_path = str(FICHIERS_DIR / f"{base_name}_intervalle.mp4")
            ok, log = extraire_images_intervalle(
                ffmpeg=ffmpeg_path,
                chemin_entree=src_path,
                chemin_sortie=interval_path,
                debut=debut,
                fin=fin,
                copy=copie_sans_reencodage,
                largeur=int(largeur_intervalle),
                crf=int(crf_intervalle),
            )
            afficher_logs_taches("Extraction par intervalle", ok, log, interval_path)
            if ok and os.path.isfile(interval_path):
                st.video(lire_fichier_binaire(interval_path))
                st.download_button("Télécharger l'extrait", data=lire_fichier_binaire(interval_path), file_name=os.path.basename(interval_path), mime="video/mp4")

    # Recomposition timelapse
    with onglet5:
        st.subheader("Recomposition en timelapse depuis images")
        col1, col2, col3 = st.columns(3)
        with col1:
            fps_tl = st.selectbox("Fréquence timelapse (i/s)", [6, 8, 10, 12, 14], index=0)
        with col2:
            largeur_tl = st.number_input("Largeur timelapse (px)", min_value=320, max_value=3840, value=1280, step=16)
        with col3:
            crf_tl = st.slider("CRF timelapse", min_value=18, max_value=35, value=28)

        st.caption("Choisir un dossier contenant des images numérotées (ex. img_000001.jpg).")
        dossiers_disponibles = sorted([str(p) for p in IMAGES_DIR.glob(f"{base_name}_*") if p.is_dir()])
        dossier_sel = st.selectbox("Dossier d'images détecté", options=["(aucun)"] + dossiers_disponibles, index=0)
        motif_images = st.text_input("Motif d'images (printf-style)", value="img_%06d.jpg")

        if st.button("Composer le timelapse"):
            if dossier_sel == "(aucun)":
                st.error("Aucun dossier d'images sélectionné.")
            else:
                sortie_tl = str(TIMELAPSE_DIR / f"{base_name}_timelapse_{fps_tl}fps.mp4")
                ok, log = composer_timelapse_depuis_images(
                    ffmpeg=ffmpeg_path,
                    dossier_images=dossier_sel,
                    motif=motif_images,
                    fps=int(fps_tl),
                    largeur=int(largeur_tl),
                    crf=int(crf_tl),
                )
                afficher_logs_taches("Composition timelapse", ok, log, sortie_tl)
                if ok and os.path.isfile(sortie_tl):
                    st.video(lire_fichier_binaire(sortie_tl))
                    st.download_button("Télécharger le timelapse", data=lire_fichier_binaire(sortie_tl), file_name=os.path.basename(sortie_tl), mime="video/mp4")

else:
    st.info("Dépose une vidéo pour activer les fonctions d'extraction et d'encodage.")
