# pages/1_Extraction.py
# Extraction classique : MP4, MP3, WAV, images 1/25 fps, zip des résultats.

import streamlit as st
from pathlib import Path
from core_media import initialiser_repertoires, extraire_ressources, zipper, SEUIL_APERCU_OCTETS

st.set_page_config(page_title="Extraction", layout="wide")
st.title("Extraction classique")
st.markdown("MP4, MP3, WAV, images 1 fps / 25 fps")

_, REP_SORTIE, _ = initialiser_repertoires()

st.session_state.setdefault("video_base", None)
st.session_state.setdefault("base_court", None)
st.session_state.setdefault("debut_secs", 0)
st.session_state.setdefault("fin_secs", 10)

if not st.session_state.get("video_base"):
    st.warning("Aucune vidéo préparée depuis la page d’accueil. Vous pouvez tout de même fournir un fichier ci-dessous.")
    f = st.file_uploader("Importer une vidéo (.mp4)", type=["mp4"])
    if f:
        p = REP_SORTIE / f"direct_{f.name}"
        with open(p, "wb") as g:
            g.write(f.read())
        st.session_state["video_base"] = str(p)
        st.session_state["base_court"] = "direct_import"

st.write(f"Vidéo courante : {Path(st.session_state['video_base']).name if st.session_state.get('video_base') else 'Aucune'}")

etendue = st.radio("Étendue", ["Toute la vidéo", "Intervalle personnalisé"], index=0, horizontal=True)
use_interval = (etendue == "Intervalle personnalisé")
if use_interval:
    st.info(f"Intervalle personnalisé activé : de {st.session_state['debut_secs']}s à {st.session_state['fin_secs']}s. L’extraction portera sur cet intervalle.")
    c1, c2 = st.columns(2)
    st.session_state["debut_secs"] = c1.number_input("Début (s)", min_value=0, value=st.session_state["debut_secs"])
    st.session_state["fin_secs"] = c2.number_input("Fin (s)", min_value=1, value=st.session_state["fin_secs"])

st.subheader("Ressources à produire")
c1, c2, c3, c4, c5 = st.columns(5)
with c1: opt_mp4 = st.checkbox("MP4", value=True)
with c2: opt_mp3 = st.checkbox("MP3", value=False)
with c3: opt_wav = st.checkbox("WAV", value=False)
with c4: opt_img1 = st.checkbox("Images 1 fps", value=False)
with c5: opt_img25 = st.checkbox("Images 25 fps", value=False)

if st.button("Extraire"):
    if not st.session_state.get("video_base"):
        st.error("Aucune vidéo n’est disponible. Allez sur la page d’accueil pour préparer la vidéo, ou importez-en une ci-dessus.")
    else:
        opts = {"mp4": opt_mp4, "mp3": opt_mp3, "wav": opt_wav, "img1": opt_img1, "img25": opt_img25}
        debut, fin = (st.session_state["debut_secs"], st.session_state["fin_secs"]) if use_interval else (0, 0)
        ok, res = extraire_ressources(st.session_state["video_base"], debut, fin, st.session_state["base_court"], opts, use_interval)
        if ok:
            # Regrouper en zip
            fichiers = []
            for r in res:
                fichiers.append(Path(r))
            zip_path = Path(REP_SORTIE) / f"resultats_{st.session_state['base_court']}.zip"
            zipper(fichiers, zip_path)
            with open(zip_path, "rb") as fh:
                st.download_button("Télécharger les résultats (.zip)", data=fh, file_name=zip_path.name, mime="application/zip")
            st.success("Extraction terminée.")
        else:
            st.error(res)
