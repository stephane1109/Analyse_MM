# pages/2_Timelapse.py
# Timelapse exclusif : on ne produit que le timelapse.

import streamlit as st
from pathlib import Path
from core_media import initialiser_repertoires, zipper
from timelapse import executer_timelapse
import hashlib

st.set_page_config(page_title="Timelapse", layout="wide")
st.title("Timelapse")

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

fps = st.selectbox("FPS timelapse", [4, 6, 8, 10, 12, 14, 16], index=2)
etendue = st.radio("Étendue", ["Toute la vidéo", "Intervalle personnalisé"], index=0, horizontal=True)
use_interval = (etendue == "Intervalle personnalisé")
if use_interval:
    st.info(f"Intervalle personnalisé activé : de {st.session_state['debut_secs']}s à {st.session_state['fin_secs']}s.")
    c1, c2 = st.columns(2)
    st.session_state["debut_secs"] = c1.number_input("Début (s)", min_value=0, value=st.session_state["debut_secs"])
    st.session_state["fin_secs"] = c2.number_input("Fin (s)", min_value=1, value=st.session_state["fin_secs"])

if st.button("Générer le timelapse"):
    if not st.session_state.get("video_base"):
        st.error("Aucune vidéo n’est disponible.")
    else:
        src = st.session_state["video_base"]
        base = st.session_state["base_court"]
        inter = (st.session_state["debut_secs"], st.session_state["fin_secs"]) if use_interval else None
        h = hashlib.sha1((src + str(fps) + str(inter)).encode("utf-8")).hexdigest()[:16]
        out, nb = executer_timelapse(
            src, h, base, fps,
            debut=st.session_state["debut_secs"] if use_interval else None,
            fin=st.session_state["fin_secs"] if use_interval else None
        )
        st.success(f"Timelapse généré ({nb} images).")
        with open(out, "rb") as fh:
            st.download_button("Télécharger le timelapse (.mp4)", data=fh, file_name=Path(out).name, mime="video/mp4")
        zip_path = Path(REP_SORTIE) / f"resultats_{base}_timelapse.zip"
        zipper([Path(out)], zip_path)
        with open(zip_path, "rb") as fhz:
            st.download_button("Télécharger le ZIP", data=fhz, file_name=zip_path.name, mime="application/zip")
