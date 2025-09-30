# pages/4_Analyse_mouvements.py
# Analyse visuelle : magnitude moyenne du flux optique au cours du temps,
# heatmap “thermique”, superposition de vecteurs.

import streamlit as st
from pathlib import Path
import numpy as np
import cv2
from opticalflow import serie_magnitude, lire_frame_a, farneback_pair, heatmap, vectors_overlay

st.set_page_config(page_title="Analyse mouvements", layout="wide")
st.title("Analyse des mouvements (flux optique)")

f = st.file_uploader("Importer une vidéo (.mp4)", type=["mp4"])
stride = st.slider("Pas d’échantillonnage (ms)", 50, 1000, 200, 50)

if f and st.button("Analyser"):
    tmp = Path("/tmp/appdata/tmp") / f.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as g:
        g.write(f.read())
    times, mags = serie_magnitude(str(tmp), stride_ms=stride)
    if not times:
        st.error("Impossible de calculer la série de magnitudes.")
    else:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=mags, mode="lines", name="Magnitude moyenne"))
        fig.update_layout(xaxis_title="Temps (s)", yaxis_title="Magnitude du flux")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        st.subheader("Inspection ponctuelle")
        t0 = st.slider("Temps à inspecter (s)", 0.0, float(times[-1]), float(times[len(times)//2]), 0.1)
        cap = cv2.VideoCapture(str(tmp))
        try:
            f1 = lire_frame_a(cap, max(t0-1.0, 0.0))
            f2 = lire_frame_a(cap, t0)
            g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
            fl = farneback_pair(g1, g2)
            hm = heatmap(fl)
            ov = vectors_overlay(f2, fl)

            c1, c2, c3 = st.columns(3)
            c1.image(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB), caption=f"t={t0:.2f}s")
            c2.image(cv2.cvtColor(hm, cv2.COLOR_BGR2RGB), caption="Heatmap magnitude")
            c3.image(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB), caption="Vecteurs superposés")
        finally:
            cap.release()
