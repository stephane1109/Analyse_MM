# pages/anomalies.py
# Analyse d'anomalies avec timeline consultable et état persistant.
# - Paramètres regroupés dans un formulaire : pas de recalcul tant que tu ne cliques pas "Lancer l'analyse".
# - Résultats conservés en session_state pour éviter toute réinitialisation lors des interactions.
# - Timeline images réelle : scrubber (navigation image par image) + fenêtre temporelle [t0, t1] défilable.
# - Axe Temps (s) cohérent : cadence fixe = frame/fps, frames natives = timestamps réels via ffprobe.
# - Méthodes : LOF / Isolation Forest / Auto-Encodeur, projection 2D Altair, timeline des scores, vignettes, tableau, export.

import math
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from core_media import initialiser_repertoires, info_ffmpeg

# ============================= Utilitaires système =============================

def trouver_ffmpeg() -> Optional[str]:
    """Retourne le chemin de ffmpeg si disponible, sinon None."""
    p, _ = info_ffmpeg()
    return p

def deviner_ffprobe(ffmpeg_path: Optional[str]) -> str:
    """Si on connaît ffmpeg, on déduit ffprobe, sinon 'ffprobe'."""
    if ffmpeg_path and "ffmpeg" in ffmpeg_path:
        return ffmpeg_path.replace("ffmpeg", "ffprobe")
    return "ffprobe"

def executer(cmd: List[str]) -> Tuple[bool, str]:
    """Exécute une commande système et retourne (ok, log)."""
    import subprocess
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

def importer_cv2():
    """Import d'OpenCV (headless recommandé)."""
    try:
        import cv2  # type: ignore
        return cv2, None
    except Exception as e:
        return None, f"OpenCV introuvable : {e}. Ajoute 'opencv-python-headless' dans requirements.txt."

# ============================= Extraction d’images (FFmpeg) =============================

@st.cache_data(show_spinner=False)
def extraire_frames_1080p_cache(ffmpeg: str, video: str, dossier: str, mode_extraction: str, fps_ech: int) -> Tuple[bool, str]:
    """Version mise en cache de l'extraction d'images pour éviter les recalculs intempestifs."""
    return extraire_frames_1080p(ffmpeg, Path(video), Path(dossier), mode_extraction, fps_ech)

def extraire_frames_1080p(
    ffmpeg: str,
    video: Path,
    dossier: Path,
    mode_extraction: str,
    fps_ech: int = 4
) -> Tuple[bool, str]:
    """Extrait des images JPG 1080p. Sortie : frame_%06d.jpg."""
    if dossier.exists():
        try:
            shutil.rmtree(dossier)
        except Exception:
            pass
    dossier.mkdir(parents=True, exist_ok=True)

    motif = str(dossier / "frame_%06d.jpg")
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video)]

    if mode_extraction == "natifs":
        cmd += ["-vf", "scale=1920:-2", "-vsync", "vfr", "-q:v", "2", motif]
    else:
        cmd += ["-vf", f"fps={fps_ech},scale=1920:-2", "-q:v", "2", motif]

    return executer(cmd)

# ============================= Timestamps réels (ffprobe) =============================

@st.cache_data(show_spinner=False)
def lire_timestamps_video_cache(ffprobe: str, video: str) -> Tuple[bool, List[float], str]:
    """Version mise en cache pour lecture des timestamps via ffprobe."""
    return lire_timestamps_video(ffprobe, Path(video))

def lire_timestamps_video(ffprobe: str, video: Path) -> Tuple[bool, List[float], str]:
    """Renvoie (ok, liste de timestamps en secondes, log)."""
    cmd1 = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_frames", "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video)
    ]
    ok1, log1 = executer(cmd1)
    lignes = []
    if ok1 and log1:
        lignes = [x.strip() for x in log1.splitlines() if x.strip()]

    if not lignes:
        cmd2 = [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_frames", "-show_entries", "frame=pkt_pts_time",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video)
        ]
        ok2, log2 = executer(cmd2)
        if ok2 and log2:
            lignes = [x.strip() for x in log2.splitlines() if x.strip()]
        ok_global = ok1 or ok2
        log_global = (log1 + "\n" + log2).strip()
    else:
        ok_global = ok1
        log_global = log1

    temps = []
    for s in lignes:
        try:
            temps.append(float(s))
        except Exception:
            continue
    return ok_global and len(temps) > 0, temps, log_global

# ============================= Chargement images (OpenCV) =============================

def lire_images_cv2(cv2, dossier: Path) -> List[np.ndarray]:
    """Lit toutes les images frame_*.jpg en RGB (uint8)."""
    imgs = []
    for f in sorted(dossier.glob("frame_*.jpg")):
        bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        imgs.append(rgb)
    return imgs

def encadrer_rouge_cv2(cv2, img_rgb: np.ndarray, e: int = 8) -> np.ndarray:
    """Dessine un cadre rouge pour signaler une anomalie."""
    vis = img_rgb.copy()
    h, w = vis.shape[:2]
    cv2.rectangle(vis, (0, 0), (w-1, h-1), (255, 0, 0), thickness=max(1, e))
    return vis

def to_gray_norm(cv2, img_rgb: np.ndarray) -> np.ndarray:
    """Convertit RGB -> Gray float32 [0,1]."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray

def aligner_taille(cv2, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ajuste b à la taille de a si nécessaire (bilinéaire)."""
    if a.shape == b.shape:
        return a, b
    h, w = a.shape[:2]
    b_res = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
    return a, b_res

# ============================= Flux optique Farneback =============================

def calculer_indicateurs_pas(cv2, prev_rgb: np.ndarray, curr_rgb: np.ndarray) -> Optional[Dict[str, float]]:
    """Indicateurs de mouvement pour une paire d'images consécutives."""
    try:
        g0 = to_gray_norm(cv2, prev_rgb)
        g1 = to_gray_norm(cv2, curr_rgb)
        g0, g1 = aligner_taille(cv2, g0, g1)
        flow = cv2.calcOpticalFlowFarneback(
            g0, g1, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag = np.linalg.norm(flow, axis=2).astype(np.float32)
        dx = flow[..., 0].astype(np.float32)
        dy = flow[..., 1].astype(np.float32)
        angle = np.arctan2(dy, dx)
        ux, uy = np.cos(angle), np.sin(angle)
        R_x, R_y = float(np.mean(ux)), float(np.mean(uy))
        R = float(np.sqrt(R_x**2 + R_y**2))
        direction_deg = float(np.degrees(np.arctan2(R_y, R_x)))
        dispersion = float(1.0 - R)
        return {
            "magnitude_moyenne": float(np.mean(mag)),
            "magnitude_ecart_type": float(np.std(mag)),
            "magnitude_p95": float(np.percentile(mag, 95)),
            "energie_mouvement": float(np.sum(mag)),
            "direction_dominante_deg": direction_deg,
            "dispersion_direction": dispersion,
        }
    except Exception:
        return None

# ============================= Prétraitement / features =============================

def construire_X(df: pd.DataFrame, choix_features: str) -> Tuple[np.ndarray, List[str]]:
    """Matrice X selon le choix d'indicateurs."""
    if choix_features == "Magnitude seule":
        cols = ["magnitude_moyenne"]
    elif choix_features == "Magnitude + énergie":
        cols = ["magnitude_moyenne", "energie_mouvement"]
    else:
        cols = ["magnitude_moyenne", "magnitude_ecart_type", "magnitude_p95",
                "energie_mouvement", "direction_dominante_deg", "dispersion_direction"]
    X = df[cols].to_numpy(dtype=np.float64)
    return X, cols

# ============================= Détecteurs d'anomalies =============================

def anomalies_lof(X: np.ndarray, contamination: float, n_neighbors: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Local Outlier Factor : scores élevés = plus anormal."""
    from sklearn.neighbors import LocalOutlierFactor
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=False)
    y = lof.fit_predict(X)  # -1 outlier
    scores = -lof.negative_outlier_factor_
    return scores, y

def anomalies_isoforest(X: np.ndarray, contamination: float, n_estimators: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Isolation Forest : scores élevés = plus anormal."""
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42, n_jobs=-1)
    y = iso.fit_predict(X)  # -1 outlier
    scores = -iso.decision_function(X)
    return scores, y

def anomalies_autoencodeur(X: np.ndarray, contamination: float, hidden: int = 8, max_iter: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """Auto-encodeur léger (MLPRegressor) : erreur de reconstruction comme score."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    ae = MLPRegressor(hidden_layer_sizes=(hidden,), activation="relu", solver="adam", max_iter=max_iter, random_state=42)
    ae.fit(Xs, Xs)
    X_pred = ae.predict(Xs)
    err = np.mean((X_pred - Xs) ** 2, axis=1)
    seuil = float(np.quantile(err, 1.0 - contamination))
    y = np.where(err >= seuil, -1, 1)
    return err, y

# ============================= Projection 2D (PCA / t-SNE) =============================

def projeter_2d(X: np.ndarray, methode: str) -> np.ndarray:
    """Retourne une projection 2D (PCA rapide, t-SNE plus lent)."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if methode == "t-SNE (lent)":
        from sklearn.manifold import TSNE
        emb = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=30, random_state=42).fit_transform(Xs)
    else:
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2, random_state=42).fit_transform(Xs)
    return emb

# ============================= App Streamlit =============================

BASE_DIR, REP_SORTIE, REP_TMP = initialiser_repertoires()
st.set_page_config(page_title="Anomalies + Timeline consultable", layout="wide")
st.title("Anomalies + Timeline images consultable")
st.markdown("www.codeandcortex.fr")

ffmpeg_path = trouver_ffmpeg()
if not ffmpeg_path:
    st.error("FFmpeg introuvable. Binaire attendu sous /usr/bin/ffmpeg ou similaire.")
    st.stop()
ffprobe_path = deviner_ffprobe(ffmpeg_path)

cv2, cv_err = importer_cv2()
if cv2 is None:
    st.error(cv_err)
    st.stop()

# État global des résultats
st.session_state.setdefault("anom", None)

# ----------------------------- Formulaire paramètres -----------------------------
with st.form("params"):
    st.subheader("Paramètres d’analyse")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        src = st.radio("Source", ["Importer un MP4", "Vidéo préparée"], index=0, horizontal=True)
    with c2:
        mode_ext = st.radio("Extraction d’images", ["Frames natives", "Cadence fixe"], index=0)
    with c3:
        fps = st.number_input("Cadence fixe (i/s)", min_value=1, max_value=60, value=4, step=1)
    with c4:
        pas = st.number_input("Pas d’analyse (1 = chaque image)", min_value=1, max_value=200, value=1, step=1)

    c5, c6, c7 = st.columns(3)
    with c5:
        choix_feat = st.selectbox("Indicateurs", ["Magnitude seule", "Magnitude + énergie", "Tous les indicateurs"])
    with c6:
        methode = st.selectbox("Méthode anomalies", ["Local Outlier Factor", "Isolation Forest", "Auto-Encodeur"])
    with c7:
        projection = st.selectbox("Projection 2D", ["PCA (rapide)", "t-SNE (lent)"], index=0)

    c8, c9, c10 = st.columns(3)
    with c8:
        contamination = st.slider("Contamination attendue", 0.01, 0.4, 0.1, 0.01)
    if methode == "Local Outlier Factor":
        with c9:
            n_neighbors = st.number_input("n_neighbors (LOF)", min_value=5, max_value=100, value=20, step=1)
    elif methode == "Isolation Forest":
        with c9:
            n_estimators = st.number_input("n_estimators (ISO)", min_value=50, max_value=1000, value=200, step=50)
    else:
        with c9:
            hidden = st.number_input("Taille couche cachée (Auto-Enc.)", min_value=2, max_value=128, value=8, step=1)

    # Choix de la source
    video_path: Optional[Path] = None
    if src == "Importer un MP4":
        up = st.file_uploader("Importer une vidéo (.mp4)", type=["mp4"], key="upload_anom")
        if up is not None:
            tmp = REP_TMP / f"anom_{up.name}"
            with open(tmp, "wb") as g:
                g.write(up.read())
            video_path = tmp
            st.success(f"Fichier chargé : {tmp.name}")
    else:
        if st.session_state.get("video_base"):
            p = Path(st.session_state["video_base"])
            if p.exists():
                video_path = p
                st.info(f"Vidéo préparée utilisée : {p.name}")
            else:
                st.warning("La vidéo préparée est introuvable.")
        else:
            st.warning("Aucune vidéo préparée en session.")

    lancer = st.form_submit_button("Lancer l’analyse", type="primary")

# ----------------------------- Exécution unique de l’analyse -----------------------------
if lancer:
    if video_path is None:
        st.error("Aucune source vidéo.")
        st.stop()

    frames_dir = (BASE_DIR / "frames_anom" / video_path.stem).resolve()
    mode = "natifs" if mode_ext == "Frames natives" else "fixe"

    ok, log = extraire_frames_1080p_cache(ffmpeg_path, str(video_path), str(frames_dir), mode, int(fps))
    if not ok:
        st.error("Échec extraction d’images avec FFmpeg.")
        st.code(log or "(journal vide)", language="bash")
        st.stop()

    imgs = lire_images_cv2(cv2, frames_dir)
    n = len(imgs)
    if n < 3:
        st.error("Trop peu d’images pour analyser.")
        st.stop()

    # Timestamps par frame
    if mode == "natifs":
        ok_ts, ts_all, _ = lire_timestamps_video_cache(ffprobe_path, str(video_path))
        if ok_ts and len(ts_all) > 0:
            m = min(len(ts_all), n)
            temps_par_frame = np.array(ts_all[:m], dtype=float)
            if m < n:
                dernier = temps_par_frame[-1] if m > 0 else 0.0
                extra = np.linspace(dernier, dernier + (n - m) * 1.0 / max(1, fps), num=(n - m), endpoint=False)
                temps_par_frame = np.concatenate([temps_par_frame, extra])
        else:
            temps_par_frame = np.arange(n, dtype=float)  # fallback
    else:
        temps_par_frame = np.arange(n, dtype=float) / float(max(1, fps))

    # Indices d’analyse
    indices = list(range(0, n, int(pas)))
    if len(indices) < 2:
        indices = list(range(0, n, 1))

    # Calcul indicateurs par pas
    lignes: List[Dict[str, float]] = []
    for k in range(1, len(indices)):
        i0 = indices[k-1]
        i1 = indices[k]
        met = calculer_indicateurs_pas(cv2, imgs[i0], imgs[i1])
        if met is None:
            continue
        d = {"etape": k, "frame_prev": i0, "frame_curr": i1, **met, "t": float(temps_par_frame[i1])}
        lignes.append(d)
    if not lignes:
        st.error("Aucune paire exploitable pour calculer les indicateurs.")
        st.stop()

    df = pd.DataFrame(lignes)
    X, cols = construire_X(df, choix_feat)

    # Détection d’anomalies
    if methode == "Local Outlier Factor":
        scores, ypred = anomalies_lof(X, contamination=float(contamination), n_neighbors=int(n_neighbors))
    elif methode == "Isolation Forest":
        scores, ypred = anomalies_isoforest(X, contamination=float(contamination), n_estimators=int(n_estimators))
    else:
        scores, ypred = anomalies_autoencodeur(X, contamination=float(contamination), hidden=int(hidden))

    df["score_anomalie"] = scores
    df["anomalie"] = (ypred == -1)
    df["etat"] = np.where(df["anomalie"], "Anomalie", "Normal")

    # Projection 2D mise en cache de session (selon X et projection)
    try:
        emb = projeter_2d(X, methode=projection)
        df["x"], df["y"] = emb[:, 0], emb[:, 1]
    except Exception:
        df["x"], df["y"] = np.nan, np.nan

    # État persistant des résultats
    st.session_state["anom"] = {
        "video_path": str(video_path),
        "frames_dir": str(frames_dir),
        "mode": mode,
        "fps": int(fps),
        "imgs": imgs,                     # numpy arrays en mémoire
        "times": temps_par_frame,         # temps par frame
        "indices": indices,               # mapping étapes -> frames
        "df": df,                         # résultats par pas
        "cols": cols,                     # features utilisées
    }

# ----------------------------- Affichage résultats persistants -----------------------------
res = st.session_state.get("anom")
if not res:
    st.info("Configure les paramètres puis clique « Lancer l’analyse » pour afficher la timeline.")
    st.stop()

imgs = res["imgs"]
temps_par_frame = res["times"]
indices = res["indices"]
df = res["df"]
fps = res["fps"]
mode = res["mode"]
n = len(imgs)

st.subheader("Résumé")
nb_ano = int(df["anomalie"].sum())
st.write(f"Images extraites : {n} • Paires analysées : {len(df)} • Anomalies détectées : {nb_ano}")

# -------- Projection 2D Altair --------
st.subheader("Projection 2D (Altair)")
try:
    hover2d = alt.selection_point(fields=["etape"], on="mouseover", nearest=True, empty=False)
    base2d = alt.Chart(df).mark_point(filled=True).encode(
        x=alt.X("x:Q", title="Composante 1"),
        y=alt.Y("y:Q", title="Composante 2"),
        color=alt.Color("etat:N",
                        scale=alt.Scale(domain=["Normal", "Anomalie"], range=["#377eb8", "#e41a1c"]),
                        title="État"),
        size=alt.Size("etat:N", scale=alt.Scale(domain=["Normal", "Anomalie"], range=[30, 70]), legend=None),
        tooltip=[alt.Tooltip("etape:Q", title="Étape"),
                 alt.Tooltip("frame_curr:Q", title="Frame"),
                 alt.Tooltip("t:Q", title="Temps (s)", format=".2f"),
                 alt.Tooltip("score_anomalie:Q", title="Score", format=".3f")]
    ).add_params(hover2d).properties(width=700, height=480)
    st.altair_chart((base2d + base2d.transform_filter(hover2d).mark_point(stroke="black", strokeWidth=1.5)).interactive(),
                    use_container_width=True)
except Exception as e:
    st.warning(f"Projection indisponible : {e}")

# -------- Timeline des scores (Altair) --------
st.subheader("Timeline des scores")
axe_timeline = st.radio("Axe", ["Temps (s)", "Frame", "Étape"], index=0, horizontal=True, key="axe_timeline_scores")

if axe_timeline == "Temps (s)":
    x_field = alt.X("t:Q", title="Temps (s)")
elif axe_timeline == "Frame":
    x_field = alt.X("frame_curr:Q", title="Frame")
else:
    x_field = alt.X("etape:Q", title="Étape")

base_line = alt.Chart(df).mark_line().encode(
    x=x_field,
    y=alt.Y("score_anomalie:Q", title="Score d’anomalie"),
    tooltip=[alt.Tooltip("etape:Q", title="Étape"),
             alt.Tooltip("frame_curr:Q", title="Frame"),
             alt.Tooltip("t:Q", title="Temps (s)", format=".2f"),
             alt.Tooltip("score_anomalie:Q", title="Score", format=".3f")]
).properties(width=900, height=320)

points_normaux = alt.Chart(df[df["anomalie"] == False]).mark_point(filled=True).encode(
    x=x_field, y=alt.Y("score_anomalie:Q"),
    color=alt.value("#377eb8"), size=alt.value(30)
)
points_anom = alt.Chart(df[df["anomalie"] == True]).mark_point(filled=True).encode(
    x=x_field, y=alt.Y("score_anomalie:Q"),
    color=alt.value("#e41a1c"), size=alt.value(70)
)
st.altair_chart((base_line + points_normaux + points_anom).interactive(), use_container_width=True)

# -------- Timeline IMAGES consultable --------
st.subheader("Timeline images consultable")
t_min = float(np.min(temps_par_frame)) if n > 0 else 0.0
t_max = float(np.max(temps_par_frame)) if n > 0 else 0.0

# État persistant de navigation
st.session_state.setdefault("scrub_t", t_min)
st.session_state.setdefault("win_t0", t_min)
st.session_state.setdefault("win_t1", min(t_min + max(5.0, (t_max - t_min) * 0.05), t_max))

cA, cB = st.columns([1, 1])
with cA:
    # Scrubber mono-valeur : navigation image par image
    scrub = st.slider(
        "Scrubber (temps, navigation image par image)",
        min_value=t_min, max_value=t_max,
        value=float(st.session_state["scrub_t"]),
        step=max(0.001, (t_max - t_min) / max(1000, n))
    )
with cB:
    # Fenêtre temporelle défilable
    t0, t1 = st.slider(
        "Fenêtre temporelle [t0, t1] pour le ruban d’images",
        min_value=t_min, max_value=t_max,
        value=(float(st.session_state["win_t0"]), float(st.session_state["win_t1"])),
        step=max(0.01, (t_max - t_min) / max(1000, n))
    )

st.session_state["scrub_t"] = float(scrub)
st.session_state["win_t0"] = float(min(t0, t1))
st.session_state["win_t1"] = float(max(t0, t1))

# Trouve la frame la plus proche du scrubber
idx_scrub = int(np.argmin(np.abs(temps_par_frame - st.session_state["scrub_t"])))
frame_scrub = np.clip(idx_scrub, 0, n - 1)

# Affichage de l'image courante (scrubber)
st.markdown("Aperçu au temps sélectionné")
is_anom = bool(df.loc[df["frame_curr"] == frame_scrub, "anomalie"].any()) if "frame_curr" in df else False
img_scrub = encadrer_rouge_cv2(cv2, imgs[frame_scrub], e=8) if is_anom else imgs[frame_scrub]
cap_scrub = f"t={temps_par_frame[frame_scrub]:.2f}s • frame {frame_scrub}"
if "etape" in df.columns:
    et_scr = df.loc[df["frame_curr"] == frame_scrub, "etape"]
    if len(et_scr) > 0:
        cap_scrub += f" • étape {int(et_scr.iloc[0])}"
if "score_anomalie" in df.columns:
    sc_scr = df.loc[df["frame_curr"] == frame_scrub, "score_anomalie"]
    if len(sc_scr) > 0:
        cap_scrub += f" • score={float(sc_scr.iloc[0]):.3f}"
st.image(img_scrub, caption=cap_scrub, use_container_width=True)

# Ruban d’images sur la fenêtre temporelle
st.markdown("Ruban d’images sur la fenêtre temporelle")
mask_win = (temps_par_frame >= st.session_state["win_t0"]) & (temps_par_frame <= st.session_state["win_t1"])
idxs = np.nonzero(mask_win)[0].tolist()

if len(idxs) == 0:
    st.info("Aucune image dans cette fenêtre. Ajuste [t0, t1].")
else:
    frames_anormales = set(df[df["anomalie"]]["frame_curr"].astype(int).tolist())
    score_map = {int(r["frame_curr"]): float(r["score_anomalie"]) for _, r in df.iterrows()}
    etape_map = {int(r["frame_curr"]): int(r["etape"]) for _, r in df.iterrows()}

    cols_par_ligne = 10
    k = 0
    for _ in range(math.ceil(len(idxs) / cols_par_ligne)):
        cols = st.columns(cols_par_ligne)
        for c in cols:
            if k >= len(idxs):
                break
            fr = int(idxs[k])
            vis = encadrer_rouge_cv2(cv2, imgs[fr], e=6) if fr in frames_anormales else imgs[fr]
            cap = f"t={temps_par_frame[fr]:.2f}s • frame {fr}"
            if fr in etape_map:
                cap += f" • étape {etape_map[fr]}"
            if fr in score_map:
                cap += f" • score={score_map[fr]:.3f}"
            c.image(vis, caption=cap, use_container_width=False)
            k += 1

# -------- Vignettes d’anomalies --------
st.subheader("Vignettes des anomalies")
if nb_ano == 0:
    st.info("Aucune anomalie détectée.")
else:
    tri = st.selectbox("Tri", ["Score décroissant", "Temps croissant", "Frame croissant"], index=0, key="tri_anom")
    dfa = df[df["anomalie"]].copy()
    if tri == "Temps croissant":
        dfa = dfa.sort_values("t", ascending=True)
    elif tri == "Frame croissant":
        dfa = dfa.sort_values("frame_curr", ascending=True)
    else:
        dfa = dfa.sort_values("score_anomalie", ascending=False)

    cols_par_ligne = 8
    k = 0
    max_show = 24
    for _ in range(math.ceil(min(len(dfa), max_show) / cols_par_ligne)):
        cols = st.columns(cols_par_ligne)
        for c in cols:
            if k >= min(len(dfa), max_show):
                break
            row = dfa.iloc[k]
            fr = int(row["frame_curr"])
            if 0 <= fr < n:
                vis = encadrer_rouge_cv2(cv2, imgs[fr], e=8)
                cap = f"t={float(row['t']):.2f}s • frame {fr} • score={float(row['score_anomalie']):.3f}"
                c.image(vis, caption=cap, use_container_width=False)
            k += 1

# -------- Tableau et export --------
st.subheader("Tableau et export")
colonnes_aff = ["t", "etape", "frame_prev", "frame_curr", *res["cols"], "score_anomalie", "anomalie"]
if "x" in df and "y" in df:
    colonnes_aff += ["x", "y"]
st.dataframe(df[colonnes_aff])
st.download_button(
    "Télécharger les scores (CSV)",
    data=df[colonnes_aff].to_csv(index=False).encode("utf-8"),
    file_name="scores_anomalies.csv",
    mime="text/csv"
)
