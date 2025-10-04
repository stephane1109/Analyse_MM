# pages/anomalies.py
# Détection d'anomalies sur indicateurs de mouvement avec visualisation 2D type "clustering".
# Méthodes : Local Outlier Factor, Isolation Forest, Auto-Encodeur léger (MLPRegressor).
# Visualisation : projection 2D (PCA rapide ou t-SNE) avec points = pas (frame d'arrivée), anomalies en rouge.

import math
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st

from core_media import initialiser_repertoires, info_ffmpeg

# =============================
# Utilitaires système
# =============================

def trouver_ffmpeg() -> Optional[str]:
    """Retourne le chemin de ffmpeg si disponible, sinon None."""
    p, _ = info_ffmpeg()
    return p

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
    """Import d'OpenCV (opencv-python-headless recommandé)."""
    try:
        import cv2  # type: ignore
        return cv2, None
    except Exception as e:
        return None, f"OpenCV introuvable : {e}. Ajoute 'opencv-python-headless' dans requirements.txt."

# =============================
# Extraction d’images (FFmpeg)
# =============================

def extraire_frames_1080p(
    ffmpeg: str,
    video: Path,
    dossier: Path,
    mode_extraction: str,
    fps_ech: int = 4
) -> Tuple[bool, str]:
    """
    Extrait des images JPG en 1080p (largeur 1920).
    mode_extraction = "natifs" -> toutes les frames sources (timelapse, VFR), -vsync vfr.
    mode_extraction = "fixe"   -> fps_ech images/seconde (uniforme).
    Sortie : frame_%06d.jpg
    """
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

# =============================
# Chargement images et utilitaires image (OpenCV)
# =============================

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
    """Convertit RGB -> Gray float32 [0,1] pour stabilité numérique."""
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

# =============================
# Indicateurs par pas (flux optique Farneback)
# =============================

def calculer_indicateurs_pas(cv2, prev_rgb: np.ndarray, curr_rgb: np.ndarray) -> Optional[Dict[str, float]]:
    """Calcule les indicateurs de mouvement pour une paire d'images."""
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

# =============================
# Prétraitement, z-score et choix des features
# =============================

def zscore(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Standardise x → z = (x - mu) / sigma. Retourne (z, mu, sigma)."""
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma < 1e-12:
        sigma = 1.0
    return (x - mu) / sigma, mu, sigma

def construire_X(df: pd.DataFrame, choix_features: str) -> Tuple[np.ndarray, List[str]]:
    """Construit la matrice X à partir du DataFrame des indicateurs."""
    if choix_features == "Magnitude seule":
        cols = ["magnitude_moyenne"]
    elif choix_features == "Magnitude + énergie":
        cols = ["magnitude_moyenne", "energie_mouvement"]
    else:
        cols = ["magnitude_moyenne", "magnitude_ecart_type", "magnitude_p95",
                "energie_mouvement", "direction_dominante_deg", "dispersion_direction"]
    X = df[cols].to_numpy(dtype=np.float64)
    return X, cols

# =============================
# Détecteurs d'anomalies
# =============================

def anomalies_lof(X: np.ndarray, contamination: float, n_neighbors: int = 20, metric: str = "euclidean") -> Tuple[np.ndarray, np.ndarray]:
    """Local Outlier Factor : retourne (scores, y_pred) où scores élevés = plus anormal."""
    from sklearn.neighbors import LocalOutlierFactor
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=False, metric=metric)
    y = lof.fit_predict(X)  # -1 = outlier, 1 = inlier
    scores = -lof.negative_outlier_factor_  # renversé : + grand = plus anormal
    return scores, y

def anomalies_isoforest(X: np.ndarray, contamination: float, n_estimators: int = 200, max_samples: str | int = "auto") -> Tuple[np.ndarray, np.ndarray]:
    """Isolation Forest : retourne (scores, y_pred) où scores élevés = plus anormal."""
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    y = iso.fit_predict(X)  # -1 = outlier
    scores = -iso.decision_function(X)  # renversé : + grand = plus anormal
    return scores, y

def anomalies_autoencodeur(X: np.ndarray, contamination: float, hidden: int = 8, max_iter: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """
    Auto-encodeur léger via MLPRegressor (scikit-learn).
    On apprend X -> X, et on prend l'erreur de reconstruction comme score (grand = plus anormal).
    Seuillage par quantile = contamination.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    ae = MLPRegressor(
        hidden_layer_sizes=(hidden,),
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=42
    )
    ae.fit(Xs, Xs)
    X_pred = ae.predict(Xs)
    err = np.mean((X_pred - Xs) ** 2, axis=1)  # MSE par échantillon

    seuil = float(np.quantile(err, 1.0 - contamination))
    y = np.where(err >= seuil, -1, 1)
    scores = err.copy()
    return scores, y

# =============================
# Projection 2D (PCA / t-SNE) pour visualisation
# =============================

def projeter_2d(X: np.ndarray, methode: str) -> np.ndarray:
    """Projette X en 2D avec PCA (rapide) ou t-SNE (plus lent). Retourne un tableau (n,2)."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if methode == "t-SNE (lent)":
        from sklearn.manifold import TSNE
        emb = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=30, random_state=42).fit_transform(Xs)
    else:
        # PCA rapide et stable
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2, random_state=42).fit_transform(Xs)
    return emb

# =============================
# Page Streamlit
# =============================

BASE_DIR, REP_SORTIE, REP_TMP = initialiser_repertoires()

st.set_page_config(page_title="Tests anomalies + projection 2D", layout="wide")
st.title("Tests d’anomalies sur indicateurs + projection 2D")
st.markdown("www.codeandcortex.fr")

st.markdown(
    "Cette page calcule des indicateurs de mouvement par flux optique entre images successives, "
    "applique une méthode d’anomalies, puis projette les points (pas) en 2D pour une visualisation de type clustering. "
    "Les **anomalies sont affichées en rouge**, les autres en bleu."
)

# Vérifications préalables
ff = trouver_ffmpeg()
if not ff:
    st.error("FFmpeg introuvable. Binaire attendu sous /usr/bin/ffmpeg ou similaire.")
    st.stop()

cv2, cv_err = importer_cv2()
if cv2 is None:
    st.error(cv_err)
    st.stop()

# Source vidéo
st.subheader("Source vidéo")
src = st.radio("Choix de la source", ["Importer un MP4", "Utiliser la vidéo préparée"], index=0, horizontal=True)

video_path: Optional[Path] = None
if src == "Importer un MP4":
    up = st.file_uploader("Importer une vidéo (.mp4)", type=["mp4"], key="upload_anom")
    if up is not None:
        video_path = REP_TMP / f"anom_{up.name}"
        with open(video_path, "wb") as g:
            g.write(up.read())
        st.success(f"Fichier chargé : {video_path.name}")
else:
    if st.session_state.get("video_base"):
        p = Path(st.session_state["video_base"])
        if p.exists():
            video_path = p
            st.info(f"Vidéo préparée utilisée : {p.name}")
        else:
            st.warning("La vidéo préparée est introuvable. Importer un MP4.")
    else:
        st.warning("Aucune vidéo préparée disponible. Importer un MP4.")

# Paramètres d’extraction et d’analyse
st.subheader("Paramètres d’extraction et d’analyse")
c1, c2, c3, c4 = st.columns(4)
with c1:
    mode_ext = st.radio("Extraction d’images", ["Frames natives", "Cadence fixe"], index=0)
with c2:
    fps = st.number_input("Cadence fixe (i/s)", min_value=1, max_value=60, value=4, step=1)
with c3:
    pas = st.number_input("Pas d’analyse (1 = chaque image)", min_value=1, max_value=200, value=1, step=1)
with c4:
    choix_feat = st.selectbox("Indicateurs utilisés", ["Magnitude seule", "Magnitude + énergie", "Tous les indicateurs"])

# Choix méthode anomalies + projection 2D
st.subheader("Méthode d’anomalies et projection")
methode = st.selectbox("Méthode", ["Local Outlier Factor", "Isolation Forest", "Auto-Encodeur"])
cL, cR = st.columns(2)
with cL:
    projection = st.selectbox("Projection 2D", ["PCA (rapide)", "t-SNE (lent)"], index=0)
with cR:
    contamination = st.slider("Contamination (proportion attendue d’anomalies)", 0.01, 0.4, 0.1, 0.01)

# Hyperparamètres spécifiques
if methode == "Local Outlier Factor":
    n_neighbors = st.number_input("n_neighbors (LOF)", min_value=5, max_value=100, value=20, step=1)
elif methode == "Isolation Forest":
    n_estimators = st.number_input("n_estimators (ISO)", min_value=50, max_value=1000, value=200, step=50)
else:  # Auto-encodeur
    hidden = st.number_input("Taille couche cachée (Auto-Enc.)", min_value=2, max_value=128, value=8, step=1)

# Lancement
if st.button("Lancer les tests", type="primary"):
    if video_path is None:
        st.error("Aucune source vidéo.")
        st.stop()

    # Extraction d'images
    frames_dir = (BASE_DIR / "frames_anom" / video_path.stem).resolve()
    mode = "natifs" if mode_ext == "Frames natives" else "fixe"
    ok, log = extraire_frames_1080p(trouver_ffmpeg(), video_path, frames_dir, mode, int(fps))
    if not ok:
        st.error("Échec extraction d’images avec FFmpeg.")
        st.code(log or "(journal vide)", language="bash")
        st.stop()

    # Chargement images
    imgs = lire_images_cv2(cv2, frames_dir)
    n = len(imgs)
    st.info(f"Images extraites : {n} ({'frames natives' if mode=='natifs' else f'{int(fps)} i/s'})")
    if n < 3:
        st.error("Trop peu d’images pour analyser.")
        st.stop()

    # Construction des indices par pas avec "pas" entre frames
    indices = list(range(0, n, int(pas)))
    if len(indices) < 2:
        st.warning("Pas d’analyse trop grand pour la séquence. Utilisation automatique de pas=1.")
        indices = list(range(0, n, 1))

    lignes: List[Dict[str, float]] = []
    echecs = 0
    for k in range(1, len(indices)):
        i0 = indices[k-1]
        i1 = indices[k]
        met = calculer_indicateurs_pas(cv2, imgs[i0], imgs[i1])
        if met is None:
            echecs += 1
            continue
        lignes.append({"etape": k, "frame_prev": i0, "frame_curr": i1, **met})

    if not lignes:
        st.error("Aucune paire exploitable pour calculer les indicateurs. Essaie pas=1.")
        st.stop()

    df = pd.DataFrame(lignes)
    if echecs > 0:
        st.warning(f"{echecs} paire(s) ont été ignorées suite à un échec du calcul du flux optique.")

    # Choix des features et construction X
    X, cols = construire_X(df, choix_feat)

    # Méthode d’anomalies
    try:
        if methode == "Local Outlier Factor":
            scores, ypred = anomalies_lof(X, contamination=float(contamination), n_neighbors=int(n_neighbors))
        elif methode == "Isolation Forest":
            scores, ypred = anomalies_isoforest(X, contamination=float(contamination), n_estimators=int(n_estimators))
        else:
            scores, ypred = anomalies_autoencodeur(X, contamination=float(contamination), hidden=int(hidden))
    except Exception as e:
        st.error(f"Échec de la méthode '{methode}' : {e}. Vérifie scikit-learn dans requirements.txt.")
        st.stop()

    df["score_anomalie"] = scores
    df["anomalie"] = (ypred == -1)

    # Résumés
    st.subheader("Résumé et indicateurs utilisés")
    st.write(f"Indicateurs : {', '.join(cols)}")
    st.write(f"Méthode : {methode}  |  Contamination : {float(contamination):.2f}")
    nb_ano = int(df["anomalie"].sum())
    st.write(f"Anomalies détectées : {nb_ano} / {len(df)} pas")

    # Courbe des scores
    st.subheader("Courbe du score d’anomalie")
    st.line_chart(df.set_index("etape")[["score_anomalie"]])

    # ========= NOUVEAU : Projection 2D et nuage de points "type clustering" =========
    st.subheader("Projection 2D des pas (clustering visuel)")
    st.caption(
        "La projection 2D (PCA ou t-SNE) place chaque pas comme un point. "
        "Les anomalies (rouge) se regroupent souvent en zones périphériques ou isolées."
    )
    try:
        emb = projeter_2d(X, methode=projection)
        df_proj = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "anomalie": df["anomalie"], "etape": df["etape"], "frame_curr": df["frame_curr"]})
        # Matplotlib simple, anomalies en rouge, normales en bleu
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        normaux = df_proj[~df_proj["anomalie"]]
        anormaux = df_proj[df_proj["anomalie"]]
        ax.scatter(normaux["x"], normaux["y"], s=18, label="Normal", alpha=0.7)
        ax.scatter(anormaux["x"], anormaux["y"], s=30, label="Anomalie", alpha=0.9, c="red")
        ax.set_xlabel("Composante 1")
        ax.set_ylabel("Composante 2")
        ax.set_title(f"Projection 2D ({projection}) - {methode}")
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Projection 2D indisponible : {e}")

    # Vignettes anomalies (encadrées en rouge)
    st.subheader("Vignettes des anomalies détectées")
    if nb_ano == 0:
        st.info("Aucune anomalie détectée au seuil demandé.")
    else:
        # On propose deux tris : par score, ou par position dans le plan
        tri = st.selectbox("Trier les vignettes d’anomalies", ["Score décroissant", "x croissant (projection)", "y croissant (projection)"], index=0)
        dfa = df.copy()
        dfa = dfa[dfa["anomalie"]]
        if "emb" in locals():
            dfa = dfa.join(pd.DataFrame(emb, columns=["x", "y"]), how="left")
        if tri == "Score décroissant":
            dfa = dfa.sort_values("score_anomalie", ascending=False)
        elif tri == "x croissant (projection)" and "x" in dfa:
            dfa = dfa.sort_values("x", ascending=True)
        elif tri == "y croissant (projection)" and "y" in dfa:
            dfa = dfa.sort_values("y", ascending=True)

        # Affiche jusqu’à 24 vignettes
        # Il nous faut les images pour les frames d'arrivée
        cols_par_ligne = 8
        k = 0
        max_show = 24
        for _ in range(math.ceil(min(len(dfa), max_show) / cols_par_ligne)):
            cols_st = st.columns(cols_par_ligne)
            for c in cols_st:
                if k >= min(len(dfa), max_show):
                    break
                row = dfa.iloc[k]
                dst = int(row["frame_curr"])
                # Relecture rapide de l'image depuis le cache en mémoire (imgs)
                if 0 <= dst < len(imgs):
                    vis = encadrer_rouge_cv2(cv2, imgs[dst], e=8)
                    cap = f"frame #{dst} • score={row['score_anomalie']:.3f}"
                    if "x" in row and "y" in row and not (pd.isna(row["x"]) or pd.isna(row["y"])):
                        cap += f" • (x={row['x']:.2f}, y={row['y']:.2f})"
                    c.image(vis, caption=cap, use_container_width=False)
                k += 1

    # Tableau et export
    st.subheader("Tableau des scores et décisions")
    colonnes_aff = ["etape", "frame_prev", "frame_curr", *cols, "score_anomalie", "anomalie"]
    if "emb" in locals():
        df["x"], df["y"] = emb[:, 0], emb[:, 1]
        colonnes_aff += ["x", "y"]
    st.dataframe(df[colonnes_aff])

    st.subheader("Exporter les résultats")
    st.download_button(
        "Télécharger les scores (CSV)",
        data=df[colonnes_aff].to_csv(index=False).encode("utf-8"),
        file_name="scores_anomalies.csv",
        mime="text/csv"
    )

# =============================
# Explications méthodologiques
# =============================

st.subheader("Explications et recommandations")
st.markdown(
    "Indicateurs. La **magnitude moyenne** du flux optique résume l’intensité du mouvement par pas. "
    "L’**énergie de mouvement** (somme des magnitudes) renforce les événements étendus. "
    "La combinaison « magnitude + énergie » est un bon point de départ. Les autres (écart-type, P95, direction, dispersion) ajoutent du contexte."
)
st.markdown(
    "Méthodes. **Isolation Forest** est souvent un premier choix robuste et rapide. "
    "**LOF** est utile si les anomalies sont isolées dans des régions de faible densité locale. "
    "L’**auto-encodeur** capture des relations non linéaires entre indicateurs et peut mieux séparer certains motifs."
)
st.markdown(
    "Projection 2D. La **PCA** donne une vue rapide et fiable ; le **t-SNE** (plus lent) sépare parfois mieux des structures fines. "
    "Dans le nuage de points, les anomalies en **rouge** sont souvent en marge ou regroupées dans des zones atypiques. "
    "Tu peux trier les vignettes en fonction de la projection pour explorer visuellement ces zones."
)
