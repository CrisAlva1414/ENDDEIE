# -*- coding: utf-8 -*-
"""
Modulo de segmentacion del sistema educativo.
Aplica clustering (KMeans) sobre los scores estructurales para identificar
tipologias de establecimientos segun su perfil de digitalizacion.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os

from src.config.settings import (
    RANDOM_STATE,
    MAX_CLUSTERS,
    MIN_CLUSTERS,
    ETIQUETAS_FACTORES,
    FIGSIZE_BAR,
    FIGSIZE_HEATMAP,
    DPI,
    PALETA_CLUSTER,
    TABLES_DIR,
    FIGURES_DIR,
)


def determinar_k_optimo(df_factores: pd.DataFrame) -> int:
    """
    Determina el numero optimo de clusters usando el metodo del codo
    y el coeficiente de silueta.

    Parametros
    ----------
    df_factores : pd.DataFrame
        DataFrame con scores por factor.

    Retorna
    -------
    int
        Numero optimo de clusters.
    """
    cols_scores = [c for c in df_factores.columns if c.startswith("SCORE_") and c != "SCORE_GLOBAL"]
    X = df_factores[cols_scores].dropna()

    if X.shape[0] < MAX_CLUSTERS:
        print(f"  [ADVERTENCIA] Pocos datos ({X.shape[0]}), usando k=3")
        return 3

    inercias = []
    siluetas = []
    rango_k = range(MIN_CLUSTERS, MAX_CLUSTERS + 1)

    for k in rango_k:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        etiquetas = kmeans.fit_predict(X)
        inercias.append(kmeans.inertia_)
        siluetas.append(silhouette_score(X, etiquetas))

    # Grafico del codo y silueta
    _grafico_seleccion_k(list(rango_k), inercias, siluetas)

    # Seleccionar k con mayor silueta
    k_optimo = list(rango_k)[np.argmax(siluetas)]
    print(f"  [K OPTIMO] k={k_optimo} (silueta={max(siluetas):.4f})")
    return k_optimo


def clusterizar_escuelas(df_factores: pd.DataFrame, k: int = None) -> pd.DataFrame:
    """
    Aplica clustering KMeans sobre los scores estructurales y asigna
    tipologias a cada establecimiento.

    Parametros
    ----------
    df_factores : pd.DataFrame
        DataFrame con scores por factor estructural.
    k : int, opcional
        Numero de clusters. Si no se especifica, se determina automaticamente.

    Retorna
    -------
    pd.DataFrame
        DataFrame original con columna adicional de cluster asignado.
    """
    cols_scores = [c for c in df_factores.columns if c.startswith("SCORE_") and c != "SCORE_GLOBAL"]

    if not cols_scores:
        print("  [ERROR] No se encontraron columnas de scores para clustering.")
        return df_factores

    # Preparar datos (solo filas completas)
    mask_validos = df_factores[cols_scores].notna().all(axis=1)
    X = df_factores.loc[mask_validos, cols_scores].values
    print(f"  [DATOS] {X.shape[0]} establecimientos con datos completos para clustering")

    if k is None:
        k = determinar_k_optimo(df_factores)

    # Aplicar KMeans
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    etiquetas = kmeans.fit_predict(X)

    # Calcular silueta
    sil = silhouette_score(X, etiquetas)
    print(f"  [CLUSTERING] k={k}, silueta={sil:.4f}")

    # Asignar cluster al DataFrame
    df_resultado = df_factores.copy()
    df_resultado["cluster"] = np.nan
    df_resultado.loc[mask_validos, "cluster"] = etiquetas
    df_resultado["cluster"] = df_resultado["cluster"].astype("Int64")

    # Asignar nombres descriptivos a los clusters
    df_resultado = _asignar_tipologias(df_resultado, cols_scores)

    return df_resultado


def generar_perfiles_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera perfiles descriptivos de cada cluster, incluyendo
    medias de scores, composicion por zona y dependencia.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame con cluster asignado.

    Retorna
    -------
    pd.DataFrame
        Perfiles de cada cluster.
    """
    cols_scores = [c for c in df.columns if c.startswith("SCORE_")]
    if "cluster" not in df.columns:
        return pd.DataFrame()

    # Medias de scores por cluster
    perfiles_scores = df.groupby("cluster")[cols_scores].mean()

    # Composicion por zona
    comp_zona = pd.crosstab(df["cluster"], df["zona"], normalize="index") * 100

    # Composicion por dependencia
    comp_dep = pd.crosstab(df["cluster"], df["COD_DEPE2"], normalize="index") * 100

    # Tamano de cada cluster
    tamanos = df.groupby("cluster").size().rename("n_establecimientos")

    # Combinar
    perfiles = perfiles_scores.join(tamanos)

    # Agregar composicion de zona
    for col in comp_zona.columns:
        perfiles[f"pct_zona_{col}"] = comp_zona[col]

    print(f"  [PERFILES] {len(perfiles)} clusters generados")
    return perfiles


def generar_graficos_clustering(df: pd.DataFrame) -> list:
    """
    Genera graficos del analisis de clustering.

    Retorna
    -------
    list
        Lista de rutas de graficos generados.
    """
    rutas = []
    cols_scores = [c for c in df.columns if c.startswith("SCORE_") and c != "SCORE_GLOBAL"]

    if "cluster" not in df.columns or not cols_scores:
        return rutas

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
    })

    # --- Grafico 1: Heatmap de perfiles de clusters ---
    ruta1 = _grafico_heatmap_clusters(df, cols_scores)
    if ruta1:
        rutas.append(ruta1)

    # --- Grafico 2: PCA scatter ---
    ruta2 = _grafico_pca_clusters(df, cols_scores)
    if ruta2:
        rutas.append(ruta2)

    # --- Grafico 3: Composicion de clusters por zona ---
    ruta3 = _grafico_composicion_zona(df)
    if ruta3:
        rutas.append(ruta3)

    plt.close("all")
    return rutas


def _grafico_seleccion_k(ks: list, inercias: list, siluetas: list) -> str:
    """Genera grafico de seleccion de k (codo + silueta)."""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(ks, inercias, "bo-", markersize=8)
        ax1.set_xlabel("Numero de clusters (k)")
        ax1.set_ylabel("Inercia")
        ax1.set_title("Metodo del Codo")

        ax2.plot(ks, siluetas, "rs-", markersize=8)
        ax2.set_xlabel("Numero de clusters (k)")
        ax2.set_ylabel("Coeficiente de Silueta")
        ax2.set_title("Analisis de Silueta")

        k_optimo = ks[np.argmax(siluetas)]
        ax2.axvline(x=k_optimo, color="green", linestyle="--",
                    label=f"k optimo = {k_optimo}")
        ax2.legend()

        plt.tight_layout()
        ruta = os.path.join(FIGURES_DIR, "seleccion_k_clusters.png")
        fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  [GRAFICO] {ruta}")
        return ruta
    except Exception as e:
        print(f"  [ERROR] Grafico seleccion k: {e}")
        return None


def _grafico_heatmap_clusters(df: pd.DataFrame, cols_scores: list) -> str:
    """Genera heatmap de medias de scores por cluster."""
    try:
        perfiles = df.groupby("cluster")[cols_scores].mean()
        etiquetas = [c.replace("SCORE_", "").replace("_", " ").title() for c in cols_scores]
        perfiles.columns = etiquetas

        fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)
        sns.heatmap(
            perfiles.T, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, ax=ax, linewidths=0.5
        )
        ax.set_title("Perfil de Factores Estructurales por Tipologia de Establecimiento")
        ax.set_xlabel("Tipologia (Cluster)")
        ax.set_ylabel("Factor Estructural")

        plt.tight_layout()
        ruta = os.path.join(FIGURES_DIR, "heatmap_perfiles_clusters.png")
        fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  [GRAFICO] {ruta}")
        return ruta
    except Exception as e:
        print(f"  [ERROR] Heatmap clusters: {e}")
        return None


def _grafico_pca_clusters(df: pd.DataFrame, cols_scores: list) -> str:
    """Genera scatter plot PCA de clusters."""
    try:
        mask = df[cols_scores].notna().all(axis=1) & df["cluster"].notna()
        X = df.loc[mask, cols_scores].values
        clusters = df.loc[mask, "cluster"].values

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        var_explicada = pca.explained_variance_ratio_

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=clusters, cmap=PALETA_CLUSTER,
            alpha=0.6, s=30, edgecolors="white", linewidth=0.5
        )
        ax.set_xlabel(f"Componente 1 ({var_explicada[0]:.1%} varianza)")
        ax.set_ylabel(f"Componente 2 ({var_explicada[1]:.1%} varianza)")
        ax.set_title("Segmentacion de Establecimientos (PCA)")
        plt.colorbar(scatter, ax=ax, label="Cluster")

        plt.tight_layout()
        ruta = os.path.join(FIGURES_DIR, "pca_clusters_establecimientos.png")
        fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  [GRAFICO] {ruta}")
        return ruta
    except Exception as e:
        print(f"  [ERROR] PCA clusters: {e}")
        return None


def _grafico_composicion_zona(df: pd.DataFrame) -> str:
    """Genera grafico de composicion de clusters por zona."""
    try:
        comp = pd.crosstab(df["cluster"], df["zona"], normalize="index") * 100

        fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
        comp.plot(kind="bar", stacked=True, ax=ax, colormap="Set2", alpha=0.85)
        ax.set_xlabel("Tipologia (Cluster)")
        ax.set_ylabel("Porcentaje (%)")
        ax.set_title("Composicion de Tipologias por Zona (Urbano/Rural)")
        ax.legend(title="Zona")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        plt.tight_layout()
        ruta = os.path.join(FIGURES_DIR, "composicion_clusters_zona.png")
        fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  [GRAFICO] {ruta}")
        return ruta
    except Exception as e:
        print(f"  [ERROR] Composicion zona: {e}")
        return None


def _asignar_tipologias(df: pd.DataFrame, cols_scores: list) -> pd.DataFrame:
    """
    Asigna nombres descriptivos a los clusters segun su perfil
    de scores (alto, medio, bajo en cada factor).
    Utiliza ranking relativo entre clusters para evitar duplicados.
    """
    if "cluster" not in df.columns:
        return df

    perfiles = df.groupby("cluster")[cols_scores].mean()
    score_medio = perfiles.mean(axis=1).sort_values(ascending=False)

    # Asignar tipologias por ranking relativo
    etiquetas_disponibles = ["Avanzado", "Emergente", "En transicion", "Rezagado"]
    n_clusters = len(score_medio)

    # Distribuir etiquetas segun posicion relativa
    if n_clusters <= len(etiquetas_disponibles):
        # Seleccionar etiquetas equidistantes del pool
        indices = np.linspace(0, len(etiquetas_disponibles) - 1, n_clusters, dtype=int)
        etiquetas_seleccionadas = [etiquetas_disponibles[i] for i in indices]
    else:
        # Mas clusters que etiquetas: agregar numeros
        etiquetas_seleccionadas = []
        for i in range(n_clusters):
            idx = min(i, len(etiquetas_disponibles) - 1)
            sufijo = f" ({i + 1})" if i >= len(etiquetas_disponibles) else ""
            etiquetas_seleccionadas.append(etiquetas_disponibles[idx] + sufijo)

    tipologias = {}
    for i, cluster_id in enumerate(score_medio.index):
        tipologias[cluster_id] = etiquetas_seleccionadas[i]

    df["tipologia"] = df["cluster"].map(tipologias)
    print(f"  [TIPOLOGIAS] {tipologias}")
    return df


def guardar_clustering(df: pd.DataFrame, perfiles: pd.DataFrame) -> tuple:
    """Guarda resultados del clustering."""
    ruta_asignacion = os.path.join(TABLES_DIR, "clustering_establecimientos.csv")
    cols_salida = ["ID", "COD_DEPE2", "RURAL_RBD", "zona", "cluster", "tipologia"] + \
                  [c for c in df.columns if c.startswith("SCORE_")]
    cols_existentes = [c for c in cols_salida if c in df.columns]
    df[cols_existentes].to_csv(ruta_asignacion, index=False, encoding="utf-8-sig")

    ruta_perfiles = os.path.join(TABLES_DIR, "perfiles_clusters.csv")
    perfiles.to_csv(ruta_perfiles, encoding="utf-8-sig")

    print(f"  [GUARDADO] Clustering: {ruta_asignacion}")
    print(f"  [GUARDADO] Perfiles: {ruta_perfiles}")
    return ruta_asignacion, ruta_perfiles
