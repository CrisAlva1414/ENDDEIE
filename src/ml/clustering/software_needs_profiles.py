# -*- coding: utf-8 -*-
"""
=============================================================================
Modulo de Perfiles de Necesidad de Software Educativo
=============================================================================

Realiza clustering orientado a identificar perfiles de necesidad de software
educativo, comparando segmentacion por scores directos vs. por patrones de
desalineacion interna. Traduce cada perfil a lenguaje de producto.

Entradas:
    - outputs/tables/scores_factores_establecimiento.csv
    - outputs/tables/clustering_establecimientos.csv
Salidas:
    - outputs/tables/ml_perfiles_necesidad_software.csv
    - outputs/tables/ml_asignacion_perfiles_software.csv
    - outputs/tables/ml_comparacion_enfoques_clustering.csv
    - outputs/figures/ml_perfiles_necesidad_software.png
    - outputs/figures/ml_comparacion_clustering_enfoques.png
    - outputs/figures/ml_radar_perfiles_necesidad.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import os

from src.config.settings import (
    TABLES_DIR,
    FIGURES_DIR,
    DPI,
    RANDOM_STATE,
)

# =========================================================================
# CONSTANTES
# =========================================================================
COLS_SCORES = [
    "SCORE_GESTION_LIDERAZGO",
    "SCORE_CULTURA_INNOVACION",
    "SCORE_APROPIACION_PEDAGOGICA",
    "SCORE_CAPACIDADES_DIGITALES",
    "SCORE_INFRAESTRUCTURA_ACCESO",
]

ETIQUETAS_CORTAS = {
    "SCORE_GESTION_LIDERAZGO": "Gestion",
    "SCORE_CULTURA_INNOVACION": "Cultura",
    "SCORE_APROPIACION_PEDAGOGICA": "Apropiacion",
    "SCORE_CAPACIDADES_DIGITALES": "Capacidades",
    "SCORE_INFRAESTRUCTURA_ACCESO": "Infraestructura",
}

# Traduccion de perfiles a lenguaje de producto
PERFILES_PRODUCTO = {
    "alto_gestion_bajo_capacidades": {
        "nombre": "Escuelas con gestion solida pero brecha digital docente",
        "necesidad_sw": "Plataforma de formacion y acompanamiento digital docente",
        "tipo": "pedagogico",
    },
    "alto_capacidades_baja_apropiacion": {
        "nombre": "Escuelas digitalmente habiles pero sin integracion curricular",
        "necesidad_sw": "Herramienta de planificacion e integracion curricular TIC",
        "tipo": "pedagogico",
    },
    "bajo_todo": {
        "nombre": "Escuelas con necesidad integral de digitalizacion",
        "necesidad_sw": "Suite integral de gestion escolar digital (admin + pedag.)",
        "tipo": "transversal",
    },
    "alto_todo": {
        "nombre": "Escuelas digitalmente maduras",
        "necesidad_sw": "Plataforma avanzada de innovacion y analitica educativa",
        "tipo": "administrativo",
    },
    "bajo_infraestructura": {
        "nombre": "Escuelas con deficit critico de infraestructura",
        "necesidad_sw": "Solucion offline-first con sincronizacion diferida",
        "tipo": "transversal",
    },
    "desalineado_cultura": {
        "nombre": "Escuelas con resistencia cultural al cambio digital",
        "necesidad_sw": "Herramienta de gestion del cambio y comunidades de practica",
        "tipo": "administrativo",
    },
}


# =========================================================================
# FUNCIONES DE CLUSTERING
# =========================================================================

def cargar_datos_clustering() -> pd.DataFrame:
    """
    Carga scores y, opcionalmente, clustering previo para comparacion.

    Retorna
    -------
    pd.DataFrame
        DataFrame con scores, zona e ID.
    """
    ruta_scores = os.path.join(TABLES_DIR, "scores_factores_establecimiento.csv")
    df = pd.read_csv(ruta_scores)

    ruta_clust = os.path.join(TABLES_DIR, "clustering_establecimientos.csv")
    if os.path.exists(ruta_clust):
        df_clust = pd.read_csv(ruta_clust)[["ID", "tipologia", "cluster"]]
        df_clust = df_clust.rename(columns={"tipologia": "tipologia_base", "cluster": "cluster_base"})
        df = df.merge(df_clust, on="ID", how="left")

    print(f"  [CARGA] {df.shape[0]} establecimientos")
    return df


def construir_features_desalineacion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye features de desalineacion interna para clustering alternativo.
    En lugar de usar los scores directos, usa las diferencias entre factores
    como input, lo que captura patrones de desequilibrio.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame con scores por factor.

    Retorna
    -------
    pd.DataFrame
        DataFrame con features de desalineacion.
    """
    cols = [c for c in COLS_SCORES if c in df.columns]
    df_feat = df[["ID"]].copy()

    # Rango interno (max - min)
    df_feat["DESALIN_RANGO"] = df[cols].max(axis=1) - df[cols].min(axis=1)

    # Diferencias entre pares estrategicos
    if "SCORE_APROPIACION_PEDAGOGICA" in df.columns and "SCORE_CAPACIDADES_DIGITALES" in df.columns:
        df_feat["DESALIN_APROPIACION_VS_CAPACIDADES"] = (
            df["SCORE_APROPIACION_PEDAGOGICA"] - df["SCORE_CAPACIDADES_DIGITALES"]
        )

    if "SCORE_GESTION_LIDERAZGO" in df.columns and "SCORE_APROPIACION_PEDAGOGICA" in df.columns:
        df_feat["DESALIN_GESTION_VS_APROPIACION"] = (
            df["SCORE_GESTION_LIDERAZGO"] - df["SCORE_APROPIACION_PEDAGOGICA"]
        )

    if "SCORE_CULTURA_INNOVACION" in df.columns and "SCORE_INFRAESTRUCTURA_ACCESO" in df.columns:
        df_feat["DESALIN_CULTURA_VS_INFRA"] = (
            df["SCORE_CULTURA_INNOVACION"] - df["SCORE_INFRAESTRUCTURA_ACCESO"]
        )

    if "SCORE_GESTION_LIDERAZGO" in df.columns and "SCORE_CAPACIDADES_DIGITALES" in df.columns:
        df_feat["DESALIN_GESTION_VS_CAPACIDADES"] = (
            df["SCORE_GESTION_LIDERAZGO"] - df["SCORE_CAPACIDADES_DIGITALES"]
        )

    # Score global como nivel general
    if "SCORE_GLOBAL" in df.columns:
        df_feat["NIVEL_GLOBAL"] = df["SCORE_GLOBAL"]

    return df_feat


def clustering_por_scores(df: pd.DataFrame, k_range: tuple = (2, 7)) -> dict:
    """
    Clustering KMeans orientado a necesidades de software sobre scores directos.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame con scores.
    k_range : tuple
        Rango de k a evaluar (min, max).

    Retorna
    -------
    dict
        Resultado con etiquetas, perfiles, silueta y k optimo.
    """
    cols = [c for c in COLS_SCORES if c in df.columns]
    mask = df[cols].notna().all(axis=1)
    X = df.loc[mask, cols].values

    # Evaluar k
    mejor_k, mejor_sil = 3, -1
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        if sil > mejor_sil:
            mejor_k, mejor_sil = k, sil

    # Ajustar con k optimo
    km = KMeans(n_clusters=mejor_k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)

    print(f"  [CLUSTERING SCORES] k={mejor_k}, silueta={sil:.4f}")
    return {"labels": labels, "mask": mask, "k": mejor_k, "silueta": sil, "enfoque": "scores"}


def clustering_por_desalineacion(df: pd.DataFrame, df_feat: pd.DataFrame,
                                  k_range: tuple = (2, 7)) -> dict:
    """
    Clustering KMeans sobre features de desalineacion interna.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame original.
    df_feat : pd.DataFrame
        DataFrame con features de desalineacion.
    k_range : tuple
        Rango de k a evaluar.

    Retorna
    -------
    dict
        Resultado con etiquetas, perfiles, silueta y k optimo.
    """
    cols_feat = [c for c in df_feat.columns if c.startswith("DESALIN_") or c == "NIVEL_GLOBAL"]
    mask = df_feat[cols_feat].notna().all(axis=1)
    X = df_feat.loc[mask, cols_feat].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mejor_k, mejor_sil = 3, -1
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        if sil > mejor_sil:
            mejor_k, mejor_sil = k, sil

    km = KMeans(n_clusters=mejor_k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)

    print(f"  [CLUSTERING DESALINEACION] k={mejor_k}, silueta={sil:.4f}")
    return {"labels": labels, "mask": mask, "k": mejor_k, "silueta": sil, "enfoque": "desalineacion"}


# =========================================================================
# FUNCIONES DE PERFILAMIENTO
# =========================================================================

def generar_perfiles_necesidad(df: pd.DataFrame, labels: np.ndarray,
                                mask: pd.Series) -> pd.DataFrame:
    """
    Genera perfiles interpretables de necesidad de software a partir
    de los clusters, traducidos a lenguaje de producto.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame con scores.
    labels : np.ndarray
        Etiquetas de cluster.
    mask : pd.Series
        Mascara de filas validas.

    Retorna
    -------
    pd.DataFrame
        Tabla de perfiles con medias, tamano y tipo de software sugerido.
    """
    cols = [c for c in COLS_SCORES if c in df.columns]
    df_work = df.loc[mask].copy()
    df_work["perfil_cluster"] = labels

    # Medias por cluster
    perfiles = df_work.groupby("perfil_cluster")[cols].mean()
    tamanos = df_work.groupby("perfil_cluster").size().rename("n_establecimientos")
    perfiles = perfiles.join(tamanos)
    perfiles["pct_sistema"] = (perfiles["n_establecimientos"] / perfiles["n_establecimientos"].sum() * 100).round(1)

    # Composicion por zona
    if "zona" in df_work.columns:
        comp_zona = pd.crosstab(df_work["perfil_cluster"], df_work["zona"], normalize="index") * 100
        for col in comp_zona.columns:
            perfiles[f"pct_{col}"] = comp_zona[col].round(1)

    # Asignar nombre de perfil y tipo de software
    perfiles = _asignar_nombre_perfil(perfiles, cols)

    return perfiles


def _asignar_nombre_perfil(perfiles: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Asigna nombres descriptivos orientados a producto a cada perfil de cluster.
    Usa heuristicas basadas en el patron de scores.
    """
    nombres = []
    tipos_sw = []
    necesidades_sw = []

    for idx, row in perfiles.iterrows():
        scores = row[cols]
        media = scores.mean()
        factor_max = scores.idxmax()
        factor_min = scores.idxmin()
        val_max = scores.max()
        val_min = scores.min()
        rango = val_max - val_min

        # Clasificar patron
        if media > 0.3:
            nombre = "Digitalmente maduro"
            tipo = "administrativo"
            sw = "Plataforma avanzada de analitica e innovacion educativa"
        elif media < -0.3 and rango < 0.8:
            nombre = "Necesidad integral de digitalizacion"
            tipo = "transversal"
            sw = "Suite integral de gestion escolar digital"
        elif "INFRAESTRUCTURA" in factor_min and val_min < -1.0:
            nombre = "Deficit critico de infraestructura"
            tipo = "transversal"
            sw = "Solucion offline-first con sincronizacion diferida"
        elif "GESTION" in factor_max and "CAPACIDADES" in factor_min:
            nombre = "Gestion solida, brecha digital docente"
            tipo = "pedagogico"
            sw = "Plataforma de formacion digital docente"
        elif "CAPACIDADES" in factor_max and "APROPIACION" in factor_min:
            nombre = "Capacidades sin integracion curricular"
            tipo = "pedagogico"
            sw = "Herramienta de planificacion e integracion curricular TIC"
        elif "CULTURA" in factor_min:
            nombre = "Resistencia cultural al cambio digital"
            tipo = "administrativo"
            sw = "Herramienta de gestion del cambio y comunidades de practica"
        elif "APROPIACION" in factor_min:
            nombre = "Brecha en apropiacion pedagogica"
            tipo = "pedagogico"
            sw = "Plataforma de acompanamiento pedagogico con TIC"
        else:
            nombre = f"Perfil mixto (fortaleza: {ETIQUETAS_CORTAS.get(factor_max, factor_max)})"
            tipo = "transversal"
            sw = "Solucion adaptativa segun diagnostico institucional"

        nombres.append(nombre)
        tipos_sw.append(tipo)
        necesidades_sw.append(sw)

    perfiles["nombre_perfil"] = nombres
    perfiles["tipo_software"] = tipos_sw
    perfiles["solucion_sugerida"] = necesidades_sw

    return perfiles


def comparar_enfoques(res_scores: dict, res_desalin: dict,
                       df: pd.DataFrame) -> pd.DataFrame:
    """
    Compara los dos enfoques de clustering (scores vs desalineacion)
    en terminos de silueta, distribucion y concordancia.

    Retorna
    -------
    pd.DataFrame
        Tabla comparativa de ambos enfoques.
    """
    comparacion = pd.DataFrame([
        {
            "enfoque": "Scores directos",
            "k_optimo": res_scores["k"],
            "silueta": round(res_scores["silueta"], 4),
            "n_validos": res_scores["mask"].sum(),
        },
        {
            "enfoque": "Desalineacion interna",
            "k_optimo": res_desalin["k"],
            "silueta": round(res_desalin["silueta"], 4),
            "n_validos": res_desalin["mask"].sum(),
        },
    ])

    print(f"\n  Comparacion de enfoques de clustering:")
    print(f"    Scores directos:      k={res_scores['k']}, silueta={res_scores['silueta']:.4f}")
    print(f"    Desalineacion interna: k={res_desalin['k']}, silueta={res_desalin['silueta']:.4f}")

    return comparacion


# =========================================================================
# FUNCIONES DE VISUALIZACION
# =========================================================================

def grafico_perfiles_necesidad(perfiles: pd.DataFrame) -> str:
    """
    Genera heatmap de perfiles de necesidad de software.

    Retorna
    -------
    str
        Ruta del grafico.
    """
    cols_scores = [c for c in COLS_SCORES if c in perfiles.columns]
    etiquetas = [ETIQUETAS_CORTAS.get(c, c) for c in cols_scores]

    datos_plot = perfiles[cols_scores].copy()
    datos_plot.columns = etiquetas

    if "nombre_perfil" in perfiles.columns:
        datos_plot.index = perfiles["nombre_perfil"].values
    else:
        datos_plot.index = [f"Perfil {i}" for i in datos_plot.index]

    fig, ax = plt.subplots(figsize=(12, max(4, len(datos_plot) * 1.2)))
    sns.heatmap(
        datos_plot.astype(float), annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, ax=ax, linewidths=0.5,
        cbar_kws={"label": "Score estandarizado"}
    )
    ax.set_title("Perfiles de Necesidad de Software Educativo\n(clustering sobre scores estructurales)")
    ax.set_xlabel("Factor Estructural")
    ax.set_ylabel("Perfil de Necesidad")

    # Agregar tipo de software en anotacion lateral
    if "tipo_software" in perfiles.columns:
        for i, tipo in enumerate(perfiles["tipo_software"].values):
            ax.annotate(
                f"[{tipo}]", xy=(len(etiquetas) + 0.1, i + 0.5),
                fontsize=8, color="gray", va="center",
                annotation_clip=False,
            )

    plt.tight_layout()
    ruta = os.path.join(FIGURES_DIR, "ml_perfiles_necesidad_software.png")
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [GRAFICO] {ruta}")
    return ruta


def grafico_comparacion_enfoques(df: pd.DataFrame, res_scores: dict,
                                   res_desalin: dict) -> str:
    """
    Genera grafico comparativo de los dos enfoques de clustering.

    Retorna
    -------
    str
        Ruta del grafico.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    cols = [c for c in COLS_SCORES if c in df.columns]
    etiquetas = [ETIQUETAS_CORTAS.get(c, c) for c in cols]

    # Panel 1: distribucion de clusters por scores
    mask1 = res_scores["mask"]
    df1 = df.loc[mask1].copy()
    df1["cluster"] = res_scores["labels"]
    medias1 = df1.groupby("cluster")[cols].mean()
    medias1.columns = etiquetas
    medias1.T.plot(kind="bar", ax=ax1, alpha=0.8)
    ax1.set_title(f"Enfoque Scores (k={res_scores['k']}, sil={res_scores['silueta']:.3f})")
    ax1.set_ylabel("Score promedio")
    ax1.set_xlabel("Factor")
    ax1.legend(title="Cluster", fontsize=8)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.tick_params(axis="x", rotation=45)

    # Panel 2: distribucion de clusters por desalineacion
    mask2 = res_desalin["mask"]
    df2 = df.loc[mask2].copy()
    df2["cluster"] = res_desalin["labels"]
    medias2 = df2.groupby("cluster")[cols].mean()
    medias2.columns = etiquetas
    medias2.T.plot(kind="bar", ax=ax2, alpha=0.8)
    ax2.set_title(f"Enfoque Desalineacion (k={res_desalin['k']}, sil={res_desalin['silueta']:.3f})")
    ax2.set_ylabel("Score promedio")
    ax2.set_xlabel("Factor")
    ax2.legend(title="Cluster", fontsize=8)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.tick_params(axis="x", rotation=45)

    plt.suptitle("Comparacion de Enfoques de Clustering para Perfiles de Software", fontsize=13, y=1.02)
    plt.tight_layout()

    ruta = os.path.join(FIGURES_DIR, "ml_comparacion_clustering_enfoques.png")
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [GRAFICO] {ruta}")
    return ruta


def grafico_radar_perfiles(perfiles: pd.DataFrame) -> str:
    """
    Genera grafico tipo radar (polar) para cada perfil de necesidad.

    Retorna
    -------
    str
        Ruta del grafico.
    """
    cols_scores = [c for c in COLS_SCORES if c in perfiles.columns]
    etiquetas = [ETIQUETAS_CORTAS.get(c, c) for c in cols_scores]
    n_vars = len(etiquetas)

    if n_vars < 3:
        return None

    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    n_perfiles = len(perfiles)
    fig, axes = plt.subplots(1, n_perfiles, figsize=(5 * n_perfiles, 5),
                              subplot_kw=dict(polar=True))
    if n_perfiles == 1:
        axes = [axes]

    colores = plt.cm.Set2(np.linspace(0, 1, n_perfiles))

    for ax, (idx, row), color in zip(axes, perfiles.iterrows(), colores):
        valores = row[cols_scores].values.tolist()
        valores += valores[:1]

        ax.fill(angles, valores, alpha=0.25, color=color)
        ax.plot(angles, valores, "o-", linewidth=2, color=color, markersize=4)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(etiquetas, fontsize=8)

        nombre = row.get("nombre_perfil", f"Perfil {idx}")
        n_est = row.get("n_establecimientos", "?")
        pct = row.get("pct_sistema", "?")
        ax.set_title(f"{nombre}\n(n={n_est}, {pct}%)", fontsize=9, pad=20)

    plt.suptitle("Radar de Perfiles de Necesidad de Software Educativo", fontsize=13, y=1.05)
    plt.tight_layout()

    ruta = os.path.join(FIGURES_DIR, "ml_radar_perfiles_necesidad.png")
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [GRAFICO] {ruta}")
    return ruta


# =========================================================================
# FUNCIONES DE GUARDADO
# =========================================================================

def guardar_resultados_perfiles(
    perfiles: pd.DataFrame,
    df_asignacion: pd.DataFrame,
    comparacion: pd.DataFrame,
) -> list:
    """
    Guarda tablas del modulo de perfiles de software.

    Retorna
    -------
    list
        Lista de rutas guardadas.
    """
    rutas = []

    ruta_perf = os.path.join(TABLES_DIR, "ml_perfiles_necesidad_software.csv")
    perfiles.to_csv(ruta_perf, encoding="utf-8-sig")
    rutas.append(ruta_perf)
    print(f"  [GUARDADO] {ruta_perf}")

    ruta_asig = os.path.join(TABLES_DIR, "ml_asignacion_perfiles_software.csv")
    df_asignacion.to_csv(ruta_asig, index=False, encoding="utf-8-sig")
    rutas.append(ruta_asig)
    print(f"  [GUARDADO] {ruta_asig}")

    ruta_comp = os.path.join(TABLES_DIR, "ml_comparacion_enfoques_clustering.csv")
    comparacion.to_csv(ruta_comp, index=False, encoding="utf-8-sig")
    rutas.append(ruta_comp)
    print(f"  [GUARDADO] {ruta_comp}")

    return rutas


# =========================================================================
# ORQUESTADOR DEL MODULO
# =========================================================================

def ejecutar_perfiles_software() -> dict:
    """
    Funcion principal del modulo. Ejecuta clustering dual,
    genera perfiles de necesidad de software, graficos y tablas.

    Retorna
    -------
    dict
        Resultados del modulo.
    """
    print("\n  --- Perfiles de Necesidad de Software Educativo ---")

    # Cargar datos
    df = cargar_datos_clustering()

    # Construir features de desalineacion
    df_feat = construir_features_desalineacion(df)

    # Clustering por scores directos
    res_scores = clustering_por_scores(df)

    # Clustering por desalineacion
    res_desalin = clustering_por_desalineacion(df, df_feat)

    # Generar perfiles (usar el enfoque con mejor silueta como principal)
    if res_scores["silueta"] >= res_desalin["silueta"]:
        enfoque_principal = res_scores
        enfoque_nombre = "scores directos"
    else:
        enfoque_principal = res_desalin
        enfoque_nombre = "desalineacion interna"

    print(f"\n  Enfoque principal seleccionado: {enfoque_nombre}")

    perfiles = generar_perfiles_necesidad(df, enfoque_principal["labels"], enfoque_principal["mask"])

    print(f"\n  Perfiles de necesidad de software identificados:")
    for _, row in perfiles.iterrows():
        print(f"    - {row['nombre_perfil']} (n={row['n_establecimientos']}, {row['pct_sistema']}%)")
        print(f"      Tipo: {row['tipo_software']} | Solucion: {row['solucion_sugerida']}")

    # Tabla de asignacion
    df_asig = df.loc[enfoque_principal["mask"], ["ID", "zona"]].copy()
    df_asig = df_asig.reset_index(drop=True)
    df_asig["perfil_cluster"] = enfoque_principal["labels"]
    # Mapear nombre de perfil
    mapa_perfil = dict(zip(perfiles.index, perfiles["nombre_perfil"]))
    df_asig["nombre_perfil"] = df_asig["perfil_cluster"].map(mapa_perfil)
    mapa_tipo = dict(zip(perfiles.index, perfiles["tipo_software"]))
    df_asig["tipo_software"] = df_asig["perfil_cluster"].map(mapa_tipo)
    mapa_sol = dict(zip(perfiles.index, perfiles["solucion_sugerida"]))
    df_asig["solucion_sugerida"] = df_asig["perfil_cluster"].map(mapa_sol)

    # Comparacion de enfoques
    comparacion = comparar_enfoques(res_scores, res_desalin, df)

    # Graficos
    rutas_graficos = []
    rg1 = grafico_perfiles_necesidad(perfiles)
    rutas_graficos.append(rg1)

    rg2 = grafico_comparacion_enfoques(df, res_scores, res_desalin)
    rutas_graficos.append(rg2)

    rg3 = grafico_radar_perfiles(perfiles)
    if rg3:
        rutas_graficos.append(rg3)

    # Guardar
    rutas_tablas = guardar_resultados_perfiles(perfiles, df_asig, comparacion)

    print(f"\n  [RESULTADO] {len(rutas_tablas)} tablas, {len(rutas_graficos)} graficos generados")

    return {
        "perfiles": perfiles,
        "df_asignacion": df_asig,
        "comparacion": comparacion,
        "res_scores": res_scores,
        "res_desalin": res_desalin,
    }
