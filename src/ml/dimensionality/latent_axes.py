# -*- coding: utf-8 -*-
"""
=============================================================================
Modulo de Reduccion Dimensional — Ejes Latentes de Necesidad Digital
=============================================================================

Aplica PCA y UMAP sobre los scores de factores estructurales y las
dimensiones compuestas originales para identificar ejes latentes de
necesidad de software educativo.

Entradas:
    - outputs/tables/scores_factores_establecimiento.csv
Salidas:
    - outputs/tables/ml_cargas_pca_ejes_latentes.csv
    - outputs/tables/ml_proyecciones_establecimientos.csv
    - outputs/figures/ml_varianza_explicada_pca.png
    - outputs/figures/ml_proyeccion_pca_2d.png
    - outputs/figures/ml_proyeccion_umap_2d.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

from src.config.settings import (
    TABLES_DIR,
    FIGURES_DIR,
    DPI,
    RANDOM_STATE,
)

# =========================================================================
# CONSTANTES DEL MODULO
# =========================================================================
COLS_SCORES = [
    "SCORE_GESTION_LIDERAZGO",
    "SCORE_CULTURA_INNOVACION",
    "SCORE_APROPIACION_PEDAGOGICA",
    "SCORE_CAPACIDADES_DIGITALES",
    "SCORE_INFRAESTRUCTURA_ACCESO",
]

COLS_DIMENSIONES = [
    "LIDERAZGO_ESCOLAR_PARA_LA_INNOVACION",
    "PRACTICAS_Y_PROCESOS_PARA_INNOVAR",
    "MENTALIDAD_FRENTE_A_LA_INNOVACION",
    "PROMOTORES_Y_BARRERAS_PARA_INNOVAR",
    "INNOVACION_EN_EL_PROCESO_ENSENANZA_Y_APRENDIZAJE",
    "ACCESO__INVERSO_",
    "MARCO_INSTITUCIONAL",
    "APOYO_AL_USO",
    "ACTIVIDADES",
    "HABILIDADES",
    "ACTITUDES",
    "EFECTOS",
]

ETIQUETAS_LEGIBLES = {
    "SCORE_GESTION_LIDERAZGO": "Gestion y Liderazgo",
    "SCORE_CULTURA_INNOVACION": "Cultura de Innovacion",
    "SCORE_APROPIACION_PEDAGOGICA": "Apropiacion Pedagogica",
    "SCORE_CAPACIDADES_DIGITALES": "Capacidades Digitales",
    "SCORE_INFRAESTRUCTURA_ACCESO": "Infraestructura y Acceso",
    "LIDERAZGO_ESCOLAR_PARA_LA_INNOVACION": "Liderazgo Escolar",
    "PRACTICAS_Y_PROCESOS_PARA_INNOVAR": "Practicas de Innovacion",
    "MENTALIDAD_FRENTE_A_LA_INNOVACION": "Mentalidad Innovadora",
    "PROMOTORES_Y_BARRERAS_PARA_INNOVAR": "Promotores/Barreras",
    "INNOVACION_EN_EL_PROCESO_ENSENANZA_Y_APRENDIZAJE": "Innov. Ensenanza-Aprendizaje",
    "ACCESO__INVERSO_": "Acceso (inverso)",
    "MARCO_INSTITUCIONAL": "Marco Institucional",
    "APOYO_AL_USO": "Apoyo al Uso",
    "ACTIVIDADES": "Actividades",
    "HABILIDADES": "Habilidades",
    "ACTITUDES": "Actitudes",
    "EFECTOS": "Efectos",
}


# =========================================================================
# FUNCIONES PRINCIPALES
# =========================================================================

def cargar_scores() -> pd.DataFrame:
    """
    Carga el CSV de scores por establecimiento generado en el paso 3.

    Retorna
    -------
    pd.DataFrame
        DataFrame con scores de factores y dimensiones originales.
    """
    ruta = os.path.join(TABLES_DIR, "scores_factores_establecimiento.csv")
    df = pd.read_csv(ruta)
    print(f"  [CARGA] {df.shape[0]} establecimientos desde {ruta}")
    return df


def ejecutar_pca_factores(df: pd.DataFrame) -> dict:
    """
    Aplica PCA sobre los 5 scores de factores estructurales.
    Identifica ejes latentes principales y sus cargas.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame con scores por factor.

    Retorna
    -------
    dict
        Diccionario con:
        - "pca": objeto PCA ajustado
        - "cargas": pd.DataFrame con cargas por componente
        - "proyecciones": np.ndarray con coordenadas PCA
        - "varianza": array con varianza explicada por componente
        - "mask_validos": mascara de filas validas
    """
    cols_disponibles = [c for c in COLS_SCORES if c in df.columns]
    mask = df[cols_disponibles].notna().all(axis=1)
    X = df.loc[mask, cols_disponibles].values

    pca = PCA(random_state=RANDOM_STATE)
    proyecciones = pca.fit_transform(X)

    etiquetas = [ETIQUETAS_LEGIBLES.get(c, c) for c in cols_disponibles]
    n_comp = pca.n_components_

    cargas = pd.DataFrame(
        pca.components_.T,
        index=etiquetas,
        columns=[f"Eje_{i+1}" for i in range(n_comp)],
    )
    cargas["variable_original"] = cols_disponibles

    print(f"  [PCA FACTORES] {n_comp} componentes extraidos")
    for i in range(n_comp):
        print(f"    Eje {i+1}: {pca.explained_variance_ratio_[i]:.1%} varianza")

    return {
        "pca": pca,
        "cargas": cargas,
        "proyecciones": proyecciones,
        "varianza": pca.explained_variance_ratio_,
        "mask_validos": mask,
    }


def ejecutar_pca_dimensiones(df: pd.DataFrame) -> dict:
    """
    Aplica PCA sobre las 12 dimensiones compuestas originales.
    Ofrece mayor granularidad que el PCA sobre factores agregados.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame con dimensiones compuestas.

    Retorna
    -------
    dict
        Mismo formato que ejecutar_pca_factores.
    """
    cols_disponibles = [c for c in COLS_DIMENSIONES if c in df.columns]
    mask = df[cols_disponibles].notna().all(axis=1)
    X = df.loc[mask, cols_disponibles].values

    # Estandarizar antes de PCA (dimensiones en escalas distintas)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(random_state=RANDOM_STATE)
    proyecciones = pca.fit_transform(X_scaled)

    etiquetas = [ETIQUETAS_LEGIBLES.get(c, c) for c in cols_disponibles]
    n_comp = pca.n_components_

    cargas = pd.DataFrame(
        pca.components_.T,
        index=etiquetas,
        columns=[f"Eje_{i+1}" for i in range(n_comp)],
    )
    cargas["variable_original"] = cols_disponibles

    print(f"  [PCA DIMENSIONES] {n_comp} componentes extraidos")
    for i in range(min(5, n_comp)):
        print(f"    Eje {i+1}: {pca.explained_variance_ratio_[i]:.1%} varianza")

    return {
        "pca": pca,
        "cargas": cargas,
        "proyecciones": proyecciones,
        "varianza": pca.explained_variance_ratio_,
        "mask_validos": mask,
    }


def ejecutar_umap(df: pd.DataFrame) -> dict:
    """
    Aplica UMAP sobre los scores de factores para proyeccion 2D no lineal.
    Complementa al PCA detectando estructuras manifold que PCA no captura.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame con scores por factor.

    Retorna
    -------
    dict
        Diccionario con:
        - "embedding": np.ndarray (n, 2) con coordenadas UMAP
        - "mask_validos": mascara de filas validas
    """
    try:
        import umap
    except ImportError:
        print("  [ADVERTENCIA] umap-learn no disponible. Omitiendo UMAP.")
        return None

    cols_disponibles = [c for c in COLS_SCORES if c in df.columns]
    mask = df[cols_disponibles].notna().all(axis=1)
    X = df.loc[mask, cols_disponibles].values

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.3,
        metric="euclidean",
        random_state=RANDOM_STATE,
    )
    embedding = reducer.fit_transform(X)

    print(f"  [UMAP] Proyeccion 2D generada para {X.shape[0]} establecimientos")
    return {"embedding": embedding, "mask_validos": mask}


def interpretar_ejes(cargas: pd.DataFrame, n_top: int = 3) -> pd.DataFrame:
    """
    Genera una tabla de interpretacion de cada eje PCA indicando
    las variables con mayor carga (positiva y negativa).

    Parametros
    ----------
    cargas : pd.DataFrame
        DataFrame de cargas PCA.
    n_top : int
        Cantidad de variables top a reportar por eje.

    Retorna
    -------
    pd.DataFrame
        Tabla de interpretacion con columnas:
        eje, tipo_necesidad_sugerida, variables_positivas, variables_negativas
    """
    cols_ejes = [c for c in cargas.columns if c.startswith("Eje_")]
    interpretaciones = []

    for eje in cols_ejes:
        vals = cargas[eje].drop("variable_original", errors="ignore")
        positivas = vals.nlargest(n_top)
        negativas = vals.nsmallest(n_top)

        # Interpretacion automatica basada en variables dominantes
        tipo = _inferir_tipo_necesidad(positivas.index.tolist(), negativas.index.tolist())

        interpretaciones.append({
            "eje": eje,
            "tipo_necesidad_sugerida": tipo,
            "variables_carga_positiva": "; ".join(
                [f"{v} ({positivas[v]:+.3f})" for v in positivas.index]
            ),
            "variables_carga_negativa": "; ".join(
                [f"{v} ({negativas[v]:+.3f})" for v in negativas.index]
            ),
        })

    return pd.DataFrame(interpretaciones)


def _inferir_tipo_necesidad(vars_positivas: list, vars_negativas: list) -> str:
    """
    Heuristica de interpretacion: asigna un tipo de necesidad de software
    basado en las variables dominantes del eje.
    """
    todas = " ".join(vars_positivas + vars_negativas).lower()

    if any(k in todas for k in ["gestion", "liderazgo", "institucional", "marco"]):
        if any(k in todas for k in ["pedagogica", "ensenanza", "actividades"]):
            return "Software de gestion pedagogica integrada"
        return "Software de gestion escolar y liderazgo digital"

    if any(k in todas for k in ["pedagogica", "ensenanza", "actividades", "apoyo"]):
        if any(k in todas for k in ["habilidades", "capacidades", "efectos"]):
            return "Plataforma de desarrollo de competencias docentes"
        return "Herramientas de apoyo a la innovacion pedagogica"

    if any(k in todas for k in ["habilidades", "capacidades", "efectos"]):
        return "Plataforma de capacitacion digital"

    if any(k in todas for k in ["acceso", "infraestructura", "inverso"]):
        return "Solucion de diagnostico y monitoreo de infraestructura"

    if any(k in todas for k in ["cultura", "mentalidad", "actitudes", "promotores"]):
        return "Herramientas de gestion del cambio y cultura digital"

    return "Solucion transversal de digitalizacion"


# =========================================================================
# FUNCIONES DE VISUALIZACION
# =========================================================================

def grafico_varianza_explicada(varianza_factores: np.ndarray,
                                varianza_dims: np.ndarray) -> str:
    """
    Genera grafico comparativo de varianza explicada PCA (factores vs dimensiones).

    Retorna
    -------
    str
        Ruta del grafico generado.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel izquierdo: factores
    n_f = len(varianza_factores)
    acum_f = np.cumsum(varianza_factores)
    ax1.bar(range(1, n_f + 1), varianza_factores, alpha=0.7, color="#3498db", label="Individual")
    ax1.plot(range(1, n_f + 1), acum_f, "ro-", label="Acumulada")
    ax1.axhline(y=0.80, color="gray", linestyle="--", alpha=0.5, label="80% referencia")
    ax1.set_xlabel("Componente")
    ax1.set_ylabel("Proporcion de varianza")
    ax1.set_title("PCA sobre Factores Estructurales (5 vars)")
    ax1.legend(fontsize=9)
    ax1.set_xticks(range(1, n_f + 1))

    # Panel derecho: dimensiones
    n_d = len(varianza_dims)
    acum_d = np.cumsum(varianza_dims)
    ax2.bar(range(1, n_d + 1), varianza_dims, alpha=0.7, color="#e67e22", label="Individual")
    ax2.plot(range(1, n_d + 1), acum_d, "ro-", label="Acumulada")
    ax2.axhline(y=0.80, color="gray", linestyle="--", alpha=0.5, label="80% referencia")
    ax2.set_xlabel("Componente")
    ax2.set_ylabel("Proporcion de varianza")
    ax2.set_title("PCA sobre Dimensiones Compuestas (12 vars)")
    ax2.legend(fontsize=9)
    ax2.set_xticks(range(1, n_d + 1))

    plt.suptitle("Varianza Explicada — Ejes Latentes de Necesidad Digital", fontsize=13, y=1.02)
    plt.tight_layout()

    ruta = os.path.join(FIGURES_DIR, "ml_varianza_explicada_pca.png")
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [GRAFICO] {ruta}")
    return ruta


def grafico_proyeccion_pca(df: pd.DataFrame, proyecciones: np.ndarray,
                            mask: pd.Series, varianza: np.ndarray) -> str:
    """
    Genera scatter 2D de la proyeccion PCA coloreada por zona y tipologia.

    Retorna
    -------
    str
        Ruta del grafico generado.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    df_plot = df.loc[mask].copy()
    df_plot["PC1"] = proyecciones[:, 0]
    df_plot["PC2"] = proyecciones[:, 1]

    # Panel 1: por zona
    for zona in df_plot["zona"].dropna().unique():
        subset = df_plot[df_plot["zona"] == zona]
        ax1.scatter(subset["PC1"], subset["PC2"], alpha=0.4, s=20, label=zona)
    ax1.set_xlabel(f"Eje 1 ({varianza[0]:.1%} varianza)")
    ax1.set_ylabel(f"Eje 2 ({varianza[1]:.1%} varianza)")
    ax1.set_title("Proyeccion PCA por Zona")
    ax1.legend()

    # Panel 2: por tipologia (si existe)
    if "tipologia" in df_plot.columns:
        for tip in df_plot["tipologia"].dropna().unique():
            subset = df_plot[df_plot["tipologia"] == tip]
            ax2.scatter(subset["PC1"], subset["PC2"], alpha=0.4, s=20, label=tip)
        ax2.set_xlabel(f"Eje 1 ({varianza[0]:.1%} varianza)")
        ax2.set_ylabel(f"Eje 2 ({varianza[1]:.1%} varianza)")
        ax2.set_title("Proyeccion PCA por Tipologia")
        ax2.legend()
    else:
        ax2.set_visible(False)

    plt.suptitle("Ejes Latentes — Proyeccion de Establecimientos", fontsize=13, y=1.02)
    plt.tight_layout()

    ruta = os.path.join(FIGURES_DIR, "ml_proyeccion_pca_2d.png")
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [GRAFICO] {ruta}")
    return ruta


def grafico_umap(df: pd.DataFrame, embedding: np.ndarray, mask: pd.Series) -> str:
    """
    Genera scatter 2D de la proyeccion UMAP coloreada por zona y tipologia.

    Retorna
    -------
    str
        Ruta del grafico generado.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    df_plot = df.loc[mask].copy()
    df_plot["UMAP1"] = embedding[:, 0]
    df_plot["UMAP2"] = embedding[:, 1]

    # Panel 1: por zona
    for zona in df_plot["zona"].dropna().unique():
        subset = df_plot[df_plot["zona"] == zona]
        ax1.scatter(subset["UMAP1"], subset["UMAP2"], alpha=0.4, s=20, label=zona)
    ax1.set_xlabel("UMAP Dimension 1")
    ax1.set_ylabel("UMAP Dimension 2")
    ax1.set_title("Proyeccion UMAP por Zona")
    ax1.legend()

    # Panel 2: por tipologia
    if "tipologia" in df_plot.columns:
        for tip in df_plot["tipologia"].dropna().unique():
            subset = df_plot[df_plot["tipologia"] == tip]
            ax2.scatter(subset["UMAP1"], subset["UMAP2"], alpha=0.4, s=20, label=tip)
        ax2.set_xlabel("UMAP Dimension 1")
        ax2.set_ylabel("UMAP Dimension 2")
        ax2.set_title("Proyeccion UMAP por Tipologia")
        ax2.legend()
    else:
        ax2.set_visible(False)

    plt.suptitle("Proyeccion UMAP — Estructura No Lineal del Sistema", fontsize=13, y=1.02)
    plt.tight_layout()

    ruta = os.path.join(FIGURES_DIR, "ml_proyeccion_umap_2d.png")
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [GRAFICO] {ruta}")
    return ruta


def grafico_cargas_pca(cargas: pd.DataFrame, titulo_sufijo: str = "") -> str:
    """
    Genera heatmap de cargas PCA para los primeros ejes relevantes.

    Retorna
    -------
    str
        Ruta del grafico generado.
    """
    cols_ejes = [c for c in cargas.columns if c.startswith("Eje_")]
    # Mostrar solo ejes que expliquen varianza relevante (max 6)
    cols_mostrar = cols_ejes[:min(6, len(cols_ejes))]
    datos_plot = cargas[cols_mostrar].copy()
    datos_plot.index = cargas.index if "variable_original" not in cargas.columns else [
        ETIQUETAS_LEGIBLES.get(v, v) for v in cargas.get("variable_original", cargas.index)
    ]

    # Limpiar indice si contiene variable_original como fila
    if "variable_original" in datos_plot.index:
        datos_plot = datos_plot.drop("variable_original", errors="ignore")

    fig, ax = plt.subplots(figsize=(10, max(6, len(datos_plot) * 0.5)))
    sns.heatmap(
        datos_plot.astype(float), annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, ax=ax, linewidths=0.5, cbar_kws={"label": "Carga"}
    )
    ax.set_title(f"Cargas PCA — Ejes Latentes de Necesidad Digital{titulo_sufijo}")
    ax.set_xlabel("Eje Latente")
    ax.set_ylabel("Variable")

    plt.tight_layout()
    sufijo = titulo_sufijo.replace(" ", "_").replace("(", "").replace(")", "").lower()
    ruta = os.path.join(FIGURES_DIR, f"ml_cargas_pca_ejes{sufijo}.png")
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [GRAFICO] {ruta}")
    return ruta


# =========================================================================
# FUNCION DE GUARDADO
# =========================================================================

def guardar_resultados_dimensionalidad(
    cargas_factores: pd.DataFrame,
    cargas_dims: pd.DataFrame,
    interpretaciones: pd.DataFrame,
    df_original: pd.DataFrame,
    proy_factores: np.ndarray,
    mask_factores: pd.Series,
    proy_umap: np.ndarray = None,
    mask_umap: pd.Series = None,
) -> list:
    """
    Guarda todas las tablas generadas por el modulo de dimensionalidad.

    Retorna
    -------
    list
        Lista de rutas de archivos guardados.
    """
    rutas = []

    # 1. Cargas PCA factores
    ruta_cf = os.path.join(TABLES_DIR, "ml_cargas_pca_factores.csv")
    cargas_factores.to_csv(ruta_cf, encoding="utf-8-sig")
    rutas.append(ruta_cf)
    print(f"  [GUARDADO] {ruta_cf}")

    # 2. Cargas PCA dimensiones
    ruta_cd = os.path.join(TABLES_DIR, "ml_cargas_pca_dimensiones.csv")
    cargas_dims.to_csv(ruta_cd, encoding="utf-8-sig")
    rutas.append(ruta_cd)
    print(f"  [GUARDADO] {ruta_cd}")

    # 3. Interpretaciones de ejes
    ruta_int = os.path.join(TABLES_DIR, "ml_interpretacion_ejes_latentes.csv")
    interpretaciones.to_csv(ruta_int, index=False, encoding="utf-8-sig")
    rutas.append(ruta_int)
    print(f"  [GUARDADO] {ruta_int}")

    # 4. Proyecciones por establecimiento
    df_proy = df_original.loc[mask_factores, ["ID", "zona"]].copy()
    df_proy = df_proy.reset_index(drop=True)
    for i in range(proy_factores.shape[1]):
        df_proy[f"PC{i+1}"] = proy_factores[:, i]

    if proy_umap is not None and mask_umap is not None:
        # UMAP puede tener mascara distinta; alinear por indice
        df_umap = df_original.loc[mask_umap, ["ID"]].copy().reset_index(drop=True)
        df_umap["UMAP1"] = proy_umap[:, 0]
        df_umap["UMAP2"] = proy_umap[:, 1]
        df_proy = df_proy.merge(df_umap, on="ID", how="left")

    ruta_proy = os.path.join(TABLES_DIR, "ml_proyecciones_establecimientos.csv")
    df_proy.to_csv(ruta_proy, index=False, encoding="utf-8-sig")
    rutas.append(ruta_proy)
    print(f"  [GUARDADO] {ruta_proy}")

    return rutas


# =========================================================================
# ORQUESTADOR DEL MODULO
# =========================================================================

def ejecutar_dimensionalidad() -> dict:
    """
    Funcion principal del modulo. Ejecuta PCA (factores + dimensiones),
    UMAP, genera interpretaciones, graficos y tablas.

    Retorna
    -------
    dict
        Resultados del modulo para uso en pasos posteriores.
    """
    print("\n  --- Reduccion Dimensional: Ejes Latentes ---")

    # Cargar datos
    df = cargar_scores()

    # Enriquecer con tipologia del clustering si existe
    ruta_clust = os.path.join(TABLES_DIR, "clustering_establecimientos.csv")
    if os.path.exists(ruta_clust):
        df_clust = pd.read_csv(ruta_clust)[["ID", "tipologia"]]
        df = df.merge(df_clust, on="ID", how="left")

    # PCA sobre factores
    res_pca_fact = ejecutar_pca_factores(df)

    # PCA sobre dimensiones compuestas
    res_pca_dims = ejecutar_pca_dimensiones(df)

    # UMAP
    res_umap = ejecutar_umap(df)

    # Interpretacion de ejes (sobre dimensiones, mayor granularidad)
    interpretaciones = interpretar_ejes(res_pca_dims["cargas"])
    print("\n  Interpretacion de ejes latentes (dimensiones):")
    for _, row in interpretaciones.iterrows():
        print(f"    {row['eje']}: {row['tipo_necesidad_sugerida']}")

    # Graficos
    rutas_graficos = []
    rg1 = grafico_varianza_explicada(res_pca_fact["varianza"], res_pca_dims["varianza"])
    rutas_graficos.append(rg1)

    rg2 = grafico_proyeccion_pca(
        df, res_pca_fact["proyecciones"],
        res_pca_fact["mask_validos"], res_pca_fact["varianza"]
    )
    rutas_graficos.append(rg2)

    rg3 = grafico_cargas_pca(res_pca_dims["cargas"], " (dimensiones)")
    rutas_graficos.append(rg3)

    rg4 = grafico_cargas_pca(res_pca_fact["cargas"], " (factores)")
    rutas_graficos.append(rg4)

    if res_umap is not None:
        rg5 = grafico_umap(df, res_umap["embedding"], res_umap["mask_validos"])
        rutas_graficos.append(rg5)

    # Guardar tablas
    umap_emb = res_umap["embedding"] if res_umap else None
    umap_mask = res_umap["mask_validos"] if res_umap else None

    rutas_tablas = guardar_resultados_dimensionalidad(
        cargas_factores=res_pca_fact["cargas"],
        cargas_dims=res_pca_dims["cargas"],
        interpretaciones=interpretaciones,
        df_original=df,
        proy_factores=res_pca_fact["proyecciones"],
        mask_factores=res_pca_fact["mask_validos"],
        proy_umap=umap_emb,
        mask_umap=umap_mask,
    )

    print(f"\n  [RESULTADO] {len(rutas_tablas)} tablas, {len(rutas_graficos)} graficos generados")

    return {
        "df": df,
        "pca_factores": res_pca_fact,
        "pca_dimensiones": res_pca_dims,
        "umap": res_umap,
        "interpretaciones": interpretaciones,
    }
