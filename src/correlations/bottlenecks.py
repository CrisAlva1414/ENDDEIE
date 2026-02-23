# -*- coding: utf-8 -*-
"""
Modulo de deteccion de cuellos de botella estructurales.
Analiza correlaciones Spearman entre factores estructurales para identificar
relaciones de dependencia y puntos de bloqueo en la digitalizacion escolar.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

from src.config.settings import (
    ETIQUETAS_FACTORES,
    UMBRAL_CORRELACION,
    FIGSIZE_HEATMAP,
    DPI,
    TABLES_DIR,
    FIGURES_DIR,
)


def analizar_correlaciones(df_factores: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la matriz de correlaciones Spearman entre factores estructurales
    e identifica relaciones significativas y cuellos de botella.

    Parametros
    ----------
    df_factores : pd.DataFrame
        DataFrame con scores por factor estructural.

    Retorna
    -------
    pd.DataFrame
        Matriz de correlacion Spearman con valores p.
    """
    cols_scores = [c for c in df_factores.columns if c.startswith("SCORE_")]

    if len(cols_scores) < 2:
        print("  [ERROR] Se requieren al menos 2 scores para correlaciones.")
        return pd.DataFrame()

    # Filtrar filas con datos completos
    df_limpio = df_factores[cols_scores].dropna()
    print(f"  [DATOS] {df_limpio.shape[0]} establecimientos con datos completos")

    # Calcular correlaciones Spearman
    n = len(cols_scores)
    corr_matrix = pd.DataFrame(np.zeros((n, n)), index=cols_scores, columns=cols_scores)
    pval_matrix = pd.DataFrame(np.zeros((n, n)), index=cols_scores, columns=cols_scores)

    for i, c1 in enumerate(cols_scores):
        for j, c2 in enumerate(cols_scores):
            if i == j:
                corr_matrix.iloc[i, j] = 1.0
                pval_matrix.iloc[i, j] = 0.0
            elif i < j:
                rho, pval = stats.spearmanr(
                    df_limpio[c1], df_limpio[c2], nan_policy="omit"
                )
                corr_matrix.iloc[i, j] = rho
                corr_matrix.iloc[j, i] = rho
                pval_matrix.iloc[i, j] = pval
                pval_matrix.iloc[j, i] = pval

    # Etiquetas legibles
    etiquetas = [c.replace("SCORE_", "").replace("_", " ").title() for c in cols_scores]
    corr_matrix.index = etiquetas
    corr_matrix.columns = etiquetas
    pval_matrix.index = etiquetas
    pval_matrix.columns = etiquetas

    return corr_matrix


def identificar_cuellos_botella(
    corr_matrix: pd.DataFrame, df_factores: pd.DataFrame
) -> pd.DataFrame:
    """
    Identifica cuellos de botella estructurales a partir de la matriz
    de correlaciones y el analisis de medianas por factor.

    Un cuello de botella se define como un factor con:
    - Scores sistematicamente bajos (mediana negativa)
    - Correlaciones fuertes con otros factores (efecto de arrastre)

    Parametros
    ----------
    corr_matrix : pd.DataFrame
        Matriz de correlacion Spearman.
    df_factores : pd.DataFrame
        DataFrame con scores por factor.

    Retorna
    -------
    pd.DataFrame
        Tabla de cuellos de botella con severidad y factores afectados.
    """
    cols_scores = [c for c in df_factores.columns if c.startswith("SCORE_")]
    medianas = df_factores[cols_scores].median()
    medias = df_factores[cols_scores].mean()

    cuellos = []

    for i, score in enumerate(cols_scores):
        nombre = score.replace("SCORE_", "").replace("_", " ").title()

        # Correlaciones fuertes con otros factores
        if nombre in corr_matrix.index:
            corrs = corr_matrix.loc[nombre].drop(nombre, errors="ignore")
            corrs_fuertes = corrs[corrs.abs() >= UMBRAL_CORRELACION]
            n_relaciones = len(corrs_fuertes)
            corr_media = corrs.abs().mean()
        else:
            n_relaciones = 0
            corr_media = 0

        # Indice de cuello de botella: combina nivel bajo + alta conectividad
        nivel = medias[score]
        severidad = -nivel * corr_media  # Mas severo si el nivel es bajo y las correlaciones altas

        cuellos.append({
            "factor": nombre,
            "media": round(medias[score], 4),
            "mediana": round(medianas[score], 4),
            "correlacion_media_abs": round(corr_media, 4),
            "n_relaciones_fuertes": n_relaciones,
            "factores_relacionados": ", ".join(corrs_fuertes.index.tolist()) if n_relaciones > 0 else "ninguno",
            "severidad_cuello_botella": round(severidad, 4),
        })

    df_cuellos = pd.DataFrame(cuellos)
    df_cuellos = df_cuellos.sort_values("severidad_cuello_botella", ascending=False)
    print(f"  [RESULTADO] {len(df_cuellos)} factores analizados como potenciales cuellos de botella")

    return df_cuellos


def analizar_correlaciones_por_zona(df_factores: pd.DataFrame) -> dict:
    """
    Calcula matrices de correlacion separadas por zona (urbano/rural)
    para identificar patrones diferenciados.

    Retorna
    -------
    dict
        Diccionario {zona: pd.DataFrame (matriz de correlacion)}.
    """
    resultado = {}
    cols_scores = [c for c in df_factores.columns if c.startswith("SCORE_")]

    for zona in df_factores["zona"].dropna().unique():
        df_zona = df_factores[df_factores["zona"] == zona][cols_scores].dropna()
        if df_zona.shape[0] < 10:
            continue

        corr = df_zona.corr(method="spearman")
        etiquetas = [c.replace("SCORE_", "").replace("_", " ").title() for c in cols_scores]
        corr.index = etiquetas
        corr.columns = etiquetas
        resultado[zona] = corr

    return resultado


def generar_graficos_correlaciones(
    corr_matrix: pd.DataFrame,
    df_cuellos: pd.DataFrame,
    corr_por_zona: dict = None,
) -> list:
    """
    Genera graficos de correlaciones y cuellos de botella.

    Retorna
    -------
    list
        Lista de rutas de graficos generados.
    """
    rutas = []

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
    })

    # --- Grafico 1: Heatmap de correlaciones ---
    ruta1 = _grafico_heatmap_correlaciones(corr_matrix)
    if ruta1:
        rutas.append(ruta1)

    # --- Grafico 2: Cuellos de botella ---
    ruta2 = _grafico_cuellos_botella(df_cuellos)
    if ruta2:
        rutas.append(ruta2)

    # --- Grafico 3: Comparacion de correlaciones por zona ---
    if corr_por_zona:
        ruta3 = _grafico_correlaciones_por_zona(corr_por_zona)
        if ruta3:
            rutas.append(ruta3)

    plt.close("all")
    return rutas


def _grafico_heatmap_correlaciones(corr_matrix: pd.DataFrame) -> str:
    """Genera heatmap de la matriz de correlaciones Spearman."""
    try:
        fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(
            corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            ax=ax, linewidths=0.5, square=True,
            cbar_kws={"label": "Correlacion Spearman"}
        )
        ax.set_title("Correlaciones Spearman entre Factores Estructurales")

        plt.tight_layout()
        ruta = os.path.join(FIGURES_DIR, "heatmap_correlaciones_spearman.png")
        fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  [GRAFICO] {ruta}")
        return ruta
    except Exception as e:
        print(f"  [ERROR] Heatmap correlaciones: {e}")
        return None


def _grafico_cuellos_botella(df_cuellos: pd.DataFrame) -> str:
    """Genera grafico de severidad de cuellos de botella."""
    try:
        fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)

        # Ordenar por severidad
        df_plot = df_cuellos.sort_values("severidad_cuello_botella", ascending=True)
        colores = ["#e74c3c" if s > 0 else "#2ecc71" for s in df_plot["severidad_cuello_botella"]]

        ax.barh(
            df_plot["factor"], df_plot["severidad_cuello_botella"],
            color=colores, alpha=0.8, edgecolor="white"
        )
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Severidad del Cuello de Botella")
        ax.set_title("Identificacion de Cuellos de Botella Estructurales\n"
                      "(mayor severidad = factor bajo con alta conectividad)")

        # Agregar anotaciones
        for i, (_, row) in enumerate(df_plot.iterrows()):
            ax.annotate(
                f"media={row['media']:.2f}",
                xy=(row["severidad_cuello_botella"], i),
                xytext=(5, 0), textcoords="offset points",
                fontsize=8, va="center"
            )

        plt.tight_layout()
        ruta = os.path.join(FIGURES_DIR, "cuellos_botella_estructurales.png")
        fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  [GRAFICO] {ruta}")
        return ruta
    except Exception as e:
        print(f"  [ERROR] Cuellos de botella: {e}")
        return None


def _grafico_correlaciones_por_zona(corr_por_zona: dict) -> str:
    """Genera heatmaps de correlaciones separados por zona."""
    try:
        n_zonas = len(corr_por_zona)
        fig, axes = plt.subplots(1, n_zonas, figsize=(8 * n_zonas, 8))
        if n_zonas == 1:
            axes = [axes]

        for ax, (zona, corr) in zip(axes, corr_por_zona.items()):
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
            sns.heatmap(
                corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                ax=ax, linewidths=0.5, square=True
            )
            ax.set_title(f"Correlaciones - Zona {zona}")

        plt.suptitle("Comparacion de Correlaciones entre Factores por Zona",
                      fontsize=14, y=1.02)
        plt.tight_layout()
        ruta = os.path.join(FIGURES_DIR, "correlaciones_por_zona.png")
        fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  [GRAFICO] {ruta}")
        return ruta
    except Exception as e:
        print(f"  [ERROR] Correlaciones por zona: {e}")
        return None


def guardar_correlaciones(
    corr_matrix: pd.DataFrame, df_cuellos: pd.DataFrame
) -> tuple:
    """Guarda resultados de correlaciones y cuellos de botella."""
    ruta_corr = os.path.join(TABLES_DIR, "matriz_correlaciones_spearman.csv")
    corr_matrix.to_csv(ruta_corr, encoding="utf-8-sig")

    ruta_cuellos = os.path.join(TABLES_DIR, "cuellos_botella_estructurales.csv")
    df_cuellos.to_csv(ruta_cuellos, index=False, encoding="utf-8-sig")

    print(f"  [GUARDADO] Correlaciones: {ruta_corr}")
    print(f"  [GUARDADO] Cuellos de botella: {ruta_cuellos}")
    return ruta_corr, ruta_cuellos
