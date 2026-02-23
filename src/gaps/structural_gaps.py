# -*- coding: utf-8 -*-
"""
Modulo de deteccion de brechas estructurales.
Identifica desalineaciones entre dimensiones de la digitalizacion escolar,
comparando factores estructurales entre zonas, dependencias y estratos.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.config.settings import (
    ETIQUETAS_FACTORES,
    ETIQUETAS_ZONA,
    FIGSIZE_BAR,
    FIGSIZE_BOX,
    FIGSIZE_HEATMAP,
    DPI,
    PALETA_COLORES,
    TABLES_DIR,
    FIGURES_DIR,
    UMBRAL_BRECHA,
)


def detectar_brechas(df_factores: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica brechas estructurales entre factores por zona (urbano/rural),
    dependencia administrativa y estrato analitico.

    Parametros
    ----------
    df_factores : pd.DataFrame
        DataFrame con scores por factor y variables de estratificacion.

    Retorna
    -------
    pd.DataFrame
        Tabla de brechas con magnitud, direccion y significancia.
    """
    cols_scores = [c for c in df_factores.columns if c.startswith("SCORE_")]
    if not cols_scores:
        print("  [ERROR] No se encontraron columnas de scores.")
        return pd.DataFrame()

    brechas = []

    # --- Brechas por zona (urbano vs rural) ---
    brechas_zona = _calcular_brechas_por_grupo(df_factores, cols_scores, "zona")
    brechas.extend(brechas_zona)

    # --- Brechas por dependencia administrativa ---
    brechas_dep = _calcular_brechas_por_grupo(df_factores, cols_scores, "COD_DEPE2")
    brechas.extend(brechas_dep)

    # --- Brechas entre factores (desalineacion interna) ---
    brechas_internas = _calcular_desalineacion_interna(df_factores, cols_scores)
    brechas.extend(brechas_internas)

    df_brechas = pd.DataFrame(brechas)

    if not df_brechas.empty:
        df_brechas = df_brechas.sort_values("magnitud_brecha", ascending=False)
        print(f"  [RESULTADO] {len(df_brechas)} brechas identificadas")

    return df_brechas


def _calcular_brechas_por_grupo(
    df: pd.DataFrame, cols_scores: list, variable_grupo: str
) -> list:
    """Calcula brechas entre grupos para cada score."""
    brechas = []
    grupos = df[variable_grupo].dropna().unique()

    if len(grupos) < 2:
        return brechas

    for score in cols_scores:
        medias = df.groupby(variable_grupo)[score].mean()
        desv = df[score].std()

        if desv == 0 or pd.isna(desv):
            continue

        for i, g1 in enumerate(grupos):
            for g2 in grupos[i + 1:]:
                if pd.isna(medias.get(g1)) or pd.isna(medias.get(g2)):
                    continue
                diferencia = medias[g1] - medias[g2]
                magnitud = abs(diferencia) / desv  # Effect size (Cohen's d aprox.)

                nombre_score = score.replace("SCORE_", "").replace("_", " ").title()

                brechas.append({
                    "tipo_brecha": f"Entre grupos ({variable_grupo})",
                    "factor": nombre_score,
                    "grupo_1": str(g1),
                    "grupo_2": str(g2),
                    "media_grupo_1": round(medias[g1], 4),
                    "media_grupo_2": round(medias[g2], 4),
                    "diferencia": round(diferencia, 4),
                    "magnitud_brecha": round(magnitud, 4),
                    "es_significativa": magnitud >= UMBRAL_BRECHA,
                })

    return brechas


def _calcular_desalineacion_interna(df: pd.DataFrame, cols_scores: list) -> list:
    """
    Calcula desalineaciones internas: diferencia entre el factor mas alto
    y el mas bajo de cada establecimiento, identificando patrones sistemicos.
    """
    brechas = []

    if len(cols_scores) < 2:
        return brechas

    # Calcular rango interno (max - min) por establecimiento
    rangos = df[cols_scores].max(axis=1) - df[cols_scores].min(axis=1)
    rango_medio = rangos.mean()
    rango_mediano = rangos.median()

    # Identificar factores sistematicamente altos y bajos
    medias_factores = df[cols_scores].mean()
    factor_max = medias_factores.idxmax()
    factor_min = medias_factores.idxmin()

    nombre_max = factor_max.replace("SCORE_", "").replace("_", " ").title()
    nombre_min = factor_min.replace("SCORE_", "").replace("_", " ").title()

    brechas.append({
        "tipo_brecha": "Desalineacion interna (sistema)",
        "factor": f"{nombre_max} vs {nombre_min}",
        "grupo_1": nombre_max,
        "grupo_2": nombre_min,
        "media_grupo_1": round(medias_factores[factor_max], 4),
        "media_grupo_2": round(medias_factores[factor_min], 4),
        "diferencia": round(medias_factores[factor_max] - medias_factores[factor_min], 4),
        "magnitud_brecha": round(rango_medio, 4),
        "es_significativa": rango_medio >= UMBRAL_BRECHA,
    })

    return brechas


def generar_graficos_brechas(df_factores: pd.DataFrame, df_brechas: pd.DataFrame) -> list:
    """
    Genera graficos de brechas estructurales y los guarda en outputs/figures/.

    Retorna
    -------
    list
        Lista de rutas de graficos generados.
    """
    rutas = []
    cols_scores = [c for c in df_factores.columns if c.startswith("SCORE_")]

    if not cols_scores:
        return rutas

    # Configuracion global de graficos
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })

    # --- Grafico 1: Scores por zona ---
    ruta1 = _grafico_scores_por_zona(df_factores, cols_scores)
    if ruta1:
        rutas.append(ruta1)

    # --- Grafico 2: Boxplot de scores por zona ---
    ruta2 = _grafico_boxplot_zona(df_factores, cols_scores)
    if ruta2:
        rutas.append(ruta2)

    # --- Grafico 3: Perfil radar de factores por zona ---
    ruta3 = _grafico_perfil_factores(df_factores, cols_scores)
    if ruta3:
        rutas.append(ruta3)

    # --- Grafico 4: Desalineacion interna ---
    ruta4 = _grafico_desalineacion_interna(df_factores, cols_scores)
    if ruta4:
        rutas.append(ruta4)

    plt.close("all")
    return rutas


def _grafico_scores_por_zona(df: pd.DataFrame, cols_scores: list) -> str:
    """Genera grafico de barras comparativo de scores por zona."""
    try:
        etiquetas = [c.replace("SCORE_", "").replace("_", " ").title() for c in cols_scores]
        medias_zona = df.groupby("zona")[cols_scores].mean()

        fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
        x = np.arange(len(cols_scores))
        ancho = 0.35

        for i, zona in enumerate(medias_zona.index):
            valores = medias_zona.loc[zona].values
            ax.bar(x + i * ancho, valores, ancho, label=zona, alpha=0.85)

        ax.set_xlabel("Factor Estructural")
        ax.set_ylabel("Score Promedio (estandarizado)")
        ax.set_title("Brechas Estructurales por Zona (Urbano vs Rural)")
        ax.set_xticks(x + ancho / 2)
        ax.set_xticklabels(etiquetas, rotation=45, ha="right", fontsize=9)
        ax.legend()
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        plt.tight_layout()
        ruta = os.path.join(FIGURES_DIR, "brechas_scores_por_zona.png")
        fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  [GRAFICO] {ruta}")
        return ruta
    except Exception as e:
        print(f"  [ERROR] Grafico scores por zona: {e}")
        return None


def _grafico_boxplot_zona(df: pd.DataFrame, cols_scores: list) -> str:
    """Genera boxplot de distribucion de scores por zona."""
    try:
        df_melted = df[["zona"] + cols_scores].melt(
            id_vars="zona", var_name="factor", value_name="score"
        )
        df_melted["factor"] = df_melted["factor"].str.replace("SCORE_", "").str.replace("_", " ").str.title()

        fig, ax = plt.subplots(figsize=FIGSIZE_BOX)
        sns.boxplot(
            data=df_melted, x="factor", y="score", hue="zona",
            palette="Set2", ax=ax
        )
        ax.set_xlabel("Factor Estructural")
        ax.set_ylabel("Score (estandarizado)")
        ax.set_title("Distribucion de Scores Estructurales por Zona")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        plt.tight_layout()
        ruta = os.path.join(FIGURES_DIR, "boxplot_scores_por_zona.png")
        fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  [GRAFICO] {ruta}")
        return ruta
    except Exception as e:
        print(f"  [ERROR] Boxplot scores por zona: {e}")
        return None


def _grafico_perfil_factores(df: pd.DataFrame, cols_scores: list) -> str:
    """Genera grafico de perfil (radar simplificado como barras horizontales) por zona."""
    try:
        medias = df.groupby("zona")[cols_scores].mean()
        etiquetas = [c.replace("SCORE_", "").replace("_", " ").title() for c in cols_scores]

        fig, axes = plt.subplots(1, len(medias), figsize=(14, 6), sharey=True)
        if len(medias) == 1:
            axes = [axes]

        for ax, (zona, valores) in zip(axes, medias.iterrows()):
            colores = ["#2ecc71" if v >= 0 else "#e74c3c" for v in valores]
            ax.barh(etiquetas, valores, color=colores, alpha=0.8)
            ax.set_title(f"Zona: {zona}")
            ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Score Promedio")

        fig.suptitle("Perfil de Factores Estructurales por Zona", fontsize=14, y=1.02)
        plt.tight_layout()
        ruta = os.path.join(FIGURES_DIR, "perfil_factores_por_zona.png")
        fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  [GRAFICO] {ruta}")
        return ruta
    except Exception as e:
        print(f"  [ERROR] Perfil de factores: {e}")
        return None


def _grafico_desalineacion_interna(df: pd.DataFrame, cols_scores: list) -> str:
    """Genera histograma de desalineacion interna (rango entre factores por establecimiento)."""
    try:
        rangos = df[cols_scores].max(axis=1) - df[cols_scores].min(axis=1)
        rangos = rangos.dropna()

        fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
        ax.hist(rangos, bins=30, color="#3498db", alpha=0.7, edgecolor="white")
        ax.axvline(x=rangos.mean(), color="#e74c3c", linestyle="--",
                   label=f"Media: {rangos.mean():.2f}")
        ax.axvline(x=rangos.median(), color="#2ecc71", linestyle="--",
                   label=f"Mediana: {rangos.median():.2f}")
        ax.set_xlabel("Rango Interno (max - min de factores)")
        ax.set_ylabel("Frecuencia")
        ax.set_title("Desalineacion Interna entre Factores Estructurales")
        ax.legend()

        plt.tight_layout()
        ruta = os.path.join(FIGURES_DIR, "desalineacion_interna_factores.png")
        fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  [GRAFICO] {ruta}")
        return ruta
    except Exception as e:
        print(f"  [ERROR] Desalineacion interna: {e}")
        return None


def guardar_brechas(df_brechas: pd.DataFrame) -> str:
    """Guarda la tabla de brechas en outputs/tables/."""
    ruta = os.path.join(TABLES_DIR, "brechas_estructurales.csv")
    df_brechas.to_csv(ruta, index=False, encoding="utf-8-sig")
    print(f"  [GUARDADO] Brechas: {ruta}")
    return ruta
