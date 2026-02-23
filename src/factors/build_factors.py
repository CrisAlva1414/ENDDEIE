# -*- coding: utf-8 -*-
"""
Modulo de construccion de scores estructurales por establecimiento.
Integra datos de multiples actores (directores, coordinadores, docentes,
estudiantes, pauta) a nivel de establecimiento, normaliza variables
y construye scores compuestos por factor estructural.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

from src.config.settings import (
    DIMENSIONES_COMPUESTAS,
    INDICADORES_DOCENTES,
    INDICADORES_ESTUDIANTES,
    INDICADORES_PAUTA,
    FACTORES_ESTRUCTURALES,
    ETIQUETAS_ZONA,
    TABLES_DIR,
)


def integrar_datos_establecimiento(datos: dict) -> pd.DataFrame:
    """
    Integra los datos de multiples fuentes a nivel de establecimiento (ID).
    Agrega indicadores de docentes y estudiantes como promedios por
    establecimiento, y fusiona con datos de directores, coordinadores y pauta.

    Parametros
    ----------
    datos : dict
        Diccionario de DataFrames cargados.

    Retorna
    -------
    pd.DataFrame
        DataFrame integrado a nivel de establecimiento.
    """
    # Base: directores (urbanos y rurales) con dimensiones compuestas
    df_base = datos["directores"][
        ["ID", "COD_DEPE2", "ESTRATO_ANALITICO", "RURAL_RBD"]
        + [c for c in DIMENSIONES_COMPUESTAS if c in datos["directores"].columns]
    ].copy()

    # Eliminar duplicados por ID (tomar primer registro si hay duplicados)
    df_base = df_base.drop_duplicates(subset="ID", keep="first")
    print(f"  [BASE] Establecimientos base (directores): {df_base.shape[0]}")

    # Agregar indicadores de docentes a nivel de establecimiento
    if "docentes" in datos:
        cols_doc = ["ID"] + [c for c in INDICADORES_DOCENTES if c in datos["docentes"].columns]
        df_doc_agg = (
            datos["docentes"][cols_doc]
            .groupby("ID")
            .mean()
            .reset_index()
        )
        # Renombrar para evitar conflictos
        rename_doc = {c: f"DOC_{c}" for c in df_doc_agg.columns if c != "ID"}
        df_doc_agg = df_doc_agg.rename(columns=rename_doc)
        df_base = df_base.merge(df_doc_agg, on="ID", how="left")
        print(f"  [MERGE] Docentes agregados: {df_doc_agg.shape[0]} establecimientos")

    # Agregar indicadores de estudiantes a nivel de establecimiento
    if "estudiantes" in datos:
        cols_est = ["ID"] + [c for c in INDICADORES_ESTUDIANTES if c in datos["estudiantes"].columns]
        df_est_agg = (
            datos["estudiantes"][cols_est]
            .groupby("ID")
            .mean()
            .reset_index()
        )
        rename_est = {c: f"EST_{c}" for c in df_est_agg.columns if c != "ID"}
        df_est_agg = df_est_agg.rename(columns=rename_est)
        df_base = df_base.merge(df_est_agg, on="ID", how="left")
        print(f"  [MERGE] Estudiantes agregados: {df_est_agg.shape[0]} establecimientos")

    # Agregar indicadores de infraestructura (pauta)
    if "pauta" in datos:
        cols_pauta = ["ID"] + [c for c in INDICADORES_PAUTA if c in datos["pauta"].columns]
        df_pauta = datos["pauta"][cols_pauta].drop_duplicates(subset="ID", keep="first")
        df_base = df_base.merge(df_pauta, on="ID", how="left")
        print(f"  [MERGE] Infraestructura (pauta): {df_pauta.shape[0]} establecimientos")

    # Agregar etiqueta de zona
    df_base["zona"] = df_base["RURAL_RBD"].map(ETIQUETAS_ZONA).fillna("Desconocido")

    print(f"  [RESULTADO] Dataset integrado: {df_base.shape[0]} filas x {df_base.shape[1]} columnas")
    return df_base


def construir_scores_factores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza las dimensiones compuestas y construye scores promedio
    por factor estructural para cada establecimiento.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame integrado a nivel de establecimiento.

    Retorna
    -------
    pd.DataFrame
        DataFrame con scores normalizados por factor estructural,
        incluyendo region, comuna, zona y variables de estratificacion.
    """
    # Identificar columnas de dimensiones disponibles
    dims_disponibles = [c for c in DIMENSIONES_COMPUESTAS if c in df.columns]

    if not dims_disponibles:
        print("  [ERROR] No se encontraron dimensiones compuestas en el DataFrame.")
        return df

    # Normalizar dimensiones compuestas (escalamiento estandar)
    scaler = StandardScaler()
    df_norm = df.copy()

    # Solo normalizar filas con datos completos en las dimensiones
    mask_validos = df_norm[dims_disponibles].notna().all(axis=1)
    n_validos = mask_validos.sum()
    print(f"  [NORMALIZACION] Establecimientos con datos completos: {n_validos}/{df_norm.shape[0]}")

    if n_validos > 0:
        valores_norm = scaler.fit_transform(df_norm.loc[mask_validos, dims_disponibles])
        for i, col in enumerate(dims_disponibles):
            df_norm.loc[mask_validos, f"{col}_NORM"] = valores_norm[:, i]

    # Construir scores por factor estructural
    for factor, config in FACTORES_ESTRUCTURALES.items():
        dims_factor = [d for d in config["dimensiones"] if d in dims_disponibles]
        if dims_factor:
            cols_norm = [f"{d}_NORM" for d in dims_factor]
            cols_existentes = [c for c in cols_norm if c in df_norm.columns]
            if cols_existentes:
                df_norm[f"SCORE_{factor.upper()}"] = df_norm[cols_existentes].mean(axis=1)
                print(f"  [SCORE] {factor}: basado en {len(cols_existentes)} dimensiones")

    # Construir score global
    cols_scores = [c for c in df_norm.columns if c.startswith("SCORE_")]
    if cols_scores:
        df_norm["SCORE_GLOBAL"] = df_norm[cols_scores].mean(axis=1)
        print(f"  [SCORE] Global: promedio de {len(cols_scores)} factores")

    return df_norm


def guardar_scores(df: pd.DataFrame) -> str:
    """
    Guarda los scores por factor en outputs/tables/.

    Retorna
    -------
    str
        Ruta del archivo guardado.
    """
    cols_salida = (
        ["ID", "COD_DEPE2", "ESTRATO_ANALITICO", "RURAL_RBD", "zona"]
        + [c for c in df.columns if c.startswith("SCORE_")]
        + [c for c in DIMENSIONES_COMPUESTAS if c in df.columns]
    )
    cols_existentes = [c for c in cols_salida if c in df.columns]

    ruta = os.path.join(TABLES_DIR, "scores_factores_establecimiento.csv")
    df[cols_existentes].to_csv(ruta, index=False, encoding="utf-8-sig")
    print(f"  [GUARDADO] Scores por factor: {ruta}")
    return ruta


def resumen_scores_por_zona(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen estadistico de los scores por zona (urbano/rural).

    Retorna
    -------
    pd.DataFrame
        Resumen con media, mediana y desviacion estandar por zona.
    """
    cols_scores = [c for c in df.columns if c.startswith("SCORE_")]
    if not cols_scores:
        return pd.DataFrame()

    resumen = df.groupby("zona")[cols_scores].agg(["mean", "median", "std"])
    resumen.columns = ["_".join(col) for col in resumen.columns]
    return resumen


def resumen_scores_por_dependencia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen estadistico de los scores por tipo de dependencia.

    Retorna
    -------
    pd.DataFrame
        Resumen con media por dependencia administrativa.
    """
    cols_scores = [c for c in df.columns if c.startswith("SCORE_")]
    if not cols_scores:
        return pd.DataFrame()

    resumen = df.groupby("COD_DEPE2")[cols_scores].mean()
    return resumen
