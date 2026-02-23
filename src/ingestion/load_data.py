# -*- coding: utf-8 -*-
"""
Modulo de carga y validacion de datos ENDDEIE 2023.
Responsable de leer los archivos CSV, validar su estructura basica
y entregar un diccionario de DataFrames listo para el analisis.
"""

import os
import pandas as pd
from src.config.settings import DATA_DIR, ARCHIVOS_CSV, COLS_ID


def cargar_datos_base(path_data: str = None) -> dict:
    """
    Carga los archivos CSV de la ENDDEIE 2023 y devuelve un diccionario
    de DataFrames indexado por nombre del dataset.

    Parametros
    ----------
    path_data : str, opcional
        Ruta al directorio de datos. Si no se especifica, usa DATA_DIR.

    Retorna
    -------
    dict
        Diccionario {nombre_dataset: pd.DataFrame}.
    """
    if path_data is None:
        path_data = DATA_DIR

    datos = {}
    for nombre, archivo in ARCHIVOS_CSV.items():
        ruta = os.path.join(path_data, archivo)
        if not os.path.exists(ruta):
            print(f"  [ADVERTENCIA] Archivo no encontrado: {archivo}")
            continue

        try:
            df = pd.read_csv(
                ruta,
                sep=";",
                decimal=",",
                encoding="utf-8",
                low_memory=False,
            )
            datos[nombre] = df
            print(f"  [OK] {nombre}: {df.shape[0]} filas x {df.shape[1]} columnas")
        except Exception as e:
            print(f"  [ERROR] No se pudo cargar {archivo}: {e}")

    return datos


def validar_estructura(datos: dict) -> pd.DataFrame:
    """
    Valida la estructura basica de los DataFrames cargados.
    Verifica presencia de columnas de identificacion, valores nulos
    y tipos de datos.

    Parametros
    ----------
    datos : dict
        Diccionario de DataFrames.

    Retorna
    -------
    pd.DataFrame
        Resumen de validacion por dataset.
    """
    resumen = []

    for nombre, df in datos.items():
        # Verificar columnas de identificacion
        cols_presentes = [c for c in COLS_ID if c in df.columns]
        cols_faltantes = [c for c in COLS_ID if c not in df.columns]

        # Porcentaje de nulos general
        pct_nulos = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100

        # Contar columnas numericas vs texto
        cols_num = df.select_dtypes(include=["number"]).shape[1]
        cols_obj = df.select_dtypes(include=["object"]).shape[1]

        # IDs unicos
        n_ids_unicos = df["ID"].nunique() if "ID" in df.columns else 0

        resumen.append({
            "dataset": nombre,
            "filas": df.shape[0],
            "columnas": df.shape[1],
            "ids_unicos": n_ids_unicos,
            "cols_id_presentes": len(cols_presentes),
            "cols_id_faltantes": ", ".join(cols_faltantes) if cols_faltantes else "ninguna",
            "cols_numericas": cols_num,
            "cols_texto": cols_obj,
            "pct_nulos": round(pct_nulos, 2),
        })

    df_resumen = pd.DataFrame(resumen)
    print("\n[VALIDACION] Resumen de estructura de datos:")
    print(df_resumen.to_string(index=False))
    return df_resumen


def obtener_indicadores_por_dataset(datos: dict) -> dict:
    """
    Identifica automaticamente las columnas de indicadores (prefijo IND_)
    y dimensiones compuestas en cada dataset.

    Parametros
    ----------
    datos : dict
        Diccionario de DataFrames.

    Retorna
    -------
    dict
        Diccionario {nombre_dataset: {"indicadores": [...], "dimensiones": [...]}}.
    """
    resultado = {}
    for nombre, df in datos.items():
        indicadores = [c for c in df.columns if c.startswith("IND_")]
        dimensiones = [c for c in df.columns if c.isupper() and "_" in c
                       and not c.startswith("IND_")
                       and not c.startswith("Q")
                       and c not in COLS_ID
                       and c not in ["WGT_EST_FINAL", "WGT_DOC_FINAL",
                                     "TASA_EST_PC1", "TASA_EST_PC2",
                                     "TASA_SALAS_LABORATORIOS",
                                     "TASA_DEPENDENCIAS_INTERNET",
                                     "TASA_SALAS_INTERNET"]]

        resultado[nombre] = {
            "indicadores": indicadores,
            "dimensiones": dimensiones,
        }
        print(f"  [INFO] {nombre}: {len(indicadores)} indicadores, "
              f"{len(dimensiones)} dimensiones compuestas")

    return resultado
