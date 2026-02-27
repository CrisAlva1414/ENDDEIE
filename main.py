# -*- coding: utf-8 -*-
"""
=============================================================================
ENDDEIE 2023 - Analisis Estructural de la Digitalizacion Escolar
=============================================================================

Orquestador principal del pipeline de analisis.
Ejecuta secuencialmente cada modulo del proyecto, desde la ingestion de datos
hasta la sintesis de problematicas estructurales.

Autor: Agente de Analisis Estructural
Datos: Encuesta Nacional de Digitalizacion Escolar (ENDDEIE 2023)
Objetivo: Identificar desalineaciones estructurales del sistema educativo
          chileno en materia de digitalizacion.
"""

import os
import sys

# Asegurar que el directorio raiz del proyecto este en el path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import matplotlib
matplotlib.use("Agg")  # Backend no interactivo para servidores

# =============================================================================
# IMPORTACIONES DE MODULOS DEL PROYECTO
# =============================================================================
from src.config.settings import crear_directorios, TABLES_DIR, REPORTS_DIR
from src.ingestion.load_data import (
    cargar_datos_base,
    validar_estructura,
    obtener_indicadores_por_dataset,
)
from src.indicators.map_indicators import (
    mapear_dimensiones,
    guardar_mapa_indicadores,
)
from src.factors.build_factors import (
    integrar_datos_establecimiento,
    construir_scores_factores,
    guardar_scores,
    resumen_scores_por_zona,
)
from src.gaps.structural_gaps import (
    detectar_brechas,
    generar_graficos_brechas,
    guardar_brechas,
)
from src.clustering.segment_schools import (
    clusterizar_escuelas,
    generar_perfiles_cluster,
    generar_graficos_clustering,
    guardar_clustering,
)
from src.correlations.bottlenecks import (
    analizar_correlaciones,
    identificar_cuellos_botella,
    analizar_correlaciones_por_zona,
    generar_graficos_correlaciones,
    guardar_correlaciones,
)
from src.synthesis.structural_pain_points import (
    sintetizar_dolores,
    generar_reporte_sintesis,
    guardar_dolores_csv,
)

# --- MODULOS ML (Etapa de Identificacion de Oportunidades de Software) ---
from src.ml.dimensionality.latent_axes import ejecutar_dimensionalidad
from src.ml.clustering.software_needs_profiles import ejecutar_perfiles_software
from src.ml.explainability.drivers_and_barriers import ejecutar_explicabilidad
from src.ml.evaluation.stability_checks import ejecutar_evaluacion_estabilidad


def ejecutar_pipeline():
    """
    Ejecuta el pipeline completo de analisis estructural
    de la ENDDEIE 2023.
    """
    print("=" * 80)
    print("ENDDEIE 2023 - ANALISIS ESTRUCTURAL DE LA DIGITALIZACION ESCOLAR")
    print("=" * 80)

    # -----------------------------------------------------------------
    # PASO 0: Crear directorios de salida
    # -----------------------------------------------------------------
    print("\n[PASO 0] Creando directorios de salida...")
    crear_directorios()

    # -----------------------------------------------------------------
    # PASO 1: Ingestion y validacion de datos
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PASO 1] INGESTION Y VALIDACION DE DATOS")
    print("=" * 80)

    print("\nCargando datos base...")
    datos = cargar_datos_base()

    if not datos:
        print("[ERROR CRITICO] No se cargaron datos. Abortando pipeline.")
        return

    print("\nValidando estructura...")
    df_validacion = validar_estructura(datos)
    df_validacion.to_csv(
        os.path.join(TABLES_DIR, "validacion_estructura_datos.csv"),
        index=False, encoding="utf-8-sig"
    )

    print("\nIdentificando indicadores por dataset...")
    indicadores_info = obtener_indicadores_por_dataset(datos)

    # -----------------------------------------------------------------
    # PASO 2: Mapeo de indicadores y dimensiones
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PASO 2] MAPEO DE INDICADORES Y DIMENSIONES ESTRUCTURALES")
    print("=" * 80)

    print("\nMapeando dimensiones estructurales...")
    df_mapa = mapear_dimensiones()
    guardar_mapa_indicadores(df_mapa)

    n_por_actor = df_mapa.groupby("actor").size()
    print(f"\nIndicadores por actor:")
    for actor, n in n_por_actor.items():
        print(f"  - {actor}: {n} indicadores")

    n_por_factor = df_mapa.groupby("factor_estructural").size()
    print(f"\nIndicadores por factor estructural:")
    for factor, n in n_por_factor.items():
        print(f"  - {factor}: {n} indicadores")

    # -----------------------------------------------------------------
    # PASO 3: Construccion de scores por factor
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PASO 3] CONSTRUCCION DE SCORES ESTRUCTURALES")
    print("=" * 80)

    print("\nIntegrando datos a nivel de establecimiento...")
    df_integrado = integrar_datos_establecimiento(datos)

    print("\nConstruyendo scores por factor estructural...")
    df_factores = construir_scores_factores(df_integrado)
    guardar_scores(df_factores)

    print("\nResumen de scores por zona:")
    resumen_zona = resumen_scores_por_zona(df_factores)
    if not resumen_zona.empty:
        print(resumen_zona.to_string())
        resumen_zona.to_csv(
            os.path.join(TABLES_DIR, "resumen_scores_por_zona.csv"),
            encoding="utf-8-sig"
        )

    # -----------------------------------------------------------------
    # PASO 4: Deteccion de brechas estructurales
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PASO 4] DETECCION DE BRECHAS ESTRUCTURALES")
    print("=" * 80)

    print("\nDetectando brechas...")
    df_brechas = detectar_brechas(df_factores)

    if not df_brechas.empty:
        guardar_brechas(df_brechas)

        brechas_sig = df_brechas[df_brechas["es_significativa"] == True]
        print(f"\nBrechas significativas: {len(brechas_sig)} de {len(df_brechas)}")

        print("\nGenerando graficos de brechas...")
        rutas_graficos_brechas = generar_graficos_brechas(df_factores, df_brechas)
        print(f"  {len(rutas_graficos_brechas)} graficos generados")
    else:
        print("  No se detectaron brechas.")

    # -----------------------------------------------------------------
    # PASO 5: Segmentacion de establecimientos (clustering)
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PASO 5] SEGMENTACION DE ESTABLECIMIENTOS")
    print("=" * 80)

    print("\nClusterizando establecimientos...")
    df_clustered = clusterizar_escuelas(df_factores)

    print("\nGenerando perfiles de clusters...")
    perfiles = generar_perfiles_cluster(df_clustered)

    if not perfiles.empty:
        guardar_clustering(df_clustered, perfiles)

        print("\nDistribucion por tipologia:")
        if "tipologia" in df_clustered.columns:
            dist_tip = df_clustered["tipologia"].value_counts()
            for tip, n in dist_tip.items():
                pct = (n / len(df_clustered)) * 100
                print(f"  - {tip}: {n} ({pct:.1f}%)")

        print("\nGenerando graficos de clustering...")
        rutas_graficos_cluster = generar_graficos_clustering(df_clustered)
        print(f"  {len(rutas_graficos_cluster)} graficos generados")

    # -----------------------------------------------------------------
    # PASO 6: Analisis de correlaciones y cuellos de botella
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PASO 6] CORRELACIONES Y CUELLOS DE BOTELLA")
    print("=" * 80)

    print("\nCalculando correlaciones Spearman...")
    corr_matrix = analizar_correlaciones(df_factores)

    if not corr_matrix.empty:
        print("\nIdentificando cuellos de botella...")
        df_cuellos = identificar_cuellos_botella(corr_matrix, df_factores)

        print("\nAnalizando correlaciones por zona...")
        corr_por_zona = analizar_correlaciones_por_zona(df_factores)

        guardar_correlaciones(corr_matrix, df_cuellos)

        print("\nGenerando graficos de correlaciones...")
        rutas_graficos_corr = generar_graficos_correlaciones(
            corr_matrix, df_cuellos, corr_por_zona
        )
        print(f"  {len(rutas_graficos_corr)} graficos generados")

        # Mostrar cuellos de botella principales
        cuellos_criticos = df_cuellos[df_cuellos["severidad_cuello_botella"] > 0]
        if not cuellos_criticos.empty:
            print("\nCuellos de botella identificados:")
            for _, fila in cuellos_criticos.iterrows():
                print(f"  - {fila['factor']}: severidad={fila['severidad_cuello_botella']:.4f}, "
                      f"media={fila['media']:.2f}")
    else:
        df_cuellos = pd.DataFrame()

    # -----------------------------------------------------------------
    # PASO 7: Sintesis de problematicas estructurales
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PASO 7] SINTESIS DE PROBLEMATICAS ESTRUCTURALES")
    print("=" * 80)

    resultados_sintesis = {
        "brechas": df_brechas,
        "cuellos_botella": df_cuellos,
        "perfiles_cluster": perfiles,
        "correlaciones": corr_matrix,
        "df_factores": df_factores,
    }

    print("\nSintetizando problematicas...")
    dolores = sintetizar_dolores(resultados_sintesis)

    if dolores:
        guardar_dolores_csv(dolores)
        generar_reporte_sintesis(dolores)

    # =================================================================
    # ETAPA ML — IDENTIFICACION DE OPORTUNIDADES DE SOFTWARE EDUCATIVO
    # =================================================================
    print("\n" + "=" * 80)
    print("ETAPA ML — IDENTIFICACION DE OPORTUNIDADES DE SOFTWARE EDUCATIVO")
    print("=" * 80)

    # -----------------------------------------------------------------
    # PASO 8: Reduccion dimensional — Ejes latentes
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[PASO 8] REDUCCION DIMENSIONAL: EJES LATENTES DE NECESIDAD DIGITAL")
    print("-" * 80)

    res_dimensionalidad = ejecutar_dimensionalidad()

    # -----------------------------------------------------------------
    # PASO 9: Perfiles de necesidad de software educativo
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[PASO 9] PERFILES DE NECESIDAD DE SOFTWARE EDUCATIVO")
    print("-" * 80)

    res_perfiles = ejecutar_perfiles_software()

    # -----------------------------------------------------------------
    # PASO 10: Explicabilidad — Drivers y barreras
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[PASO 10] EXPLICABILIDAD: DRIVERS Y BARRERAS PARA LA ADOPCION DIGITAL")
    print("-" * 80)

    res_explicabilidad = ejecutar_explicabilidad()

    # -----------------------------------------------------------------
    # PASO 11: Evaluacion de estabilidad
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[PASO 11] EVALUACION DE ESTABILIDAD")
    print("-" * 80)

    res_estabilidad = ejecutar_evaluacion_estabilidad()

    # -----------------------------------------------------------------
    # RESUMEN FINAL
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 80)

    # Contar archivos generados
    n_tablas = len([f for f in os.listdir(TABLES_DIR) if f.endswith(".csv")])
    from src.config.settings import FIGURES_DIR
    n_figuras = len([f for f in os.listdir(FIGURES_DIR) if f.endswith(".png")])
    n_reportes = len([f for f in os.listdir(REPORTS_DIR) if os.path.isfile(os.path.join(REPORTS_DIR, f))])

    print(f"\nOutputs generados:")
    print(f"  - Tablas CSV:   {n_tablas} archivos en outputs/tables/")
    print(f"  - Figuras PNG:  {n_figuras} archivos en outputs/figures/")
    print(f"  - Reportes:     {n_reportes} archivos en outputs/reports/")
    print(f"\nProblematicas estructurales identificadas: {len(dolores)}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print(
        "\nLas principales problematicas de la digitalizacion educativa en Chile\n"
        "no responden a deficits aislados, sino a desalineaciones estructurales\n"
        "entre capacidades, gestion y apropiacion pedagogica, que afectan de forma\n"
        "diferenciada a distintos tipos de establecimientos y territorios."
    )
    print("=" * 80)


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
if __name__ == "__main__":
    # Importar pandas aqui para evitar imports circulares en el scope de modulos
    import pandas as pd
    ejecutar_pipeline()
