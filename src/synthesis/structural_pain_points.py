# -*- coding: utf-8 -*-
"""
Modulo de sintesis de problematicas estructurales.
Traduce los resultados analiticos en un listado priorizado de
problematicas ("dolores") del sistema educativo chileno en materia
de digitalizacion escolar.
"""

import pandas as pd
import os
from datetime import datetime

from src.config.settings import (
    ETIQUETAS_FACTORES,
    TABLES_DIR,
    REPORTS_DIR,
)


def sintetizar_dolores(resultados: dict) -> list:
    """
    Genera un listado priorizado de problematicas estructurales
    del sistema educativo en materia de digitalizacion.

    Parametros
    ----------
    resultados : dict
        Diccionario con resultados de los modulos anteriores:
        - "brechas": pd.DataFrame
        - "cuellos_botella": pd.DataFrame
        - "perfiles_cluster": pd.DataFrame
        - "correlaciones": pd.DataFrame
        - "df_factores": pd.DataFrame

    Retorna
    -------
    list
        Lista de diccionarios con problematicas estructurales priorizadas.
    """
    dolores = []

    # 1. Dolores derivados de brechas territoriales
    if "brechas" in resultados and not resultados["brechas"].empty:
        dolores.extend(_dolores_por_brechas(resultados["brechas"]))

    # 2. Dolores derivados de cuellos de botella
    if "cuellos_botella" in resultados and not resultados["cuellos_botella"].empty:
        dolores.extend(_dolores_por_cuellos_botella(resultados["cuellos_botella"]))

    # 3. Dolores derivados de la segmentacion
    if "perfiles_cluster" in resultados and not resultados["perfiles_cluster"].empty:
        dolores.extend(_dolores_por_segmentacion(resultados["perfiles_cluster"]))

    # 4. Dolores derivados de desalineaciones internas
    if "df_factores" in resultados:
        dolores.extend(_dolores_por_desalineacion(resultados["df_factores"]))

    # Priorizar dolores
    dolores = _priorizar_dolores(dolores)

    print(f"\n[SINTESIS] {len(dolores)} problematicas estructurales identificadas")
    for i, dolor in enumerate(dolores, 1):
        print(f"  {i}. [{dolor['prioridad']}] {dolor['titulo']}")

    return dolores


def _dolores_por_brechas(df_brechas: pd.DataFrame) -> list:
    """Genera dolores a partir del analisis de brechas."""
    dolores = []

    # Brechas significativas por zona
    brechas_zona = df_brechas[
        (df_brechas["tipo_brecha"].str.contains("zona", case=False)) &
        (df_brechas["es_significativa"] == True)
    ]

    if not brechas_zona.empty:
        factores_afectados = brechas_zona["factor"].unique().tolist()
        magnitud_max = brechas_zona["magnitud_brecha"].max()

        dolores.append({
            "titulo": "Brecha urbano-rural en digitalizacion escolar",
            "tipo": "Brecha territorial",
            "descripcion": (
                f"Se identifican diferencias significativas entre establecimientos "
                f"urbanos y rurales en {len(factores_afectados)} factores estructurales. "
                f"La magnitud maxima alcanza {magnitud_max:.2f} desviaciones estandar. "
                f"Factores afectados: {', '.join(factores_afectados)}."
            ),
            "evidencia": f"{len(brechas_zona)} brechas significativas detectadas",
            "factores": factores_afectados,
            "magnitud": magnitud_max,
            "nivel_analisis": "Territorial",
        })

    # Brechas por dependencia administrativa
    brechas_dep = df_brechas[
        (df_brechas["tipo_brecha"].str.contains("COD_DEPE2", case=False)) &
        (df_brechas["es_significativa"] == True)
    ]

    if not brechas_dep.empty:
        factores_dep = brechas_dep["factor"].unique().tolist()
        dolores.append({
            "titulo": "Inequidades segun tipo de administracion educativa",
            "tipo": "Brecha institucional",
            "descripcion": (
                f"La dependencia administrativa del establecimiento se asocia a "
                f"diferencias en {len(factores_dep)} factores de digitalizacion. "
                f"Esto sugiere que las condiciones institucionales mediatizan "
                f"el acceso a la innovacion digital de forma diferenciada."
            ),
            "evidencia": f"{len(brechas_dep)} brechas significativas por dependencia",
            "factores": factores_dep,
            "magnitud": brechas_dep["magnitud_brecha"].max(),
            "nivel_analisis": "Institucional",
        })

    # Desalineacion interna
    brechas_internas = df_brechas[
        df_brechas["tipo_brecha"].str.contains("interna", case=False)
    ]
    if not brechas_internas.empty:
        for _, fila in brechas_internas.iterrows():
            dolores.append({
                "titulo": "Desalineacion entre capacidades y apropiacion",
                "tipo": "Desalineacion interna",
                "descripcion": (
                    f"Se observa una desalineacion sistematica entre {fila['grupo_1']} "
                    f"y {fila['grupo_2']}. La diferencia media entre factores "
                    f"({fila['diferencia']:.2f}) sugiere que la digitalizacion no "
                    f"avanza de forma homogenea en todas sus dimensiones."
                ),
                "evidencia": f"Rango interno medio: {fila['magnitud_brecha']:.2f}",
                "factores": [fila["grupo_1"], fila["grupo_2"]],
                "magnitud": fila["magnitud_brecha"],
                "nivel_analisis": "Sistema educativo",
            })

    return dolores


def _dolores_por_cuellos_botella(df_cuellos: pd.DataFrame) -> list:
    """Genera dolores a partir de los cuellos de botella identificados."""
    dolores = []

    # Factores con severidad alta (positiva) son cuellos de botella
    cuellos_severos = df_cuellos[df_cuellos["severidad_cuello_botella"] > 0]

    if not cuellos_severos.empty:
        for _, fila in cuellos_severos.iterrows():
            dolores.append({
                "titulo": f"Cuello de botella en {fila['factor']}",
                "tipo": "Cuello de botella",
                "descripcion": (
                    f"El factor '{fila['factor']}' presenta un nivel bajo "
                    f"(media={fila['media']:.2f}) y alta conectividad con otros "
                    f"factores ({fila['n_relaciones_fuertes']} relaciones fuertes). "
                    f"Esto implica que su debilidad arrastra al resto del sistema. "
                    f"Factores afectados: {fila['factores_relacionados']}."
                ),
                "evidencia": f"Severidad: {fila['severidad_cuello_botella']:.4f}",
                "factores": [fila["factor"]],
                "magnitud": abs(fila["severidad_cuello_botella"]),
                "nivel_analisis": "Sistema educativo",
            })

    return dolores


def _dolores_por_segmentacion(perfiles: pd.DataFrame) -> list:
    """Genera dolores a partir de la segmentacion de establecimientos."""
    dolores = []

    if "n_establecimientos" in perfiles.columns:
        total = perfiles["n_establecimientos"].sum()

        # Identificar clusters rezagados
        cols_scores = [c for c in perfiles.columns if c.startswith("SCORE_")]
        if cols_scores:
            score_medio = perfiles[cols_scores].mean(axis=1)
            clusters_bajos = score_medio[score_medio < -0.3].index.tolist()

            if clusters_bajos:
                n_rezagados = perfiles.loc[clusters_bajos, "n_establecimientos"].sum()
                pct_rezagados = (n_rezagados / total) * 100

                dolores.append({
                    "titulo": "Segmento significativo de establecimientos rezagados",
                    "tipo": "Segmentacion",
                    "descripcion": (
                        f"Un {pct_rezagados:.1f}% de los establecimientos "
                        f"({n_rezagados} de {total}) se ubica en tipologias con "
                        f"scores sistematicamente bajos en todas las dimensiones "
                        f"de digitalizacion. Estos establecimientos requieren "
                        f"intervencion integral, no solo de infraestructura."
                    ),
                    "evidencia": f"{n_rezagados} establecimientos en clusters rezagados",
                    "factores": cols_scores,
                    "magnitud": pct_rezagados / 100,
                    "nivel_analisis": "Tipologias de establecimientos",
                })

    return dolores


def _dolores_por_desalineacion(df_factores: pd.DataFrame) -> list:
    """Genera dolores basados en la desalineacion entre factores."""
    dolores = []
    cols_scores = [c for c in df_factores.columns if c.startswith("SCORE_") and c != "SCORE_GLOBAL"]

    if len(cols_scores) < 2:
        return dolores

    # Calcular coeficiente de variacion entre factores por establecimiento
    rangos = df_factores[cols_scores].max(axis=1) - df_factores[cols_scores].min(axis=1)
    rangos = rangos.dropna()

    # Establecimientos con alta desalineacion (rango > percentil 75)
    umbral_alto = rangos.quantile(0.75)
    n_desalineados = (rangos > umbral_alto).sum()
    pct_desalineados = (n_desalineados / len(rangos)) * 100

    dolores.append({
        "titulo": "Desarrollo desigual de capacidades digitales dentro de establecimientos",
        "tipo": "Desalineacion interna",
        "descripcion": (
            f"El 25% de los establecimientos con mayor desalineacion interna "
            f"presenta un rango entre factores superior a {umbral_alto:.2f}. "
            f"Esto indica que el desarrollo digital es fragmentario: "
            f"algunos factores avanzan mientras otros se estancan, generando "
            f"una digitalizacion incompleta y potencialmente ineficaz."
        ),
        "evidencia": f"Rango medio: {rangos.mean():.2f}, Q75: {umbral_alto:.2f}",
        "factores": cols_scores,
        "magnitud": rangos.mean(),
        "nivel_analisis": "Establecimientos",
    })

    return dolores


def _priorizar_dolores(dolores: list) -> list:
    """
    Prioriza los dolores segun magnitud, tipo y alcance.
    Asigna una etiqueta de prioridad (CRITICA, ALTA, MEDIA, BAJA).
    """
    for dolor in dolores:
        magnitud = dolor.get("magnitud", 0)
        if magnitud >= 0.8:
            dolor["prioridad"] = "CRITICA"
        elif magnitud >= 0.5:
            dolor["prioridad"] = "ALTA"
        elif magnitud >= 0.3:
            dolor["prioridad"] = "MEDIA"
        else:
            dolor["prioridad"] = "BAJA"

    # Ordenar por prioridad
    orden_prioridad = {"CRITICA": 0, "ALTA": 1, "MEDIA": 2, "BAJA": 3}
    dolores.sort(key=lambda d: (orden_prioridad.get(d["prioridad"], 4), -d.get("magnitud", 0)))

    return dolores


def generar_reporte_sintesis(dolores: list) -> str:
    """
    Genera un reporte textual de sintesis de las problematicas
    estructurales identificadas.

    Retorna
    -------
    str
        Ruta del reporte generado.
    """
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")

    lineas = [
        "=" * 80,
        "REPORTE DE SINTESIS: PROBLEMATICAS ESTRUCTURALES",
        "DE LA DIGITALIZACION ESCOLAR EN CHILE (ENDDEIE 2023)",
        "=" * 80,
        f"Fecha de generacion: {fecha}",
        f"Total de problematicas identificadas: {len(dolores)}",
        "",
        "-" * 80,
        "",
    ]

    # Agrupar por prioridad
    for prioridad in ["CRITICA", "ALTA", "MEDIA", "BAJA"]:
        dolores_p = [d for d in dolores if d["prioridad"] == prioridad]
        if not dolores_p:
            continue

        lineas.append(f"### PRIORIDAD {prioridad} ({len(dolores_p)} problematicas)")
        lineas.append("")

        for i, dolor in enumerate(dolores_p, 1):
            lineas.append(f"  {i}. {dolor['titulo']}")
            lineas.append(f"     Tipo: {dolor['tipo']}")
            lineas.append(f"     Nivel de analisis: {dolor['nivel_analisis']}")
            lineas.append(f"     Descripcion: {dolor['descripcion']}")
            lineas.append(f"     Evidencia: {dolor['evidencia']}")
            lineas.append("")

        lineas.append("-" * 80)
        lineas.append("")

    # Conclusion general
    lineas.extend([
        "CONCLUSION GENERAL",
        "=" * 80,
        "",
        "Las principales problematicas de la digitalizacion educativa en Chile",
        "no responden a deficits aislados, sino a desalineaciones estructurales",
        "entre capacidades, gestion y apropiacion pedagogica, que afectan de forma",
        "diferenciada a distintos tipos de establecimientos y territorios.",
        "",
        "Los hallazgos sugieren que:",
        "",
        "  1. La brecha urbano-rural trasciende la infraestructura y se extiende",
        "     a la gestion, la cultura de innovacion y las capacidades digitales.",
        "",
        "  2. Existen cuellos de botella estructurales que impiden que mejoras",
        "     parciales se traduzcan en avances sostenidos del sistema.",
        "",
        "  3. La segmentacion del sistema revela tipologias de establecimientos",
        "     con necesidades diferenciadas que requieren politicas focalizadas.",
        "",
        "  4. La desalineacion interna entre factores indica que el desarrollo",
        "     digital es fragmentario, lo que limita su impacto en el aprendizaje.",
        "",
        "=" * 80,
    ])

    texto = "\n".join(lineas)

    # Guardar reporte
    ruta = os.path.join(REPORTS_DIR, "reporte_sintesis_problematicas.txt")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(texto)

    print(f"  [GUARDADO] Reporte de sintesis: {ruta}")
    return ruta


def guardar_dolores_csv(dolores: list) -> str:
    """Guarda la lista de dolores como CSV."""
    df_dolores = pd.DataFrame(dolores)

    # Convertir listas a strings para CSV
    if "factores" in df_dolores.columns:
        df_dolores["factores"] = df_dolores["factores"].apply(
            lambda x: "; ".join(x) if isinstance(x, list) else str(x)
        )

    ruta = os.path.join(TABLES_DIR, "dolores_estructurales_priorizados.csv")
    df_dolores.to_csv(ruta, index=False, encoding="utf-8-sig")
    print(f"  [GUARDADO] Dolores priorizados: {ruta}")
    return ruta
