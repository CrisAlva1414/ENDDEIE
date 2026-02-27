# -*- coding: utf-8 -*-
"""
=============================================================================
Modulo de Explicabilidad — Drivers y Barreras para la Adopcion Digital
=============================================================================

Usa modelos explicativos (arboles de decision, Random Forest) para identificar
que variables de la ENDDEIE actuan como drivers o barreras para la
adopcion digital y la pertenencia a perfiles de necesidad de software.

Entradas:
    - outputs/tables/scores_factores_establecimiento.csv
    - outputs/tables/ml_asignacion_perfiles_software.csv
Salidas:
    - outputs/tables/ml_importancia_variables_rf.csv
    - outputs/tables/ml_drivers_barreras_adopcion.csv
    - outputs/tables/ml_reglas_arbol_decision.csv
    - outputs/figures/ml_importancia_variables_rf.png
    - outputs/figures/ml_drivers_vs_barreras.png
    - outputs/figures/ml_arbol_decision_perfiles.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
import os

from src.config.settings import (
    TABLES_DIR,
    FIGURES_DIR,
    DPI,
    RANDOM_STATE,
    DIMENSIONES_COMPUESTAS,
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
    "RURAL_RBD": "Zona (Rural=1)",
    "COD_DEPE2": "Dependencia Adm.",
    "DESALIN_RANGO": "Desalineacion Interna",
}


# =========================================================================
# FUNCIONES PRINCIPALES
# =========================================================================

def cargar_datos_explicabilidad() -> tuple:
    """
    Carga scores de factores y asignacion de perfiles de software.

    Retorna
    -------
    tuple
        (df_scores, df_perfiles): DataFrames de scores y asignacion de perfiles.
    """
    ruta_scores = os.path.join(TABLES_DIR, "scores_factores_establecimiento.csv")
    df_scores = pd.read_csv(ruta_scores)

    ruta_perfiles = os.path.join(TABLES_DIR, "ml_asignacion_perfiles_software.csv")
    df_perfiles = pd.read_csv(ruta_perfiles)

    print(f"  [CARGA] Scores: {df_scores.shape[0]} estab. | Perfiles: {df_perfiles.shape[0]} asignados")
    return df_scores, df_perfiles


def construir_dataset_explicativo(df_scores: pd.DataFrame,
                                   df_perfiles: pd.DataFrame) -> tuple:
    """
    Construye el dataset para modelos explicativos, combinando
    dimensiones compuestas, scores, variables de contexto y la
    variable objetivo (perfil de necesidad de software).

    Parametros
    ----------
    df_scores : pd.DataFrame
        Scores por establecimiento.
    df_perfiles : pd.DataFrame
        Asignacion de perfiles de software.

    Retorna
    -------
    tuple
        (X, y, feature_names, df_merged): features, target, nombres, datos combinados.
    """
    # Merge
    df = df_scores.merge(df_perfiles[["ID", "perfil_cluster", "nombre_perfil"]], on="ID", how="inner")

    # Features: dimensiones compuestas + variables de contexto
    cols_dims = [c for c in DIMENSIONES_COMPUESTAS if c in df.columns]
    cols_contexto = []
    if "RURAL_RBD" in df.columns:
        cols_contexto.append("RURAL_RBD")
    if "COD_DEPE2" in df.columns:
        cols_contexto.append("COD_DEPE2")

    # Agregar desalineacion interna como feature
    cols_score_disp = [c for c in COLS_SCORES if c in df.columns]
    if len(cols_score_disp) >= 2:
        df["DESALIN_RANGO"] = df[cols_score_disp].max(axis=1) - df[cols_score_disp].min(axis=1)
        cols_dims.append("DESALIN_RANGO")

    feature_cols = cols_dims + cols_contexto
    mask = df[feature_cols + ["perfil_cluster"]].notna().all(axis=1)
    X = df.loc[mask, feature_cols].values
    y = df.loc[mask, "perfil_cluster"].values

    print(f"  [DATASET] {X.shape[0]} observaciones, {X.shape[1]} features")
    print(f"    Dimensiones compuestas: {len(cols_dims)}")
    print(f"    Variables de contexto: {len(cols_contexto)}")
    print(f"    Clases de perfil: {len(np.unique(y))}")

    return X, y, feature_cols, df.loc[mask]


def entrenar_random_forest(X: np.ndarray, y: np.ndarray,
                            feature_names: list) -> dict:
    """
    Entrena Random Forest para obtener importancia de variables.
    El objetivo NO es predecir, sino EXPLICAR que variables discriminan
    entre perfiles de necesidad de software.

    Parametros
    ----------
    X : np.ndarray
        Matriz de features.
    y : np.ndarray
        Vector de etiquetas de perfil.
    feature_names : list
        Nombres de features.

    Retorna
    -------
    dict
        Diccionario con modelo, importancias y metricas.
    """
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X, y)

    # Cross-validation (solo informativo, no es el objetivo)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")

    # Importancias
    importancias = pd.DataFrame({
        "variable": feature_names,
        "etiqueta": [ETIQUETAS_LEGIBLES.get(f, f) for f in feature_names],
        "importancia_gini": rf.feature_importances_,
    }).sort_values("importancia_gini", ascending=False)

    importancias["ranking"] = range(1, len(importancias) + 1)
    importancias["pct_importancia"] = (importancias["importancia_gini"] /
                                        importancias["importancia_gini"].sum() * 100).round(2)

    print(f"\n  [RANDOM FOREST] Accuracy CV (5-fold): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"  Top 5 variables mas discriminantes:")
    for _, row in importancias.head(5).iterrows():
        print(f"    {row['ranking']}. {row['etiqueta']}: {row['pct_importancia']:.1f}%")

    return {
        "modelo": rf,
        "importancias": importancias,
        "cv_accuracy": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    }


def entrenar_arbol_decision(X: np.ndarray, y: np.ndarray,
                              feature_names: list,
                              nombres_perfiles: dict = None) -> dict:
    """
    Entrena arbol de decision interpretable para extraer reglas explicitas
    de clasificacion de perfiles.

    Parametros
    ----------
    X : np.ndarray
        Matriz de features.
    y : np.ndarray
        Vector de etiquetas.
    feature_names : list
        Nombres de features.
    nombres_perfiles : dict
        Mapeo cluster_id -> nombre_perfil.

    Retorna
    -------
    dict
        Diccionario con modelo, reglas textuales y metricas.
    """
    dt = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=30,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    dt.fit(X, y)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(dt, X, y, cv=cv, scoring="accuracy")

    # Extraer reglas como texto
    nombres_legibles = [ETIQUETAS_LEGIBLES.get(f, f) for f in feature_names]
    class_names = [str(c) for c in sorted(np.unique(y))]
    if nombres_perfiles:
        class_names = [nombres_perfiles.get(int(c), c) for c in class_names]

    reglas_texto = export_text(
        dt,
        feature_names=nombres_legibles,
        class_names=class_names,
        max_depth=4,
    )

    print(f"\n  [ARBOL DECISION] Accuracy CV (5-fold): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"  Profundidad: {dt.get_depth()}, Hojas: {dt.get_n_leaves()}")

    return {
        "modelo": dt,
        "reglas_texto": reglas_texto,
        "feature_names_legibles": nombres_legibles,
        "class_names": class_names,
        "cv_accuracy": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    }


def clasificar_drivers_barreras(importancias: pd.DataFrame,
                                  df_merged: pd.DataFrame,
                                  feature_names: list,
                                  y: np.ndarray) -> pd.DataFrame:
    """
    Clasifica cada variable como driver o barrera segun su relacion
    con los perfiles de alta vs baja madurez digital.

    Parametros
    ----------
    importancias : pd.DataFrame
        Tabla de importancias del RF.
    df_merged : pd.DataFrame
        Dataset con features y perfil asignado.
    feature_names : list
        Nombres de features.
    y : np.ndarray
        Etiquetas de perfil.

    Retorna
    -------
    pd.DataFrame
        Tabla con cada variable clasificada como driver o barrera.
    """
    cols_scores = [c for c in COLS_SCORES if c in df_merged.columns]

    # Identificar cluster mas "maduro" (mayor score global promedio)
    df_work = df_merged.copy()
    df_work["_perfil"] = y
    if "SCORE_GLOBAL" in df_work.columns:
        media_global = df_work.groupby("_perfil")["SCORE_GLOBAL"].mean()
        perfil_maduro = media_global.idxmax()
        perfil_rezagado = media_global.idxmin()
    else:
        media_scores = df_work.groupby("_perfil")[cols_scores].mean().mean(axis=1)
        perfil_maduro = media_scores.idxmax()
        perfil_rezagado = media_scores.idxmin()

    # Para cada feature, comparar media en perfil maduro vs rezagado
    resultado = []
    for feat in feature_names:
        if feat not in df_work.columns:
            continue
        media_maduro = df_work[df_work["_perfil"] == perfil_maduro][feat].mean()
        media_rezagado = df_work[df_work["_perfil"] == perfil_rezagado][feat].mean()
        diferencia = media_maduro - media_rezagado

        # Buscar importancia
        imp_row = importancias[importancias["variable"] == feat]
        importancia = imp_row["importancia_gini"].values[0] if len(imp_row) > 0 else 0

        # Clasificar
        if diferencia > 0.1:
            rol = "DRIVER"
            interpretacion = "Valores altos facilitan la madurez digital"
        elif diferencia < -0.1:
            rol = "BARRERA"
            interpretacion = "Valores altos se asocian a menor madurez digital"
        else:
            rol = "NEUTRAL"
            interpretacion = "No discrimina claramente entre perfiles"

        # Caso especial: ACCESO__INVERSO_ tiene logica invertida
        if "INVERSO" in feat.upper() and rol == "DRIVER":
            rol = "BARRERA"
            interpretacion = "Variable en escala inversa: valores altos indican peor acceso"
        elif "INVERSO" in feat.upper() and rol == "BARRERA":
            rol = "DRIVER"
            interpretacion = "Variable en escala inversa: valores bajos indican mejor acceso"

        resultado.append({
            "variable": feat,
            "etiqueta": ETIQUETAS_LEGIBLES.get(feat, feat),
            "importancia": round(importancia, 4),
            "media_perfil_maduro": round(media_maduro, 4),
            "media_perfil_rezagado": round(media_rezagado, 4),
            "diferencia": round(diferencia, 4),
            "clasificacion": rol,
            "interpretacion": interpretacion,
        })

    df_result = pd.DataFrame(resultado)
    df_result = df_result.sort_values("importancia", ascending=False)

    n_drivers = (df_result["clasificacion"] == "DRIVER").sum()
    n_barreras = (df_result["clasificacion"] == "BARRERA").sum()
    n_neutral = (df_result["clasificacion"] == "NEUTRAL").sum()

    print(f"\n  [DRIVERS/BARRERAS] {n_drivers} drivers, {n_barreras} barreras, {n_neutral} neutrales")

    return df_result


# =========================================================================
# FUNCIONES DE VISUALIZACION
# =========================================================================

def grafico_importancia_rf(importancias: pd.DataFrame) -> str:
    """
    Genera grafico de importancia de variables del Random Forest.

    Retorna
    -------
    str
        Ruta del grafico.
    """
    top_n = min(15, len(importancias))
    datos = importancias.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.4)))
    colores = ["#2ecc71" if v > importancias["importancia_gini"].median() else "#3498db"
               for v in datos["importancia_gini"]]

    ax.barh(
        range(top_n), datos["importancia_gini"].values,
        color=colores, alpha=0.85, edgecolor="white",
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(datos["etiqueta"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importancia (Gini)")
    ax.set_title("Variables que Discriminan Perfiles de Necesidad de Software\n(Random Forest — Importancia Gini)")

    # Agregar porcentaje
    for i, (val, pct) in enumerate(zip(datos["importancia_gini"], datos["pct_importancia"])):
        ax.text(val + 0.002, i, f"{pct:.1f}%", va="center", fontsize=8)

    plt.tight_layout()
    ruta = os.path.join(FIGURES_DIR, "ml_importancia_variables_rf.png")
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [GRAFICO] {ruta}")
    return ruta


def grafico_drivers_barreras(df_db: pd.DataFrame) -> str:
    """
    Genera grafico de barras divergentes drivers vs barreras.

    Retorna
    -------
    str
        Ruta del grafico.
    """
    # Filtrar solo variables relevantes (no neutrales)
    datos = df_db[df_db["clasificacion"] != "NEUTRAL"].copy()
    datos = datos.sort_values("diferencia", ascending=True)

    if len(datos) == 0:
        print("  [ADVERTENCIA] No se encontraron drivers ni barreras para graficar.")
        return None

    fig, ax = plt.subplots(figsize=(10, max(5, len(datos) * 0.4)))

    colores = ["#2ecc71" if c == "DRIVER" else "#e74c3c" for c in datos["clasificacion"]]
    ax.barh(range(len(datos)), datos["diferencia"].values, color=colores, alpha=0.85)
    ax.set_yticks(range(len(datos)))
    ax.set_yticklabels(datos["etiqueta"].values, fontsize=9)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Diferencia de media (perfil maduro - perfil rezagado)")
    ax.set_title("Drivers y Barreras para la Adopcion Digital\n(verde = driver, rojo = barrera)")

    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", alpha=0.85, label="Driver (facilita adopcion)"),
        Patch(facecolor="#e74c3c", alpha=0.85, label="Barrera (bloquea adopcion)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    ruta = os.path.join(FIGURES_DIR, "ml_drivers_vs_barreras.png")
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [GRAFICO] {ruta}")
    return ruta


def grafico_arbol_decision(res_arbol: dict) -> str:
    """
    Genera visualizacion del arbol de decision.

    Retorna
    -------
    str
        Ruta del grafico.
    """
    dt = res_arbol["modelo"]
    n_leaves = dt.get_n_leaves()
    depth = dt.get_depth()

    fig, ax = plt.subplots(figsize=(max(20, n_leaves * 3), max(8, depth * 2.5)))
    plot_tree(
        dt,
        feature_names=res_arbol["feature_names_legibles"],
        class_names=res_arbol["class_names"],
        filled=True,
        rounded=True,
        fontsize=8,
        proportion=True,
        ax=ax,
    )
    ax.set_title("Arbol de Decision — Reglas de Clasificacion de Perfiles de Software", fontsize=14)

    plt.tight_layout()
    ruta = os.path.join(FIGURES_DIR, "ml_arbol_decision_perfiles.png")
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [GRAFICO] {ruta}")
    return ruta


# =========================================================================
# FUNCIONES DE GUARDADO
# =========================================================================

def guardar_resultados_explicabilidad(
    importancias: pd.DataFrame,
    drivers_barreras: pd.DataFrame,
    reglas_texto: str,
    cv_rf: float,
    cv_dt: float,
) -> list:
    """
    Guarda todas las tablas del modulo de explicabilidad.

    Retorna
    -------
    list
        Lista de rutas guardadas.
    """
    rutas = []

    ruta_imp = os.path.join(TABLES_DIR, "ml_importancia_variables_rf.csv")
    importancias.to_csv(ruta_imp, index=False, encoding="utf-8-sig")
    rutas.append(ruta_imp)
    print(f"  [GUARDADO] {ruta_imp}")

    ruta_db = os.path.join(TABLES_DIR, "ml_drivers_barreras_adopcion.csv")
    drivers_barreras.to_csv(ruta_db, index=False, encoding="utf-8-sig")
    rutas.append(ruta_db)
    print(f"  [GUARDADO] {ruta_db}")

    ruta_reglas = os.path.join(TABLES_DIR, "ml_reglas_arbol_decision.csv")
    df_reglas = pd.DataFrame({
        "contenido": [reglas_texto],
        "accuracy_cv_rf": [round(cv_rf, 4)],
        "accuracy_cv_dt": [round(cv_dt, 4)],
    })
    df_reglas.to_csv(ruta_reglas, index=False, encoding="utf-8-sig")
    rutas.append(ruta_reglas)
    print(f"  [GUARDADO] {ruta_reglas}")

    return rutas


# =========================================================================
# ORQUESTADOR DEL MODULO
# =========================================================================

def ejecutar_explicabilidad() -> dict:
    """
    Funcion principal del modulo. Entrena RF y arbol de decision,
    clasifica drivers/barreras, genera graficos y tablas.

    Retorna
    -------
    dict
        Resultados del modulo.
    """
    print("\n  --- Explicabilidad: Drivers y Barreras ---")

    # Cargar datos
    df_scores, df_perfiles = cargar_datos_explicabilidad()

    # Construir dataset explicativo
    X, y, feature_names, df_merged = construir_dataset_explicativo(df_scores, df_perfiles)

    # Random Forest para importancias
    res_rf = entrenar_random_forest(X, y, feature_names)

    # Construir mapeo de nombres de perfil
    nombres_perfiles = {}
    ruta_perf_tabla = os.path.join(TABLES_DIR, "ml_perfiles_necesidad_software.csv")
    if os.path.exists(ruta_perf_tabla):
        df_perf_tabla = pd.read_csv(ruta_perf_tabla)
        if "nombre_perfil" in df_perf_tabla.columns:
            for idx, row in df_perf_tabla.iterrows():
                nombres_perfiles[idx] = row["nombre_perfil"]

    # Arbol de decision para reglas
    res_dt = entrenar_arbol_decision(X, y, feature_names, nombres_perfiles)

    # Clasificar drivers y barreras
    drivers_barreras = clasificar_drivers_barreras(
        res_rf["importancias"], df_merged, feature_names, y
    )

    print(f"\n  Reglas del arbol de decision:")
    print(res_dt["reglas_texto"][:500] + "..." if len(res_dt["reglas_texto"]) > 500 else res_dt["reglas_texto"])

    # Graficos
    rutas_graficos = []
    rg1 = grafico_importancia_rf(res_rf["importancias"])
    rutas_graficos.append(rg1)

    rg2 = grafico_drivers_barreras(drivers_barreras)
    if rg2:
        rutas_graficos.append(rg2)

    rg3 = grafico_arbol_decision(res_dt)
    rutas_graficos.append(rg3)

    # Guardar
    rutas_tablas = guardar_resultados_explicabilidad(
        importancias=res_rf["importancias"],
        drivers_barreras=drivers_barreras,
        reglas_texto=res_dt["reglas_texto"],
        cv_rf=res_rf["cv_accuracy"],
        cv_dt=res_dt["cv_accuracy"],
    )

    print(f"\n  [RESULTADO] {len(rutas_tablas)} tablas, {len(rutas_graficos)} graficos generados")

    return {
        "importancias": res_rf["importancias"],
        "drivers_barreras": drivers_barreras,
        "reglas": res_dt["reglas_texto"],
        "res_rf": res_rf,
        "res_dt": res_dt,
    }
