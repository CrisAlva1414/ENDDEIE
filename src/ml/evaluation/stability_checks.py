# -*- coding: utf-8 -*-
"""
=============================================================================
Modulo de Evaluacion de Estabilidad
=============================================================================

Evalua la robustez de los clusters y ejes latentes identificados mediante
bootstrap y subsampling. Reporta la consistencia de las estructuras
encontradas y genera una tabla de estabilidad.

Entradas:
    - outputs/tables/scores_factores_establecimiento.csv
    - outputs/tables/ml_asignacion_perfiles_software.csv
Salidas:
    - outputs/tables/ml_estabilidad_clusters.csv
    - outputs/tables/ml_estabilidad_pca.csv
    - outputs/figures/ml_estabilidad_bootstrap.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
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
# CONSTANTES
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

N_BOOTSTRAP = 100
SUBSAMPLE_FRAC = 0.80


# =========================================================================
# FUNCIONES PRINCIPALES
# =========================================================================

def cargar_datos_estabilidad() -> pd.DataFrame:
    """
    Carga scores de factores para evaluacion de estabilidad.

    Retorna
    -------
    pd.DataFrame
        DataFrame con scores y dimensiones.
    """
    ruta = os.path.join(TABLES_DIR, "scores_factores_establecimiento.csv")
    df = pd.read_csv(ruta)
    print(f"  [CARGA] {df.shape[0]} establecimientos")
    return df


def evaluar_estabilidad_clusters(df: pd.DataFrame, k: int = None,
                                   n_iter: int = N_BOOTSTRAP,
                                   frac: float = SUBSAMPLE_FRAC) -> pd.DataFrame:
    """
    Evalua la estabilidad de la solucion de clustering mediante bootstrap.
    En cada iteracion:
      1. Se muestrea un subconjunto del dataset.
      2. Se ejecuta KMeans.
      3. Se compara la particion con la solucion de referencia usando ARI.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame con scores.
    k : int, opcional
        Numero de clusters. Si no se especifica, se determina primero.
    n_iter : int
        Numero de iteraciones de bootstrap.
    frac : float
        Fraccion del dataset para cada muestra.

    Retorna
    -------
    pd.DataFrame
        Tabla de estabilidad con ARI, silueta y distribucion por iteracion.
    """
    cols = [c for c in COLS_SCORES if c in df.columns]
    mask = df[cols].notna().all(axis=1)
    X_full = df.loc[mask, cols].values
    indices_validos = df.loc[mask].index.values

    # Solucion de referencia
    if k is None:
        # Usar k del archivo de perfiles si existe
        ruta_comp = os.path.join(TABLES_DIR, "ml_comparacion_enfoques_clustering.csv")
        if os.path.exists(ruta_comp):
            df_comp = pd.read_csv(ruta_comp)
            k = int(df_comp.iloc[0]["k_optimo"])
        else:
            k = 3

    km_ref = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels_ref = km_ref.fit_predict(X_full)
    sil_ref = silhouette_score(X_full, labels_ref)

    print(f"  [REFERENCIA] k={k}, silueta={sil_ref:.4f}, n={X_full.shape[0]}")

    # Bootstrap
    resultados = []
    np.random.seed(RANDOM_STATE)

    for i in range(n_iter):
        # Subsampling sin reemplazo
        n_sample = int(len(X_full) * frac)
        idx_sample = np.random.choice(len(X_full), size=n_sample, replace=False)
        X_sample = X_full[idx_sample]

        km_boot = KMeans(n_clusters=k, random_state=RANDOM_STATE + i, n_init=10)
        labels_boot = km_boot.fit_predict(X_sample)

        sil_boot = silhouette_score(X_sample, labels_boot)

        # Comparar con referencia (sobre las observaciones muestreadas)
        labels_ref_sub = labels_ref[idx_sample]

        # ARI entre la referencia restringida y la solucion bootstrap
        ari = adjusted_rand_score(labels_ref_sub, labels_boot)

        resultados.append({
            "iteracion": i + 1,
            "n_muestra": n_sample,
            "silueta": round(sil_boot, 4),
            "ari_vs_referencia": round(ari, 4),
        })

    df_estab = pd.DataFrame(resultados)

    print(f"\n  [ESTABILIDAD CLUSTERS] {n_iter} iteraciones bootstrap")
    print(f"    ARI medio: {df_estab['ari_vs_referencia'].mean():.4f} "
          f"+/- {df_estab['ari_vs_referencia'].std():.4f}")
    print(f"    Silueta media: {df_estab['silueta'].mean():.4f} "
          f"+/- {df_estab['silueta'].std():.4f}")

    return df_estab


def evaluar_estabilidad_pca(df: pd.DataFrame, n_iter: int = N_BOOTSTRAP,
                              frac: float = SUBSAMPLE_FRAC) -> pd.DataFrame:
    """
    Evalua la estabilidad de la estructura PCA (dimensiones compuestas)
    mediante bootstrap, comparando la varianza explicada y las cargas
    del primer componente.

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame con dimensiones compuestas.
    n_iter : int
        Numero de iteraciones.
    frac : float
        Fraccion de subsampling.

    Retorna
    -------
    pd.DataFrame
        Tabla de estabilidad de la estructura PCA.
    """
    cols = [c for c in COLS_DIMENSIONES if c in df.columns]
    mask = df[cols].notna().all(axis=1)
    X_full = df.loc[mask, cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    # Referencia
    pca_ref = PCA(random_state=RANDOM_STATE)
    pca_ref.fit(X_scaled)
    cargas_ref = pca_ref.components_[0]  # primer componente
    var_ref = pca_ref.explained_variance_ratio_

    print(f"  [PCA REFERENCIA] Varianza PC1: {var_ref[0]:.4f}, PC2: {var_ref[1]:.4f}")

    # Bootstrap
    resultados = []
    np.random.seed(RANDOM_STATE)

    for i in range(n_iter):
        n_sample = int(len(X_scaled) * frac)
        idx_sample = np.random.choice(len(X_scaled), size=n_sample, replace=False)
        X_sample = X_scaled[idx_sample]

        pca_boot = PCA(random_state=RANDOM_STATE)
        pca_boot.fit(X_sample)

        var_boot = pca_boot.explained_variance_ratio_
        cargas_boot = pca_boot.components_[0]

        # Similitud de cargas (correlacion absoluta, ya que el signo puede invertirse)
        correlacion_cargas = abs(np.corrcoef(cargas_ref, cargas_boot)[0, 1])

        resultados.append({
            "iteracion": i + 1,
            "varianza_pc1": round(var_boot[0], 4),
            "varianza_pc2": round(var_boot[1], 4),
            "correlacion_cargas_pc1": round(correlacion_cargas, 4),
        })

    df_estab = pd.DataFrame(resultados)

    print(f"\n  [ESTABILIDAD PCA] {n_iter} iteraciones bootstrap")
    print(f"    Varianza PC1 media: {df_estab['varianza_pc1'].mean():.4f} "
          f"+/- {df_estab['varianza_pc1'].std():.4f}")
    print(f"    Correlacion cargas PC1 media: {df_estab['correlacion_cargas_pc1'].mean():.4f} "
          f"+/- {df_estab['correlacion_cargas_pc1'].std():.4f}")

    return df_estab


def generar_resumen_estabilidad(df_clust: pd.DataFrame, df_pca: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen consolidado de la estabilidad de todos los analisis.

    Retorna
    -------
    pd.DataFrame
        Tabla resumen de estabilidad.
    """
    resumen = pd.DataFrame([
        {
            "analisis": "Clustering (KMeans)",
            "metrica": "ARI vs referencia",
            "media": round(df_clust["ari_vs_referencia"].mean(), 4),
            "desv_est": round(df_clust["ari_vs_referencia"].std(), 4),
            "min": round(df_clust["ari_vs_referencia"].min(), 4),
            "max": round(df_clust["ari_vs_referencia"].max(), 4),
            "n_iteraciones": len(df_clust),
            "interpretacion": _interpretar_ari(df_clust["ari_vs_referencia"].mean()),
        },
        {
            "analisis": "Clustering (KMeans)",
            "metrica": "Silueta bootstrap",
            "media": round(df_clust["silueta"].mean(), 4),
            "desv_est": round(df_clust["silueta"].std(), 4),
            "min": round(df_clust["silueta"].min(), 4),
            "max": round(df_clust["silueta"].max(), 4),
            "n_iteraciones": len(df_clust),
            "interpretacion": _interpretar_silueta(df_clust["silueta"].mean()),
        },
        {
            "analisis": "PCA (dimensiones)",
            "metrica": "Varianza PC1",
            "media": round(df_pca["varianza_pc1"].mean(), 4),
            "desv_est": round(df_pca["varianza_pc1"].std(), 4),
            "min": round(df_pca["varianza_pc1"].min(), 4),
            "max": round(df_pca["varianza_pc1"].max(), 4),
            "n_iteraciones": len(df_pca),
            "interpretacion": "Estructura dimensional estable" if df_pca["varianza_pc1"].std() < 0.02 else "Variabilidad moderada",
        },
        {
            "analisis": "PCA (dimensiones)",
            "metrica": "Correlacion cargas PC1",
            "media": round(df_pca["correlacion_cargas_pc1"].mean(), 4),
            "desv_est": round(df_pca["correlacion_cargas_pc1"].std(), 4),
            "min": round(df_pca["correlacion_cargas_pc1"].min(), 4),
            "max": round(df_pca["correlacion_cargas_pc1"].max(), 4),
            "n_iteraciones": len(df_pca),
            "interpretacion": _interpretar_correlacion_cargas(df_pca["correlacion_cargas_pc1"].mean()),
        },
    ])

    return resumen


def _interpretar_ari(ari_medio: float) -> str:
    """Interpreta el ARI medio."""
    if ari_medio >= 0.8:
        return "Clusters altamente estables"
    elif ari_medio >= 0.6:
        return "Clusters moderadamente estables"
    elif ari_medio >= 0.4:
        return "Estabilidad parcial, interpretar con cautela"
    else:
        return "Clusters inestables, estructura debil"


def _interpretar_silueta(sil_media: float) -> str:
    """Interpreta la silueta media."""
    if sil_media >= 0.5:
        return "Separacion clara entre clusters"
    elif sil_media >= 0.25:
        return "Estructura razonable con superposicion parcial"
    elif sil_media >= 0.1:
        return "Superposicion significativa, coherente con datos continuos"
    else:
        return "Sin estructura clara de clusters"


def _interpretar_correlacion_cargas(corr_media: float) -> str:
    """Interpreta la correlacion media de cargas PCA."""
    if corr_media >= 0.95:
        return "Ejes latentes muy estables"
    elif corr_media >= 0.85:
        return "Ejes latentes estables"
    elif corr_media >= 0.70:
        return "Ejes moderadamente estables"
    else:
        return "Ejes inestables, revisar estructura"


# =========================================================================
# FUNCIONES DE VISUALIZACION
# =========================================================================

def grafico_estabilidad_bootstrap(df_clust: pd.DataFrame,
                                    df_pca: pd.DataFrame) -> str:
    """
    Genera grafico de distribucion de metricas de estabilidad.

    Retorna
    -------
    str
        Ruta del grafico.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: ARI clusters
    axes[0, 0].hist(df_clust["ari_vs_referencia"], bins=20, color="#3498db", alpha=0.7, edgecolor="white")
    media_ari = df_clust["ari_vs_referencia"].mean()
    axes[0, 0].axvline(x=media_ari, color="#e74c3c", linestyle="--",
                        label=f"Media: {media_ari:.3f}")
    axes[0, 0].set_xlabel("ARI (Adjusted Rand Index)")
    axes[0, 0].set_ylabel("Frecuencia")
    axes[0, 0].set_title("Estabilidad de Clusters (ARI vs referencia)")
    axes[0, 0].legend()

    # Panel 2: Silueta clusters
    axes[0, 1].hist(df_clust["silueta"], bins=20, color="#2ecc71", alpha=0.7, edgecolor="white")
    media_sil = df_clust["silueta"].mean()
    axes[0, 1].axvline(x=media_sil, color="#e74c3c", linestyle="--",
                        label=f"Media: {media_sil:.3f}")
    axes[0, 1].set_xlabel("Coeficiente de Silueta")
    axes[0, 1].set_ylabel("Frecuencia")
    axes[0, 1].set_title("Estabilidad de Silueta (bootstrap)")
    axes[0, 1].legend()

    # Panel 3: Varianza PC1
    axes[1, 0].hist(df_pca["varianza_pc1"], bins=20, color="#e67e22", alpha=0.7, edgecolor="white")
    media_var = df_pca["varianza_pc1"].mean()
    axes[1, 0].axvline(x=media_var, color="#e74c3c", linestyle="--",
                        label=f"Media: {media_var:.3f}")
    axes[1, 0].set_xlabel("Varianza Explicada (PC1)")
    axes[1, 0].set_ylabel("Frecuencia")
    axes[1, 0].set_title("Estabilidad de Varianza PCA (PC1)")
    axes[1, 0].legend()

    # Panel 4: Correlacion cargas PC1
    axes[1, 1].hist(df_pca["correlacion_cargas_pc1"], bins=20, color="#9b59b6", alpha=0.7, edgecolor="white")
    media_corr = df_pca["correlacion_cargas_pc1"].mean()
    axes[1, 1].axvline(x=media_corr, color="#e74c3c", linestyle="--",
                        label=f"Media: {media_corr:.3f}")
    axes[1, 1].set_xlabel("Correlacion Absoluta de Cargas (PC1)")
    axes[1, 1].set_ylabel("Frecuencia")
    axes[1, 1].set_title("Estabilidad de Estructura PCA (cargas PC1)")
    axes[1, 1].legend()

    plt.suptitle("Evaluacion de Estabilidad — Bootstrap (100 iteraciones, 80% subsampling)",
                  fontsize=13, y=1.02)
    plt.tight_layout()

    ruta = os.path.join(FIGURES_DIR, "ml_estabilidad_bootstrap.png")
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [GRAFICO] {ruta}")
    return ruta


# =========================================================================
# FUNCIONES DE GUARDADO
# =========================================================================

def guardar_resultados_estabilidad(
    df_clust: pd.DataFrame,
    df_pca: pd.DataFrame,
    resumen: pd.DataFrame,
) -> list:
    """
    Guarda tablas del modulo de estabilidad.

    Retorna
    -------
    list
        Lista de rutas guardadas.
    """
    rutas = []

    ruta_clust = os.path.join(TABLES_DIR, "ml_estabilidad_clusters.csv")
    df_clust.to_csv(ruta_clust, index=False, encoding="utf-8-sig")
    rutas.append(ruta_clust)
    print(f"  [GUARDADO] {ruta_clust}")

    ruta_pca = os.path.join(TABLES_DIR, "ml_estabilidad_pca.csv")
    df_pca.to_csv(ruta_pca, index=False, encoding="utf-8-sig")
    rutas.append(ruta_pca)
    print(f"  [GUARDADO] {ruta_pca}")

    ruta_resumen = os.path.join(TABLES_DIR, "ml_resumen_estabilidad.csv")
    resumen.to_csv(ruta_resumen, index=False, encoding="utf-8-sig")
    rutas.append(ruta_resumen)
    print(f"  [GUARDADO] {ruta_resumen}")

    return rutas


# =========================================================================
# ORQUESTADOR DEL MODULO
# =========================================================================

def ejecutar_evaluacion_estabilidad() -> dict:
    """
    Funcion principal del modulo. Ejecuta bootstrap para clusters y PCA,
    genera resumen, graficos y tablas.

    Retorna
    -------
    dict
        Resultados del modulo.
    """
    print("\n  --- Evaluacion de Estabilidad ---")

    # Cargar datos
    df = cargar_datos_estabilidad()

    # Determinar k del clustering principal
    ruta_comp = os.path.join(TABLES_DIR, "ml_comparacion_enfoques_clustering.csv")
    k = None
    if os.path.exists(ruta_comp):
        df_comp = pd.read_csv(ruta_comp)
        k = int(df_comp.iloc[0]["k_optimo"])

    # Estabilidad de clusters
    df_estab_clust = evaluar_estabilidad_clusters(df, k=k)

    # Estabilidad de PCA
    df_estab_pca = evaluar_estabilidad_pca(df)

    # Resumen consolidado
    resumen = generar_resumen_estabilidad(df_estab_clust, df_estab_pca)

    print(f"\n  Resumen de estabilidad:")
    for _, row in resumen.iterrows():
        print(f"    {row['analisis']} ({row['metrica']}): "
              f"{row['media']:.4f} +/- {row['desv_est']:.4f} -> {row['interpretacion']}")

    # Graficos
    rutas_graficos = []
    rg1 = grafico_estabilidad_bootstrap(df_estab_clust, df_estab_pca)
    rutas_graficos.append(rg1)

    # Guardar
    rutas_tablas = guardar_resultados_estabilidad(df_estab_clust, df_estab_pca, resumen)

    print(f"\n  [RESULTADO] {len(rutas_tablas)} tablas, {len(rutas_graficos)} graficos generados")

    return {
        "estabilidad_clusters": df_estab_clust,
        "estabilidad_pca": df_estab_pca,
        "resumen": resumen,
    }
