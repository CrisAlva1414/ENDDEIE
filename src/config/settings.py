# -*- coding: utf-8 -*-
"""
Configuracion centralizada del proyecto ENDDEIE 2023.
Define paths, parametros globales y constantes utilizadas en todo el pipeline.
"""

import os

# =============================================================================
# PATHS BASE DEL PROYECTO
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

# =============================================================================
# ARCHIVOS DE DATOS
# =============================================================================
ARCHIVOS_CSV = {
    "estudiantes": "ESTUDIANTES_PUBLICA_2023.csv",
    "docentes": "DOCENTES_PUBLICA_2023.csv",
    "directores": "DIRECTORES_URBANOS_Y_RURALES_PUBLICA_2023.csv",
    "directores_urbanos": "DIRECTORES_URBANOS_PUBLICA_2023.csv",
    "directores_rurales": "DIRECTORES_RURALES_PUBLICA_2023.csv",
    "coordinadores": "COORDINADORES_PUBLICA_2023.csv",
    "pauta": "PAUTA_PUBLICA_2023.csv",
}

# =============================================================================
# COLUMNAS COMUNES DE IDENTIFICACION Y ESTRATIFICACION
# =============================================================================
COLS_ID = ["AGNO", "ID", "COD_DEPE2", "ESTRATO_ANALITICO", "RURAL_RBD"]

# =============================================================================
# INDICADORES COMPUESTOS POR ACTOR (columnas IND_ precalculadas)
# =============================================================================
INDICADORES_ESTUDIANTES = [
    "IND_ACCNES_FORM_TD",
    "IND_FREC_FORM_CD",
    "IND_FREC_ACTS_TD",
    "IND_FREC_USO_INT",
    "IND_FREC_USO_DD",
    "IND_AUTP_HAB_DIG1",
    "IND_AUTP_HAB_DIG2",
    "IND_NIVEL_MOT_TIC1",
    "IND_NIVEL_MOT_TIC2",
    "IND_VAL_TD_EST",
    "IND_AUTP_FORM_PENSAR",
    "IND_PERC_PROT_EST",
    "IND_FREC_APR_ACT",
    "IND_AUTP_APR_CLASES",
    "IND_PERC_IMP_TD",
]

INDICADORES_DOCENTES = [
    "IND_PERC_CULT_INN",
    "IND_CON_IMPL_MET",
    "IND_COND_INC_TD",
    "IND_FREC_ACTS_TD_EST",
    "IND_FREC_ACTS_TD_DOC1",
    "IND_FREC_ACTS_TD_DOC2",
    "IND_FREC_ACTS_TD_DOC3",
    "IND_AUTP_HAB_DIG1",
    "IND_AUTP_HAB_DIG2",
    "IND_NIVEL_MOT_TIC",
    "IND_NIVEL_PERC_TIC",
    "IND_PERC_IMP_TD",
    "IND_EFECTOS_POSITIVOS",
    "IND_EFECTOS_NEGATIVOS1",
    "IND_EFECTOS_NEGATIVOS2",
    "IND_IMPL_INN_EDUCATIVAS",
    "IND_COLAB_MEJORA",
    "IND_BAJA_FLEX_PD",
    "IND_PERC_TEC_INN",
    "IND_PERC_POLEDUC_INN",
    "IND_PERC_AMB_ESCOLAR",
    "IND_PERC_MET_ESTR",
    "IND_INC_INTS_EST",
]

INDICADORES_DIRECTORES = [
    "IND_PERC_CULT_INN",
    "IND_CONT_CAP_INN",
    "IND_IMPL_INN_EDUCATIVAS",
    "IND_INC_TD_ENS_APR",
    "IND_NUM_ESP_APOYO_TIC",
    "IND_PERC_IMP_TD",
    "IND_PROM_CULT_INN",
    "IND_MEC_FORT_INN",
    "IND_INIC_MEJORA",
    "IND_COLAB_MEJORA",
    "IND_DISP_IMPL_CAM",
    "IND_PERC_COND_INN",
    "IND_PERC_POLEDUC_INN",
    "IND_COND_COMED_INN",
    "IND_ELEM_MOT_INN",
]

# Dimensiones compuestas a nivel de establecimiento (presentes en directores y coordinadores)
DIMENSIONES_COMPUESTAS = [
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

# Indicadores de infraestructura (pauta)
INDICADORES_PAUTA = [
    "TASA_SALAS_LABORATORIOS",
    "TASA_DEPENDENCIAS_INTERNET",
    "TASA_SALAS_INTERNET",
]

# =============================================================================
# MAPEO DE DIMENSIONES ESTRUCTURALES PARA EL ANALISIS
# =============================================================================
# Agrupacion tematica de las dimensiones compuestas en factores estructurales
FACTORES_ESTRUCTURALES = {
    "gestion_liderazgo": {
        "descripcion": "Gestion y liderazgo para la innovacion digital",
        "dimensiones": [
            "LIDERAZGO_ESCOLAR_PARA_LA_INNOVACION",
            "PRACTICAS_Y_PROCESOS_PARA_INNOVAR",
            "MARCO_INSTITUCIONAL",
        ],
    },
    "cultura_innovacion": {
        "descripcion": "Cultura y mentalidad frente a la innovacion",
        "dimensiones": [
            "MENTALIDAD_FRENTE_A_LA_INNOVACION",
            "PROMOTORES_Y_BARRERAS_PARA_INNOVAR",
            "ACTITUDES",
        ],
    },
    "apropiacion_pedagogica": {
        "descripcion": "Apropiacion pedagogica de tecnologias digitales",
        "dimensiones": [
            "INNOVACION_EN_EL_PROCESO_ENSENANZA_Y_APRENDIZAJE",
            "ACTIVIDADES",
            "APOYO_AL_USO",
        ],
    },
    "capacidades_digitales": {
        "descripcion": "Habilidades y capacidades digitales",
        "dimensiones": [
            "HABILIDADES",
            "EFECTOS",
        ],
    },
    "infraestructura_acceso": {
        "descripcion": "Infraestructura y acceso tecnologico",
        "dimensiones": [
            "ACCESO__INVERSO_",
        ],
    },
}

# =============================================================================
# PARAMETROS DE ANALISIS
# =============================================================================
RANDOM_STATE = 42
MAX_CLUSTERS = 8
MIN_CLUSTERS = 2
UMBRAL_CORRELACION = 0.3  # Correlacion minima para considerar relacion relevante
UMBRAL_BRECHA = 0.5       # Diferencia estandarizada minima para considerar brecha

# =============================================================================
# CONFIGURACION DE GRAFICOS
# =============================================================================
FIGSIZE_HEATMAP = (14, 10)
FIGSIZE_BAR = (12, 6)
FIGSIZE_BOX = (14, 7)
DPI = 150
PALETA_COLORES = "RdYlGn"
PALETA_CLUSTER = "Set2"

# =============================================================================
# ETIQUETAS LEGIBLES PARA FACTORES Y DIMENSIONES
# =============================================================================
ETIQUETAS_FACTORES = {
    "gestion_liderazgo": "Gestion y Liderazgo",
    "cultura_innovacion": "Cultura de Innovacion",
    "apropiacion_pedagogica": "Apropiacion Pedagogica",
    "capacidades_digitales": "Capacidades Digitales",
    "infraestructura_acceso": "Infraestructura y Acceso",
}

ETIQUETAS_ZONA = {
    0: "Urbano",
    1: "Rural",
}

ETIQUETAS_DEPENDENCIA = {
    1: "Municipal",
    2: "Part. Subvencionado",
    3: "Corp. Adm. Delegada",
}


def crear_directorios():
    """Crea los directorios de salida si no existen."""
    for directorio in [TABLES_DIR, FIGURES_DIR, REPORTS_DIR]:
        os.makedirs(directorio, exist_ok=True)
    print("[CONFIG] Directorios de salida verificados/creados.")
