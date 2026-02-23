# -*- coding: utf-8 -*-
"""
Modulo de mapeo de indicadores y dimensiones estructurales.
Construye un marco analitico que asocia cada indicador a dimensiones
tematicas, actores y factores estructurales de la digitalizacion escolar.
"""

import pandas as pd
from src.config.settings import (
    INDICADORES_ESTUDIANTES,
    INDICADORES_DOCENTES,
    INDICADORES_DIRECTORES,
    DIMENSIONES_COMPUESTAS,
    INDICADORES_PAUTA,
    FACTORES_ESTRUCTURALES,
    TABLES_DIR,
)
import os


def mapear_dimensiones() -> pd.DataFrame:
    """
    Construye un DataFrame que mapea cada indicador a su actor, dimension
    tematica y factor estructural correspondiente.

    Retorna
    -------
    pd.DataFrame
        Mapa de indicadores con columnas:
        [indicador, actor, dimension_tematica, factor_estructural, descripcion_factor].
    """
    registros = []

    # --- Indicadores de estudiantes ---
    mapeo_est = {
        "IND_ACCNES_FORM_TD": "Acceso a formacion en tecnologia digital",
        "IND_FREC_FORM_CD": "Frecuencia de formacion en competencias digitales",
        "IND_FREC_ACTS_TD": "Frecuencia de actividades con tecnologia digital",
        "IND_FREC_USO_INT": "Frecuencia de uso de internet",
        "IND_FREC_USO_DD": "Frecuencia de uso de dispositivos digitales",
        "IND_AUTP_HAB_DIG1": "Autopercepcion de habilidades digitales (basicas)",
        "IND_AUTP_HAB_DIG2": "Autopercepcion de habilidades digitales (avanzadas)",
        "IND_NIVEL_MOT_TIC1": "Nivel de motivacion hacia TIC (intrinseca)",
        "IND_NIVEL_MOT_TIC2": "Nivel de motivacion hacia TIC (extrinseca)",
        "IND_VAL_TD_EST": "Valoracion de la tecnologia digital por estudiantes",
        "IND_AUTP_FORM_PENSAR": "Autopercepcion de formacion en pensamiento",
        "IND_PERC_PROT_EST": "Percepcion de proteccion estudiantil",
        "IND_FREC_APR_ACT": "Frecuencia de aprendizaje activo",
        "IND_AUTP_APR_CLASES": "Autopercepcion de aprendizaje en clases",
        "IND_PERC_IMP_TD": "Percepcion de importancia de la tecnologia digital",
    }
    for ind, desc in mapeo_est.items():
        registros.append({
            "indicador": ind,
            "actor": "Estudiante",
            "descripcion_indicador": desc,
            "dimension_tematica": _clasificar_dimension_tematica(ind, "estudiante"),
        })

    # --- Indicadores de docentes ---
    mapeo_doc = {
        "IND_PERC_CULT_INN": "Percepcion de cultura de innovacion",
        "IND_CON_IMPL_MET": "Conocimiento e implementacion de metodologias",
        "IND_COND_INC_TD": "Condiciones para incorporar tecnologia digital",
        "IND_FREC_ACTS_TD_EST": "Frecuencia de actividades TD con estudiantes",
        "IND_FREC_ACTS_TD_DOC1": "Frecuencia de actividades TD docentes (tipo 1)",
        "IND_FREC_ACTS_TD_DOC2": "Frecuencia de actividades TD docentes (tipo 2)",
        "IND_FREC_ACTS_TD_DOC3": "Frecuencia de actividades TD docentes (tipo 3)",
        "IND_AUTP_HAB_DIG1": "Autopercepcion habilidades digitales (basicas)",
        "IND_AUTP_HAB_DIG2": "Autopercepcion habilidades digitales (avanzadas)",
        "IND_NIVEL_MOT_TIC": "Nivel de motivacion hacia TIC",
        "IND_NIVEL_PERC_TIC": "Nivel de percepcion de TIC",
        "IND_PERC_IMP_TD": "Percepcion de importancia de TD",
        "IND_EFECTOS_POSITIVOS": "Percepcion de efectos positivos de TD",
        "IND_EFECTOS_NEGATIVOS1": "Percepcion de efectos negativos (tipo 1)",
        "IND_EFECTOS_NEGATIVOS2": "Percepcion de efectos negativos (tipo 2)",
        "IND_IMPL_INN_EDUCATIVAS": "Implementacion de innovaciones educativas",
        "IND_COLAB_MEJORA": "Colaboracion para la mejora",
        "IND_BAJA_FLEX_PD": "Baja flexibilidad en practica docente",
        "IND_PERC_TEC_INN": "Percepcion de tecnologia como innovacion",
        "IND_PERC_POLEDUC_INN": "Percepcion de politica educativa e innovacion",
        "IND_PERC_AMB_ESCOLAR": "Percepcion del ambiente escolar",
        "IND_PERC_MET_ESTR": "Percepcion de metodologias y estrategias",
        "IND_INC_INTS_EST": "Incorporacion de intereses estudiantiles",
    }
    for ind, desc in mapeo_doc.items():
        registros.append({
            "indicador": ind,
            "actor": "Docente",
            "descripcion_indicador": desc,
            "dimension_tematica": _clasificar_dimension_tematica(ind, "docente"),
        })

    # --- Indicadores de directores ---
    mapeo_dir = {
        "IND_PERC_CULT_INN": "Percepcion de cultura de innovacion",
        "IND_CONT_CAP_INN": "Contribucion a capacidades de innovacion",
        "IND_IMPL_INN_EDUCATIVAS": "Implementacion de innovaciones educativas",
        "IND_INC_TD_ENS_APR": "Incorporacion de TD en ensenanza-aprendizaje",
        "IND_NUM_ESP_APOYO_TIC": "Numero de espacios de apoyo TIC",
        "IND_PERC_IMP_TD": "Percepcion de importancia de TD",
        "IND_PROM_CULT_INN": "Promocion de cultura de innovacion",
        "IND_MEC_FORT_INN": "Mecanismos para fortalecer la innovacion",
        "IND_INIC_MEJORA": "Iniciativas de mejora",
        "IND_COLAB_MEJORA": "Colaboracion para la mejora",
        "IND_DISP_IMPL_CAM": "Disposicion a implementar cambios",
        "IND_PERC_COND_INN": "Percepcion de condiciones para innovar",
        "IND_PERC_POLEDUC_INN": "Percepcion de politica educativa e innovacion",
        "IND_COND_COMED_INN": "Condiciones y comedimiento para innovar",
        "IND_ELEM_MOT_INN": "Elementos motivadores de la innovacion",
    }
    for ind, desc in mapeo_dir.items():
        registros.append({
            "indicador": ind,
            "actor": "Director",
            "descripcion_indicador": desc,
            "dimension_tematica": _clasificar_dimension_tematica(ind, "director"),
        })

    # --- Dimensiones compuestas ---
    mapeo_dim = {
        "LIDERAZGO_ESCOLAR_PARA_LA_INNOVACION": "Liderazgo escolar para la innovacion",
        "PRACTICAS_Y_PROCESOS_PARA_INNOVAR": "Practicas y procesos para innovar",
        "MENTALIDAD_FRENTE_A_LA_INNOVACION": "Mentalidad frente a la innovacion",
        "PROMOTORES_Y_BARRERAS_PARA_INNOVAR": "Promotores y barreras para innovar",
        "INNOVACION_EN_EL_PROCESO_ENSENANZA_Y_APRENDIZAJE": "Innovacion en ensenanza-aprendizaje",
        "ACCESO__INVERSO_": "Acceso tecnologico (escala inversa)",
        "MARCO_INSTITUCIONAL": "Marco institucional",
        "APOYO_AL_USO": "Apoyo al uso de tecnologias",
        "ACTIVIDADES": "Actividades con tecnologia digital",
        "HABILIDADES": "Habilidades digitales",
        "ACTITUDES": "Actitudes hacia la tecnologia",
        "EFECTOS": "Efectos percibidos de la tecnologia",
    }
    for dim, desc in mapeo_dim.items():
        registros.append({
            "indicador": dim,
            "actor": "Establecimiento",
            "descripcion_indicador": desc,
            "dimension_tematica": _clasificar_dimension_compuesta(dim),
        })

    # --- Indicadores de infraestructura ---
    mapeo_infra = {
        "TASA_SALAS_LABORATORIOS": "Tasa de salas con laboratorios",
        "TASA_DEPENDENCIAS_INTERNET": "Tasa de dependencias con internet",
        "TASA_SALAS_INTERNET": "Tasa de salas con internet",
    }
    for ind, desc in mapeo_infra.items():
        registros.append({
            "indicador": ind,
            "actor": "Infraestructura",
            "descripcion_indicador": desc,
            "dimension_tematica": "Infraestructura y Acceso",
        })

    df_mapa = pd.DataFrame(registros)

    # Asignar factor estructural
    df_mapa["factor_estructural"] = df_mapa["dimension_tematica"].apply(
        _asignar_factor_estructural
    )

    return df_mapa


def _clasificar_dimension_tematica(indicador: str, actor: str) -> str:
    """Clasifica un indicador en una dimension tematica segun su nombre y actor."""
    ind = indicador.upper()

    if any(k in ind for k in ["HAB_DIG", "HABILIDADES"]):
        return "Capacidades Digitales"
    elif any(k in ind for k in ["FREC_ACTS", "FREC_USO", "FREC_APR", "ACTIVIDADES"]):
        return "Apropiacion Pedagogica"
    elif any(k in ind for k in ["MOT_TIC", "VAL_TD", "PERC_IMP", "ACTITUDES"]):
        return "Cultura de Innovacion"
    elif any(k in ind for k in ["CULT_INN", "PROM_CULT", "MEC_FORT", "INIC_MEJORA",
                                 "COLAB_MEJORA", "DISP_IMPL", "IMPL_INN"]):
        return "Gestion y Liderazgo"
    elif any(k in ind for k in ["COND_INC", "COND_COMED", "PERC_COND",
                                 "PERC_POLEDUC", "PERC_AMB", "PERC_MET",
                                 "CONT_CAP", "BAJA_FLEX"]):
        return "Condiciones Institucionales"
    elif any(k in ind for k in ["FORM_TD", "FORM_CD", "FORM_PENSAR", "CON_IMPL"]):
        return "Formacion y Desarrollo Profesional"
    elif any(k in ind for k in ["EFECTO", "PROT_EST", "INC_INTS", "APR_CLASES",
                                 "PERC_TEC"]):
        return "Efectos y Resultados"
    elif any(k in ind for k in ["ACCESO", "NUM_ESP", "INC_TD_ENS"]):
        return "Infraestructura y Acceso"
    else:
        return "Otros"


def _clasificar_dimension_compuesta(dimension: str) -> str:
    """Clasifica una dimension compuesta del establecimiento."""
    mapeo = {
        "LIDERAZGO_ESCOLAR_PARA_LA_INNOVACION": "Gestion y Liderazgo",
        "PRACTICAS_Y_PROCESOS_PARA_INNOVAR": "Gestion y Liderazgo",
        "MENTALIDAD_FRENTE_A_LA_INNOVACION": "Cultura de Innovacion",
        "PROMOTORES_Y_BARRERAS_PARA_INNOVAR": "Cultura de Innovacion",
        "INNOVACION_EN_EL_PROCESO_ENSENANZA_Y_APRENDIZAJE": "Apropiacion Pedagogica",
        "ACCESO__INVERSO_": "Infraestructura y Acceso",
        "MARCO_INSTITUCIONAL": "Condiciones Institucionales",
        "APOYO_AL_USO": "Apropiacion Pedagogica",
        "ACTIVIDADES": "Apropiacion Pedagogica",
        "HABILIDADES": "Capacidades Digitales",
        "ACTITUDES": "Cultura de Innovacion",
        "EFECTOS": "Efectos y Resultados",
    }
    return mapeo.get(dimension, "Otros")


def _asignar_factor_estructural(dimension_tematica: str) -> str:
    """Asigna un factor estructural a partir de la dimension tematica."""
    mapeo = {
        "Gestion y Liderazgo": "gestion_liderazgo",
        "Cultura de Innovacion": "cultura_innovacion",
        "Apropiacion Pedagogica": "apropiacion_pedagogica",
        "Capacidades Digitales": "capacidades_digitales",
        "Infraestructura y Acceso": "infraestructura_acceso",
        "Condiciones Institucionales": "gestion_liderazgo",
        "Formacion y Desarrollo Profesional": "capacidades_digitales",
        "Efectos y Resultados": "apropiacion_pedagogica",
    }
    return mapeo.get(dimension_tematica, "otros")


def guardar_mapa_indicadores(df_mapa: pd.DataFrame) -> str:
    """
    Guarda el mapa de indicadores como CSV en outputs/tables/.

    Retorna
    -------
    str
        Ruta del archivo guardado.
    """
    ruta = os.path.join(TABLES_DIR, "mapa_indicadores_dimensiones.csv")
    df_mapa.to_csv(ruta, index=False, encoding="utf-8-sig")
    print(f"  [GUARDADO] Mapa de indicadores: {ruta}")
    return ruta
