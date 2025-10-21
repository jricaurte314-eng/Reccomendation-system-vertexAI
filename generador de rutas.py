"""Ruta Builder – generación de rutas de aprendizaje a partir de reglas

Este módulo contiene utilidades para:
- Cargar y normalizar catálogos de contenidos (CSV/XLSX).
- Cargar equivalencias de habilidades cliente→plataforma (CSV).
- Cargar reglas (YAML) y componer un *scope* efectivo por habilidad/nivel.
- Puntuar contenidos según reglas y preferencias.
- Seleccionar una ruta con ILP (PuLP) o heurística *greedy*.
- Ejecutar en *batch* para todas las habilidades/niveles del archivo de equivalencias.

Listo para usar en repositorios públicos: incluye *type hints*, *docstrings*,
validaciones y manejo de dependencias opcionales (PuLP).

Requisitos básicos: pandas, numpy, pyyaml. PuLP es opcional (solo si usarás ILP).

Autor: Juan David Ricaurte Manrique
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

import math
import sys

import numpy as np
import pandas as pd
import yaml

# Dependencia opcional: PuLP. Se importa de forma *lazily* cuando se usa ILP.
try:  # pragma: no cover - import opcional
    import pulp as pl
except Exception:  # pragma: no cover - si no está instalado, seguimos
    pl = None  # type: ignore


# ============================
# Utilidades de texto/fechas
# ============================

def _to_list(x: object) -> list[str]:
    """Convierte valores varios a lista de *strings* (separador ";" o ",")."""
    if x is None:
        return []
    if isinstance(x, float) and math.isnan(x):
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    return [i.strip() for i in str(x).replace(",", ";").split(";") if i.strip()]


def _norm_str(x: object) -> str:
    """Normaliza a texto en minúsculas sin espacios extremos."""
    if x is None:
        return ""
    s = str(x).strip().lower()
    return s


def _parse_mes_lanzamiento(x: object) -> Optional[datetime]:
    """Interpreta fechas como mes/año. Acepta "YYYY-mm", "mm/YYYY", "YYYY-mm-dd" o solo año.

    Devuelve primer día del mes o del año si solo se pasa el año.
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip()
    for fmt in ("%Y-%m", "%Y/%m", "%m/%Y", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return datetime(dt.year, dt.month, 1)
        except Exception:
            pass
    if s.isdigit() and len(s) == 4:
        return datetime(int(s), 1, 1)
    return None


def months_since(dt: Optional[datetime], ref: Optional[datetime] = None) -> Optional[int]:
    """Meses transcurridos desde *dt* hasta *ref* (hoy por defecto)."""
    if dt is None:
        return None
    ref = ref or datetime.today()
    return (ref.year - dt.year) * 12 + (ref.month - dt.month)


# ============================
# Carga/normalización de datos
# ============================

def load_contenidos(path_csv_or_xlsx: str | Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Carga catálogo de contenidos desde CSV/XLSX y normaliza columnas clave.

    Normalizaciones incluidas:
    - Duración a entero (columna "Duración (minutos)").
    - Banderas booleanas: Idioma_norm, CoCert_norm, Plataforma_externa, UBITS_Max, En_APP.
    - Detección de rutas/programas, libros e idioma.
    - Tipo_norm (usa columna del nuevo buscador si existe; si no, la anterior).
    - Habs_norm: lista normalizada de 1..3 habilidades.
    - Nivel_norm, Partner_norm.
    - Meses_desde_lanzamiento calculado si hay "Mes de Lanzamiento".
    """
    path = Path(path_csv_or_xlsx)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        df = pd.read_csv(path)

    # Duración
    if "Duración (minutos)" in df.columns:
        df["Duración (minutos)"] = pd.to_numeric(df["Duración (minutos)"], errors="coerce").fillna(0).astype(int)
    else:
        df["Duración (minutos)"] = 0

    # Idioma
    if "Idioma" in df.columns:
        df["Idioma_norm"] = df["Idioma"].astype(str).str.strip().str.lower()
    else:
        df["Idioma_norm"] = ""

    # Booleans frecuentes
    def _to_bool_col(series_name: str) -> pd.Series:
        vals = {"si", "sí", "yes", "true", "1"}
        if series_name in df.columns:
            return df[series_name].astype(str).str.strip().str.lower().isin(vals)
        return pd.Series(False, index=df.index)

    df["CoCert_norm"] = _to_bool_col("CoCertificado (Si/No)")
    df["Plataforma_externa"] = _to_bool_col("Plataforma externa de contenido")
    df["UBITS_Max"] = _to_bool_col("UBITS Max (Si/No)")
    df["En_APP"] = _to_bool_col("Contenidos en la APP")

    # Rutas/programas
    df["EsRutaPrograma"] = False
    col_ruta = "Título de la Ruta o Programa (si hace parte de un programa/ruta)"
    if col_ruta in df.columns:
        df["EsRutaPrograma"] = df[col_ruta].astype(str).str.strip().ne("")

    # Fecha de lanzamiento → meses desde
    if "Mes de Lanzamiento" in df.columns:
        df["Mes_dt"] = df["Mes de Lanzamiento"].apply(_parse_mes_lanzamiento)
        df["Meses_desde_lanzamiento"] = df["Mes_dt"].apply(lambda d: months_since(d) if d else None)
    else:
        df["Meses_desde_lanzamiento"] = None

    # Tipo
    tipo_new = 'Tipo de Contenido\n"NUEVO BUSCADOR"'
    tipo_old = 'Tipo de Contenido\n"ANTERIOR"'
    if tipo_new in df.columns:
        df["Tipo_norm"] = df[tipo_new].astype(str).str.strip()
    elif tipo_old in df.columns:
        df["Tipo_norm"] = df[tipo_old].astype(str).str.strip()
    else:
        df["Tipo_norm"] = ""

    # Habilidades 1..3 → lista normalizada
    for col in ["Habilidad Principal (1)", "Habilidad (2)", "Habilidad (3)"]:
        if col not in df.columns:
            df[col] = ""
    df["Habs_norm"] = (
        df[["Habilidad Principal (1)", "Habilidad (2)", "Habilidad (3)"]]
        .apply(lambda r: [str(h).strip().lower() for h in r if str(h).strip()], axis=1)
    )

    # Nivel/partner
    df["Nivel_norm"] = df.get("Nivel de la Audiencia", pd.Series("", index=df.index)).astype(str).str.strip().str.title()
    df["Partner_norm"] = df.get("Partner", pd.Series("", index=df.index)).astype(str).str.strip()

    # Libros
    df["Es_Libro"] = df.get("ISBN (Libros)", pd.Series(np.nan, index=df.index)).notna()

    # ID: si no existe, creamos uno para trazabilidad
    if "ID" not in df.columns:
        df.insert(0, "ID", np.arange(1, len(df) + 1))

    return df


def load_equivalencias(path_csv: str | Path) -> pd.DataFrame:
    """Carga equivalencias habilidad cliente→plataforma desde CSV."""
    eq = pd.read_csv(path_csv)
    eq["habilidad_cliente_norm"] = eq["habilidad_cliente"].astype(str).str.strip().str.lower()
    eq["nivel_cliente_norm"] = eq["nivel_cliente"].astype(str).str.strip().str.title()
    eq["nivel_objetivo_norm"] = eq["nivel_objetivo"].astype(str).str.strip().str.title()
    eq["equivalentes_list"] = eq["habilidades_equivalentes_plataforma"].apply(_to_list).apply(
        lambda xs: [x.lower() for x in xs]
    )
    return eq


def load_reglas(path_yaml: str | Path) -> dict:
    """Carga un archivo YAML con reglas.

    Estructura mínima esperada (ejemplo):
    ```yaml
    default:
      pesos:
        habilidad_principal: 1.0
        habilidad_secundaria: 0.6
        habilidad_equivalente: 0.8
        nivel_match_exacto: 0.5
        nivel_un_abajo: 0.2
        nivel_un_arriba: 0.1
        recencia: 0.2
        partner_priorizado: 0.2
        tipo_priorizado: 0.2
        ubits_max: 0.3
        app: 0.1
      hard_filters:
        excluir_idiomas: ["portugués"]
        excluir_cocertificados: false
        excluir_rutas_programas: false
        excluir_plataforma_externa: false
        excluir_duracion_min_lt: 3
        excluir_duracion_min_gt: 600
      preferencias:
        priorizar_recencia: true
        priorizar_app: false
        ubits_max_vale_mas: true
        partners_priorizados: ["ACME"]
        tipos_priorizados: ["Curso", "Taller"]
    habilidades:
      "python":
        niveles:
          "Básico":
            minutos_objetivo: 240
            tolerancia_minutos: [210, 270]
            diversidad:
              minimo_tipos: ["Curso", "Taller"]
              max_por_partner: 0.7
    ```
    """
    with open(path_yaml, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ============================
# Filtros y *scoring*
# ============================

def apply_hard_filters(df: pd.DataFrame, rules_scope: dict) -> pd.DataFrame:
    """Aplica filtros "duros" definidos en el *scope* de reglas."""
    df2 = df.copy()
    hard = rules_scope.get("hard_filters", {}) or {}

    # Idiomas a excluir
    excl_idiomas = {str(s).lower() for s in hard.get("excluir_idiomas", [])}
    if excl_idiomas:
        df2 = df2[~df2["Idioma_norm"].isin(excl_idiomas)]

    if hard.get("excluir_cocertificados", False):
        df2 = df2[~df2["CoCert_norm"]]

    if hard.get("excluir_rutas_programas", False):
        df2 = df2[~df2["EsRutaPrograma"]]

    if hard.get("excluir_plataforma_externa", False):
        df2 = df2[~df2["Plataforma_externa"]]

    # Extremos de duración
    if "excluir_duracion_min_lt" in hard:
        df2 = df2[df2["Duración (minutos)"] >= int(hard["excluir_duracion_min_lt"]) ]
    if "excluir_duracion_min_gt" in hard:
        df2 = df2[df2["Duración (minutos)"] <= int(hard["excluir_duracion_min_gt"]) ]

    # Alcance local (si existen columnas)
    if hard.get("excluir_alcance_local_no_co", False) and "País" in df2.columns:
        df2 = df2[df2["País"].astype(str).str.strip().isin(["Colombia", "Global"])]

    # Sectores
    excl_secs = {str(s).strip() for s in hard.get("excluir_sectores_no_aplicables", [])}
    if excl_secs and "Sector" in df2.columns:
        df2 = df2[~df2["Sector"].astype(str).str.strip().isin(excl_secs)]

    # Desactualizados
    if hard.get("excluir_desactualizados", False) and "Desactualizado" in df2.columns:
        df2 = df2[~df2["Desactualizado"].astype(bool)]

    return df2


def compute_score(row: pd.Series, skill_equivs: set[str], rules_scope: dict, pesos: dict) -> float:
    """Calcula un *score* aditivo para un contenido dado el *scope* y equivalencias."""
    score = 0.0

    # Habilidades
    habs_lower = {str(h).lower().strip() for h in (row.get("Habs_norm") or [])}
    main = str(row.get("Habilidad Principal (1)") or "").lower().strip()

    if main and main in skill_equivs:
        score += float(pesos.get("habilidad_principal", 1.0))

    if any((h in skill_equivs) for h in habs_lower if h and h != main):
        score += float(pesos.get("habilidad_secundaria", 0.6))

    if any((eq in habs_lower) for eq in skill_equivs):
        score += float(pesos.get("habilidad_equivalente", 0.8))

    # Nivel
    nivel: str = str(row.get("Nivel_norm") or "")
    nivel_obj: str = str(rules_scope.get("_nivel_objetivo", ""))
    levels = ["Básico", "Intermedio", "Avanzado"]
    if nivel in levels and nivel_obj in levels:
        di = levels.index(nivel) - levels.index(nivel_obj)
        if di == 0:
            score += float(pesos.get("nivel_match_exacto", 0.5))
        elif abs(di) == 1:
            score += float(pesos.get("nivel_un_abajo", 0.2)) if di < 0 else float(pesos.get("nivel_un_arriba", 0.1))

    # Preferencias
    prefs = rules_scope.get("preferencias", {}) or {}
    if row.get("Partner_norm") in set(prefs.get("partners_priorizados", []) or []):
        score += float(pesos.get("partner_priorizado", 0.2))

    if str(row.get("Tipo_norm") or "") in set(prefs.get("tipos_priorizados", []) or []):
        score += float(pesos.get("tipo_priorizado", 0.2))

    if prefs.get("ubits_max_vale_mas", False) and bool(row.get("UBITS_Max")):
        score += float(pesos.get("ubits_max", 0.3))

    if prefs.get("priorizar_app", False) and bool(row.get("En_APP")):
        score += float(pesos.get("app", 0.1))

    if prefs.get("priorizar_recencia", False):
        m = row.get("Meses_desde_lanzamiento")
        if m is not None:
            rec = max(0.0, 1.0 - min(int(m), 24) / 24.0)  # 0..1
            score += rec * float(pesos.get("recencia", 0.2))

    return float(score)


# ============================
# Reglas (scope efectivo)
# ============================

def get_rules_scope(rules: dict, habilidad_cliente: str, nivel_cliente: str) -> dict:
    """Crea un *scope* efectivo a partir de reglas, habilidad y nivel del cliente."""
    scope: dict = {}

    # default
    for k, v in (rules.get("default", {}) or {}).items():
        scope[k] = v

    # por habilidad
    hab_key = str(habilidad_cliente).strip().lower()
    hsec = (rules.get("habilidades", {}) or {}).get(hab_key, {})
    for k, v in hsec.items():
        if k != "niveles":
            scope[k] = _deep_merge(scope.get(k, {}), v) if isinstance(v, dict) else v

    # por nivel
    nsec = (hsec.get("niveles", {}) or {}).get(str(nivel_cliente).strip().title(), {})
    for k, v in nsec.items():
        scope[k] = _deep_merge(scope.get(k, {}), v) if isinstance(v, dict) else v

    scope["_nivel_objetivo"] = str(nivel_cliente).strip().title()
    scope["pesos"] = _with_defaults(scope.get("pesos", {}), (rules.get("default", {}) or {}).get("pesos", {}))
    return scope


def _deep_merge(base: dict, extra: dict) -> dict:
    if not isinstance(base, dict) or not isinstance(extra, dict):
        return extra
    out = dict(base)
    for k, v in extra.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _with_defaults(p: Optional[dict], defaults: Optional[dict]) -> dict:
    q = dict(defaults or {})
    q.update(p or {})
    return q


# ============================
# Selección (ILP / Greedy)
# ============================

def _ensure_pulp_available():  # pragma: no cover - verificación simple
    if pl is None:
        raise RuntimeError(
            "PuLP no está instalado. Instala con `pip install pulp` o usa `usar_ilp=False`."
        )


def select_ilp(df_scored: pd.DataFrame, rules_scope: dict) -> pd.DataFrame:
    """Selección óptima con ILP (PuLP).

    Requiere columnas: 'ID', 'Duración (minutos)', 'Tipo_norm', 'Partner_norm', 'Score'.
    Respeta, si existen en reglas:
    - minutos_objetivo / tolerancia_minutos [lo, hi]
    - diversidad.minimo_tipos (lista de tipos a cubrir al menos una vez)
    - diversidad.max_por_partner (proporción \in (0,1])
    """
    if df_scored.empty:
        return pd.DataFrame()

    _ensure_pulp_available()

    # Parámetros
    target = int(rules_scope.get("minutos_objetivo", 240))
    tol = rules_scope.get("tolerancia_minutos", [int(target * 0.9), int(target * 1.1)])
    tol_lo = int(tol[0])
    tol_hi = int(tol[1])
    min_tipos = list((rules_scope.get("diversidad", {}) or {}).get("minimo_tipos", []))
    max_por_partner = float((rules_scope.get("diversidad", {}) or {}).get("max_por_partner", 1.0))

    # Validación columnas
    req_cols = ["ID", "Duración (minutos)", "Tipo_norm", "Partner_norm", "Score"]
    faltantes = [c for c in req_cols if c not in df_scored.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas para ILP: {faltantes}")

    df = df_scored.copy()
    df["Duración (minutos)"] = pd.to_numeric(df["Duración (minutos)"], errors="coerce").fillna(0).astype(int)
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce").fillna(0.0)

    items = list(df.index)
    dur = df["Duración (minutos)"].to_dict()
    score = df["Score"].to_dict()
    tipos = df["Tipo_norm"].fillna("").astype(str).to_dict()
    partners = df["Partner_norm"].fillna("").astype(str).to_dict()

    # Variables binarias
    x = {i: pl.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat="Binary") for i in items}

    prob = pl.LpProblem("seleccion_ruta", pl.LpMaximize)
    prob += pl.lpSum(score[i] * x[i] for i in items)
    prob += pl.lpSum(dur[i] * x[i] for i in items) >= tol_lo, "min_minutes"
    prob += pl.lpSum(dur[i] * x[i] for i in items) <= tol_hi, "max_minutes"

    # Diversidad mínima por tipo
    for t in sorted({t for t in min_tipos if t}):
        prob += pl.lpSum(x[i] for i in items if tipos[i] == t) >= 1, f"min_tipo_{t}"

    # Ratio por partner (lineal)
    if 0 < max_por_partner < 1.0:
        total_sel = pl.lpSum(x[i] for i in items)
        for p in sorted(set(partners.values())):
            prob += (
                pl.lpSum(x[i] for i in items if partners[i] == p) <= max_por_partner * total_sel
            ), f"ratio_partner_{p}"

    # Resolver
    prob.solve(pl.PULP_CBC_CMD(msg=False))
    status = pl.LpStatus.get(prob.status, "Unknown")
    if status not in ("Optimal", "Feasible"):
        return pd.DataFrame()

    chosen_idx = [i for i in items if float(pl.value(x[i]) or 0.0) > 0.5]
    out = df.loc[chosen_idx].copy()

    # Orden amigable
    tipo_order = {t: i for i, t in enumerate(["Charla", "Short", "Caso", "Taller", "Podcast", "Curso", "Libro", "Documento técnico"])}
    out["Tipo_ord"] = out["Tipo_norm"].map(lambda v: tipo_order.get(v, 999))
    lvl_order = {"Básico": 0, "Intermedio": 1, "Avanzado": 2}
    out["Nivel_ord"] = out.get("Nivel_norm", "").map(lambda v: lvl_order.get(v, 99) if isinstance(v, str) else 99)
    out = out.sort_values(["Nivel_ord", "Tipo_ord", "Score"], ascending=[True, True, False]).reset_index(drop=True)
    out.insert(0, "Orden", out.index + 1)
    out.drop(columns=["Tipo_ord", "Nivel_ord"], inplace=True, errors="ignore")
    return out


def select_greedy(df_scored: pd.DataFrame, rules_scope: dict) -> pd.DataFrame:
    """Heurística *greedy* para selección rápida bajo banda de minutos y diversidad básica."""
    if df_scored.empty:
        return df_scored.copy()

    target = int(rules_scope.get("minutos_objetivo", 240))
    tol = rules_scope.get("tolerancia_minutos", [int(target * 0.9), int(target * 1.1)])
    tol_lo = int(tol[0])
    tol_hi = int(tol[1])
    min_tipos = set((rules_scope.get("diversidad", {}) or {}).get("minimo_tipos", []))
    max_por_partner = float((rules_scope.get("diversidad", {}) or {}).get("max_por_partner", 1.0))

    total = 0
    chosen: list[pd.Series] = []
    count_por_partner: dict[str, int] = {}

    cand = df_scored.sort_values(["Score", "Duración (minutos)"], ascending=[False, True]).copy()

    for _, row in cand.iterrows():
        dur = int(row["Duración (minutos)"])
        if total + dur > tol_hi:
            continue
        p = str(row.get("Partner_norm") or "")
        nuevo_ratio = (count_por_partner.get(p, 0) + 1) / max(1, (len(chosen) + 1))
        if 0 < max_por_partner < 1.0 and nuevo_ratio > max_por_partner:
            continue
        chosen.append(row)
        count_por_partner[p] = count_por_partner.get(p, 0) + 1
        total += dur
        if total >= tol_lo:
            break

    # Fallback: si faltan minutos y se permite usar libros
    if total < tol_lo and (rules_scope.get("fallback", {}) or {}).get("permitir_libros_si_falta_minutos", False):
        libros = cand[cand["Es_Libro"] & ~cand["ID"].isin([c["ID"] for c in chosen])]
        for _, row in libros.iterrows():
            dur = int(row["Duración (minutos)"])
            if total + dur > tol_hi:
                continue
            chosen.append(row)
            total += dur
            if total >= tol_lo:
                break

    # Diversidad mínima por tipo
    tipos_presentes = {str(r.get("Tipo_norm") or "") for r in chosen}
    faltantes = list(min_tipos - tipos_presentes)
    if faltantes:
        for t in faltantes:
            opc = cand[(cand["Tipo_norm"] == t) & ~cand["ID"].isin([c["ID"] for c in chosen])]
            for _, row in opc.iterrows():
                dur = int(row["Duración (minutos)"])
                if total + dur <= tol_hi:
                    chosen.append(row)
                    total += dur
                    break

    out = pd.DataFrame(chosen)
    if out.empty:
        return out

    tipo_order = {t: i for i, t in enumerate(["Charla", "Short", "Caso", "Taller", "Podcast", "Curso", "Libro", "Documento técnico"])}
    out["Tipo_ord"] = out["Tipo_norm"].map(lambda x: tipo_order.get(x, 999))
    lvl_order = {"Básico": 0, "Intermedio": 1, "Avanzado": 2}
    out["Nivel_ord"] = out.get("Nivel_norm", "").map(lambda x: lvl_order.get(x, 99))
    out = out.sort_values(["Nivel_ord", "Tipo_ord", "Score"], ascending=[True, True, False]).reset_index(drop=True)
    out.insert(0, "Orden", out.index + 1)
    out.drop(columns=["Tipo_ord", "Nivel_ord"], inplace=True, errors="ignore")
    return out


# ============================
# Orquestación de rutas
# ============================

def construir_ruta_para(
    contenidos_df: pd.DataFrame,
    reglas: dict,
    equivalencias_df: pd.DataFrame,
    habilidad_cliente: str,
    nivel_cliente: str,
    usar_ilp: bool = False,
) -> pd.DataFrame:
    """Construye una ruta para una habilidad/nivel concretos."""
    scope = get_rules_scope(reglas, habilidad_cliente, nivel_cliente)

    # Equivalencias para la habilidad/nivel
    eq = equivalencias_df[
        (equivalencias_df["habilidad_cliente_norm"] == _norm_str(habilidad_cliente))
        & (equivalencias_df["nivel_cliente_norm"] == str(nivel_cliente).strip().title())
    ]
    if eq.empty:
        eq = equivalencias_df[(equivalencias_df["habilidad_cliente_norm"] == _norm_str(habilidad_cliente))]

    skill_equivs: set[str] = set()
    for _, r in eq.iterrows():
        skill_equivs.update(set(r["equivalentes_list"]))

    base = apply_hard_filters(contenidos_df, scope)
    if base.empty:
        return pd.DataFrame()

    pesos = scope.get("pesos", {}) or {}
    base = base.copy()
    base["Score"] = base.apply(lambda row: compute_score(row, skill_equivs, scope, pesos), axis=1)

    ruta = select_ilp(base, scope) if usar_ilp else select_greedy(base, scope)

    cols = [
        "Orden",
        "ID",
        "Nombre del Contenido",
        "Tipo_norm",
        "Duración (minutos)",
        "Habilidad Principal (1)",
        "Habilidad (2)",
        "Habilidad (3)",
        "Nivel_norm",
        "Partner_norm",
        "Idioma",
        "UBITS_Max",
        "En_APP",
        "Score",
    ]
    existentes = [c for c in cols if c in ruta.columns]
    out = ruta[existentes].rename(columns={
        "Tipo_norm": "Tipo",
        "Nivel_norm": "Nivel",
        "Partner_norm": "Partner",
    })
    return out


def construir_rutas_batch(
    contenidos_path: str | Path,
    equivalencias_path: str | Path,
    reglas_path: str | Path,
    output_csv: str | Path = "rutas_generadas.csv",
    sheet_name: Optional[str] = None,
    usar_ilp: bool = False,
) -> pd.DataFrame:
    """Construye rutas para todas las (habilidad, nivel) del archivo de equivalencias.

    Devuelve el DataFrame concatenado y guarda un CSV en *output_csv* si hay resultados.
    """
    contenidos = load_contenidos(contenidos_path, sheet_name=sheet_name)
    eq = load_equivalencias(equivalencias_path)
    reglas = load_reglas(reglas_path)

    salidas: list[pd.DataFrame] = []
    pares = sorted(set(zip(eq["habilidad_cliente_norm"], eq["nivel_cliente_norm"])) )
    for (hab_cli, nivel_cli) in pares:
        # recuperar etiquetas originales (legibles)
        sub = eq[(eq["habilidad_cliente_norm"] == hab_cli) & (eq["nivel_cliente_norm"] == nivel_cli)]
        hab_txt = sub["habilidad_cliente"].iloc[0]
        niv_txt = sub["nivel_cliente"].iloc[0]

        ruta = construir_ruta_para(contenidos, reglas, eq, hab_txt, niv_txt, usar_ilp=usar_ilp)
        if not ruta.empty:
            ruta.insert(0, "Habilidad Cliente", hab_txt)
            ruta.insert(1, "Nivel Cliente", niv_txt)
            salidas.append(ruta)

    if salidas:
        out = pd.concat(salidas, ignore_index=True)
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False, encoding="utf-8-sig")
        return out

    return pd.DataFrame()


# ============================
# Entrada por CLI (opcional)
# ============================
if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Construcción de rutas de aprendizaje")
    parser.add_argument("contenidos", help="Ruta a CSV/XLSX de contenidos")
    parser.add_argument("equivalencias", help="Ruta a CSV de equivalencias")
    parser.add_argument("reglas", help="Ruta a YAML de reglas")
    parser.add_argument("--out", default="rutas_generadas.csv", help="CSV de salida")
    parser.add_argument("--sheet", default=None, help="Nombre de hoja si es Excel")
    parser.add_argument("--ilp", action="store_true", help="Usar ILP (requiere PuLP)")

    args = parser.parse_args()

    try:
        df_out = construir_rutas_batch(
            contenidos_path=args.contenidos,
            equivalencias_path=args.equivalencias,
            reglas_path=args.reglas,
            output_csv=args.out,
            sheet_name=args.sheet,
            usar_ilp=args.ilp,
        )
        if df_out.empty:
            print("No se generaron rutas (sin candidatos tras filtros o conflicto de reglas).")
            sys.exit(2)
        print(f"OK: {len(df_out)} filas → {args.out}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
