
"""Routes Planner – construcción de planes por alumno a partir de rutas objetivo.

Limpieza y documentación del script original. Cambios clave:
- Tipos y docstrings en funciones principales.
- Normalización de columnas tolerante a alias.
- Índice invertido y *matcher* con memoización.
- Cálculo de *score* (si no viene) basado en CSAT/NPS + texto + nivel.
- Corrección de bugs en `build_plan_for_alumno` (variables no definidas, dedupe por ID,
  contadores por ruta, control de duración y *overflow* opcional).
- Pipeline completo `construir_planes_desde_recs` y ejecución desde Excel.

Autor: Tu Nombre | Licencia: MIT
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import re
import time
import unicodedata

# ================ Config =================
CONFIG: Dict[str, object] = {
    "max_minutes_per_alumno": 6 * 60,  # 6 horas
    "max_items_per_route": 5,
    "jaccard_threshold": 0.18,
    "prefer_same_level": True,
    "duration_floor_minutes": 2,
    "allow_overflow_last_item": False,
}

# ============== Utilidades generales ==============

def strip_accents(s: object) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return "".join(c for c in unicodedata.normalize("NFD", str(s)).lower() if unicodedata.category(c) != "Mn")


def tokenize(s: object) -> set[str]:
    s = strip_accents(s)
    return set(re.findall(r"[a-z0-9]{2,}", s))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _col_exists(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def _as_numeric(series: pd.Series, default: float | int = 0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _norm_cols(df: pd.DataFrame, alias_map: dict) -> pd.DataFrame:
    """Estándariza nombres de columnas según alias (tolerante a variantes)."""
    cols_std = {c: strip_accents(c).strip() for c in df.columns}
    inv: Dict[str, List[str]] = {}
    for orig, std in cols_std.items():
        inv.setdefault(std, []).append(orig)

    rename_map: Dict[str, str] = {}
    for want_std, alias_list in alias_map.items():
        # si ya existe exacto, no tocar
        if any(strip_accents(c) == want_std for c in df.columns):
            continue
        for alias in alias_list:
            std_alias = strip_accents(alias)
            if std_alias in inv:
                rename_map[inv[std_alias][0]] = alias  # mantener forma original del alias
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


# ============== Parser de rutas con código ==============
ROUTE_RE = re.compile(r"""^\s*(?P<codigo>[UD]\d{2}[A-Z]?)\s*-\s*(?P<nombre>[^:]+?)(?:\s*:\s*(?P<nivel_texto>.+))?\s*$""", re.VERBOSE | re.UNICODE)

NIVEL_CORTO_A_TEXTO = {
    "N1": "Básico (Nivel 1)",
    "N2": "Intermedio (Nivel 2)",
    "N3": "Intermedio (Nivel 3)",
    "N4": "Avanzado (Nivel 4)",
}
TEXTO_A_NIVEL_CORTO = {v: k for k, v in NIVEL_CORTO_A_TEXTO.items()}


def parse_route_item(raw: object) -> dict:
    """Parsea una cadena de ruta en {codigo, route, nivel_texto, nivel_corto}."""
    if not isinstance(raw, str):
        return {"codigo": "", "route": "", "nivel_texto": "", "nivel_corto": ""}
    s = raw.strip()
    if not s:
        return {"codigo": "", "route": "", "nivel_texto": "", "nivel_corto": ""}
    m = ROUTE_RE.match(s)
    if m:
        nivel_txt = (m.group("nivel_texto") or "").strip()
        return {
            "codigo": m.group("codigo").strip(),
            "route": m.group("nombre").strip(),
            "nivel_texto": nivel_txt,
            "nivel_corto": TEXTO_A_NIVEL_CORTO.get(nivel_txt, ""),
        }
    # fallback simple
    nivel_txt = s.split(":", 1)[1].strip() if ":" in s else ""
    left = s.split(":", 1)[0]
    if "-" in left:
        codigo, nombre = left.split("-", 1)
        return {
            "codigo": codigo.strip(),
            "route": nombre.strip(),
            "nivel_texto": nivel_txt,
            "nivel_corto": TEXTO_A_NIVEL_CORTO.get(nivel_txt, ""),
        }
    return {"codigo": "", "route": left.strip(), "nivel_texto": nivel_txt, "nivel_corto": TEXTO_A_NIVEL_CORTO.get(nivel_txt, "")}


def parse_rutas_cell(cell: object) -> list[dict]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    parts = [p.strip() for p in cell.split(";") if p.strip()]
    return [parse_route_item(p) for p in parts]


# ============== Scoring básico (si no viene) ==============

def ensure_scores(df_recs: pd.DataFrame) -> pd.DataFrame:
    """Asegura una columna `score_total` si no existe.
    Mezcla CSAT/NPS + similitud con habilidad cliente + bonus por nivel.
    """
    if "score_total" in df_recs.columns:
        return df_recs

    def quality_score(r: pd.Series) -> float:
        csat = pd.to_numeric(r.get("CSAT AVG (%)", 0), errors="coerce"); csat = 0 if pd.isna(csat) else float(csat)
        nps = pd.to_numeric(r.get("NPS AVG (%)", 0), errors="coerce"); nps = 0 if pd.isna(nps) else float(nps)
        return (0.6 * csat + 0.4 * nps) / 100.0

    def fallback_row_score(r: pd.Series) -> float:
        rel = 0.0
        if _col_exists(df_recs, "Nombre del Contenido") and _col_exists(df_recs, "Habilidad Cliente"):
            rel = jaccard(tokenize(r.get("Nombre del Contenido", "")), tokenize(r.get("Habilidad Cliente", ""))) * 60.0
        q = quality_score(r) * 60.0
        lvl = 0.0
        if _col_exists(df_recs, "Nivel") and _col_exists(df_recs, "Nivel Cliente"):
            lvl = 10.0 if strip_accents(r.get("Nivel", "")) == strip_accents(r.get("Nivel Cliente", "")) else 0.0
        return rel + q + lvl

    out = df_recs.copy()
    out["score_total"] = out.apply(fallback_row_score, axis=1)
    return out


# ============== Preprocesamiento e índice invertido ==============

def tokenize_fast(s: object) -> set[str]:
    s = strip_accents(s)
    return set(re.findall(r"[a-z0-9]{2,}", s))


def build_inverted_index(df_recs: pd.DataFrame) -> dict[str, np.ndarray]:
    """Precalcula tokens combinados por fila y un índice invertido token->indices.
    Añade `__dur__` (float32) y `__token_count__`.
    """
    text_cols = [c for c in ["Habilidad Cliente", "Habilidad Principal (1)", "Habilidad (2)", "Habilidad (3)", "Nombre del Contenido"] if c in df_recs.columns] or [c for c in ["Nombre del Contenido"] if c in df_recs.columns]

    tokens_all: list[frozenset[str]] = []
    for _, r in df_recs[text_cols].fillna("").iterrows():
        toks: set[str] = set()
        for c in text_cols:
            toks |= tokenize_fast(r[c])
        tokens_all.append(frozenset(toks))

    df_recs["__tokens_all__"] = tokens_all
    df_recs["__token_count__"] = df_recs["__tokens_all__"].map(len).astype("int32")

    if "Duración (minutos)" in df_recs.columns:
        df_recs["__dur__"] = pd.to_numeric(df_recs["Duración (minutos)"], errors="coerce").fillna(0.0).astype("float32")
    else:
        df_recs["__dur__"] = 0.0

    inv: dict[str, list[int]] = defaultdict(list)
    for i, toks in enumerate(tokens_all):
        for t in toks:
            inv[t].append(i)
    for k in list(inv.keys()):
        inv[k] = np.asarray(inv[k], dtype=np.int32)

    return inv


def make_route_matcher(df_recs: pd.DataFrame, inverted_index: dict[str, np.ndarray]):
    """Devuelve una función *match* enlazada a df_recs + índice. Con memoización."""

    @lru_cache(maxsize=4096)
    def _candidate_indices_for_tokens(route_tokens_fz: frozenset[str]) -> np.ndarray:
        if not route_tokens_fz:
            return np.array([], dtype=np.int32)
        acc: np.ndarray | None = None
        for t in route_tokens_fz:
            arr = inverted_index.get(t)
            if arr is None:
                continue
            acc = arr if acc is None else np.union1d(acc, arr)
        return acc if acc is not None else np.array([], dtype=np.int32)

    @lru_cache(maxsize=4096)
    def match_candidates_for_route(route_name: str, route_level_text: str, route_code: str = "", route_level_short: str = "") -> pd.DataFrame:
        # 1) match por código exacto (si existe)
        df_code = None
        if route_code and "codigo" in df_recs.columns:
            mask = df_recs["codigo"].astype(str).str.strip().str.upper().values == route_code.upper()
            idx_code = np.where(mask)[0]
            if idx_code.size > 0:
                df_code = df_recs.iloc[idx_code].copy()
                df_code.loc[:, "sim_ruta"] = 1.0

        # 2) candidatos por tokens
        route_tokens = tokenize_fast(route_name)
        idx_txt = _candidate_indices_for_tokens(frozenset(route_tokens))
        df_txt = df_recs.iloc[idx_txt].copy() if idx_txt.size > 0 else df_recs.iloc[[]].copy()

        # 3) similitud Jaccard
        if not df_txt.empty and len(route_tokens) > 0:
            inter = df_txt["__tokens_all__"].map(lambda s: len(route_tokens & s)).astype("int16")
            union = (len(route_tokens) + df_txt["__token_count__"] - inter).replace(0, 1)
            sim = (inter / union).astype("float32")
            df_txt.loc[:, "sim_ruta"] = sim
            df_txt = df_txt[df_txt["sim_ruta"] >= float(CONFIG["jaccard_threshold"])]
        else:
            df_txt["sim_ruta"] = np.array([], dtype="float32")

        # 4) bonus por nivel
        def _lvl_bonus(df_: pd.DataFrame) -> np.ndarray:
            if df_.empty or "Nivel" not in df_.columns or not route_level_text:
                return np.zeros(len(df_), dtype="float32")
            return (df_["Nivel"].fillna("").map(strip_accents).values == strip_accents(route_level_text)).astype("float32")

        if df_code is not None and not df_code.empty:
            df_code.loc[:, "lvl_bonus"] = _lvl_bonus(df_code)
            df_code.loc[:, "rank_score"] = df_code["score_total"].astype("float32") + 35.0 * df_code["sim_ruta"].astype("float32") + 6.0 * df_code["lvl_bonus"].astype("float32")

        if not df_txt.empty:
            df_txt.loc[:, "lvl_bonus"] = _lvl_bonus(df_txt)
            df_txt.loc[:, "rank_score"] = df_txt["score_total"].astype("float32") + 35.0 * df_txt["sim_ruta"].astype("float32") + 6.0 * df_txt["lvl_bonus"].astype("float32")

        # 5) mezcla priorizando código
        out = df_txt
        if df_code is not None and not df_code.empty:
            out = pd.concat([df_code, df_txt[~df_txt.index.isin(df_code.index)]], axis=0)

        if out.empty:
            return out
        sort_cols = [c for c in ["rank_score", "NPS AVG (%)", "CSAT AVG (%)"] if c in out.columns]
        return out.sort_values(sort_cols, ascending=False)

    return match_candidates_for_route


# ============== Construcción del plan (≤ max_minutes) ==============

def build_plan_for_alumno(df_recs_scored: pd.DataFrame, rutas: list[dict], max_minutes: int, match_fn) -> tuple[pd.DataFrame, dict]:
    """Construye un plan para un alumno dado un conjunto de rutas objetivo.

    Estrategia:
      1) Cobertura mínima: tomar el mejor candidato por ruta (sin exceder `max_items_per_route`) y respetando minutos.
      2) Relleno óptimo: *round-robin* por rutas, agregando mejores candidatos restantes hasta agotar minutos.
      3) Fallback laxo para caso de 1 ruta: intentar alcanzar ~80 minutos con *overflow* moderado si está permitido.
    """
    remaining = int(max_minutes)
    selections: list[dict] = []
    covered: set[str] = set()
    taken_ids: set[str] = set()
    per_route_counts: Dict[str, int] = defaultdict(int)

    # 1) Cobertura mínima (mejor candidato por ruta)
    for r in rutas:
        if remaining <= 0:
            break
        cands = match_fn(r.get("route", ""), r.get("nivel_texto", ""), r.get("codigo", ""), r.get("nivel_corto", ""))
        if cands.empty:
            continue
        # dedupe por ID ya tomado
        if "ID" in cands.columns:
            ids_series = cands["ID"].astype("string").where(cands["ID"].notna(), None)
            cands = cands[~ids_series.isin(taken_ids)]
            if cands.empty:
                continue
        # capacidad por duración
        floor = int(CONFIG["duration_floor_minutes"]) if "__dur__" in cands.columns else 0
        cands_ok = cands[cands["__dur__"] >= floor] if "__dur__" in cands.columns else cands
        cands_ok = cands_ok[cands_ok["__dur__"] <= remaining] if "__dur__" in cands_ok.columns else cands_ok
        if cands_ok.empty and bool(CONFIG["allow_overflow_last_item"]) and remaining > 0:
            cands_ok = cands.head(1)
        if cands_ok.empty:
            continue
        pick = cands_ok.iloc[0]
        dur = float(pick.get("__dur__", pd.to_numeric(pick.get("Duración (minutos)"), errors="coerce") or 0))
        if dur <= 0:
            continue
        # acepta
        rec = pick.to_dict()
        rec["ruta_objetivo"] = r.get("route")
        rec["codigo_ruta"] = r.get("codigo")
        rec["nivel_objetivo"] = r.get("nivel_texto")
        rec["fase"] = "cobertura_minima"
        selections.append(rec)
        if pick.get("ID") is not None:
            taken_ids.add(str(pick.get("ID")))
        remaining = max(0, remaining - int(dur))
        covered.add(str(r.get("codigo") or r.get("route")))
        per_route_counts[r.get("codigo", "")] += 1

    # 2) Relleno óptimo (round-robin por rutas)
    if remaining > 0:
        changed = True
        while remaining > 0 and changed:
            changed = False
            for r in rutas:
                if remaining <= 0:
                    break
                code = r.get("codigo", "")
                if per_route_counts[code] >= int(CONFIG["max_items_per_route"]):
                    continue
                cands = match_fn(r.get("route", ""), r.get("nivel_texto", ""), r.get("codigo", ""), r.get("nivel_corto", ""))
                if cands.empty:
                    continue
                # filtra IDs ya tomados
                if "ID" in cands.columns:
                    ids = cands["ID"].astype("string")
                    mask_new = ~(ids.notna() & ids.isin(taken_ids))
                    cands = cands[mask_new]
                if cands.empty:
                    continue
                # toma el primer que quepa
                for _, row in cands.iterrows():
                    dur = float(row.get("__dur__", pd.to_numeric(row.get("Duración (minutos)"), errors="coerce") or 0))
                    if dur <= 0:
                        continue
                    if not bool(CONFIG["allow_overflow_last_item"]) and dur > remaining:
                        continue
                    rec = row.to_dict()
                    rec["ruta_objetivo"] = r.get("route")
                    rec["codigo_ruta"] = r.get("codigo")
                    rec["nivel_objetivo"] = r.get("nivel_texto")
                    rec["fase"] = "relleno_optimo"
                    selections.append(rec)
                    if row.get("ID") is not None:
                        taken_ids.add(str(row.get("ID")))
                    per_route_counts[code] += 1
                    remaining -= int(dur)
                    changed = True
                    break

    # Construir plan base
    if selections:
        plan = pd.DataFrame(selections)
        plan["razon"] = "match por código y/o texto + score_total + duración"
        # dedup final por ID
        if "ID" in plan.columns:
            plan["ID"] = plan["ID"].astype("string")
            plan = plan.drop_duplicates(subset=["ID"], keep="first").reset_index(drop=True)
        plan["Level_usuario"] = plan["nivel_objetivo"]
        keep = [
            "ruta_objetivo", "codigo_ruta", "nivel_objetivo", "Level_usuario",
            "ID", "Nombre del Contenido", "Tipo", "Duración (minutos)", "__dur__",
            "Partner", "Idioma", "Habilidad Cliente", "Habilidad Principal (1)",
            "Habilidad (2)", "Habilidad (3)", "Nivel", "score_total", "sim_ruta",
            "rank_score", "fase", "razon",
        ]
        plan = plan[[c for c in keep if c in plan.columns]].reset_index(drop=True)
    else:
        plan = pd.DataFrame([
            {
                "ruta_objetivo": None,
                "codigo_ruta": None,
                "nivel_objetivo": None,
                "Level_usuario": None,
                "ID": None,
                "Nombre del Contenido": None,
                "Duración (minutos)": None,
                "razon": "Sin candidatos que cumplan umbrales/duración",
            }
        ])

    # Fallback laxo (1 ruta): intentar llegar a ~80 min
    if len(rutas) == 1 and remaining > 0:
        min_target = 80
        if "__dur__" in plan.columns:
            total_min_actual = float(pd.to_numeric(plan["__dur__"], errors="coerce").fillna(0).sum())
        else:
            total_min_actual = float(pd.to_numeric(plan.get("Duración (minutos)"), errors="coerce").fillna(0).sum())
        if total_min_actual < min_target:
            r = rutas[0]
            # pool: por código si existe; si no, todo
            if "codigo" in df_recs_scored.columns and r.get("codigo"):
                pool = df_recs_scored[df_recs_scored["codigo"].astype(str).str.strip().str.upper() == str(r["codigo"]).upper()].copy()
            else:
                pool = df_recs_scored.copy()
            sort_cols = [c for c in ["score_total", "NPS AVG (%)", "CSAT AVG (%)"] if c in pool.columns]
            pool = pool.sort_values(sort_cols + (["__dur__"] if "__dur__" in pool.columns else []), ascending=False)
            # quitar IDs ya presentes
            if "ID" in pool.columns and "ID" in plan.columns:
                already_ids = set(plan["ID"].astype("string").dropna())
                ids_series = pool["ID"].astype("string")
                mask_new = ~(ids_series.notna() & ids_series.isin(already_ids))
                pool = pool[mask_new]
            OVERFLOW_MARGIN = 30
            for _, row in pool.iterrows():
                if per_route_counts.get(r.get("codigo", ""), 0) >= int(CONFIG["max_items_per_route"]):
                    break
                dur = float(row.get("__dur__", pd.to_numeric(row.get("Duración (minutos)"), errors="coerce") or 0))
                if dur <= 0:
                    continue
                can_take = (dur <= remaining) or ((dur - remaining) <= OVERFLOW_MARGIN and total_min_actual < min_target)
                if not can_take:
                    continue
                rec = row.to_dict()
                rec["ruta_objetivo"] = r.get("route")
                rec["codigo_ruta"] = r.get("codigo")
                rec["nivel_objetivo"] = r.get("nivel_texto")
                rec["fase"] = "relleno_optimo"  # laxo
                # append al plan
                add_row = {
                    "ruta_objetivo": rec["ruta_objetivo"],
                    "codigo_ruta": rec["codigo_ruta"],
                    "nivel_objetivo": rec["nivel_objetivo"],
                    "Level_usuario": rec["nivel_objetivo"],
                    "ID": row.get("ID"),
                    "Nombre del Contenido": row.get("Nombre del Contenido"),
                    "Tipo": row.get("Tipo"),
                    "Duración (minutos)": row.get("Duración (minutos)"),
                    "__dur__": row.get("__dur__"),
                    "Partner": row.get("Partner"),
                    "Idioma": row.get("Idioma"),
                    "Habilidad Cliente": row.get("Habilidad Cliente"),
                    "Habilidad Principal (1)": row.get("Habilidad Principal (1)"),
                    "Habilidad (2)": row.get("Habilidad (2)"),
                    "Habilidad (3)": row.get("Habilidad (3)"),
                    "Nivel": row.get("Nivel"),
                    "score_total": row.get("score_total"),
                    "sim_ruta": row.get("sim_ruta", np.nan),
                    "rank_score": row.get("rank_score", np.nan),
                    "fase": "relleno_optimo",
                    "razon": "match por código y/o texto + score_total + duración",
                }
                plan = pd.concat([plan, pd.DataFrame([add_row])], ignore_index=True)
                per_route_counts[r.get("codigo", "")] = per_route_counts.get(r.get("codigo", ""), 0) + 1
                remaining -= int(dur)
                total_min_actual += dur
                if total_min_actual >= min_target or remaining <= 0:
                    break
            # dedup final
            if "ID" in plan.columns:
                plan["ID"] = plan["ID"].astype("string")
                plan = plan.drop_duplicates(subset=["ID"], keep="first").reset_index(drop=True)

    # summary
    total_min = (
        int(pd.to_numeric(plan["__dur__"], errors="coerce").fillna(0).sum()) if "__dur__" in plan.columns
        else int(pd.to_numeric(plan.get("Duración (minutos)"), errors="coerce").fillna(0).sum())
    )
    summary = {
        "total_minutos": total_min,
        "num_items": int(len(plan)),
        "rutas_cubiertas": sorted(list(covered)),
        "num_rutas_cubiertas": int(len(covered)),
    }
    return plan, summary


# ============== Pipeline principal (recs + brechas) ==============

def construir_planes_desde_recs(df_recs: pd.DataFrame, df_brechas: pd.DataFrame):
    """Construye planes por alumno a partir de un catálogo (`df_recs`) y una hoja de brechas (`df_brechas`)."""
    t0 = time.perf_counter()

    # alias tolerantes
    alias_recs = {
        strip_accents("Nombre del Contenido"): ["Nombre del Contenido", "titulo", "nombre_contenido"],
        strip_accents("Habilidad Cliente"): ["Habilidad Cliente", "habilidad", "habilidad_cliente"],
        strip_accents("Habilidad Principal (1)"): ["Habilidad Principal (1)", "habilidad_principal_1"],
        strip_accents("Habilidad (2)"): ["Habilidad (2)", "habilidad_2"],
        strip_accents("Habilidad (3)"): ["Habilidad (3)", "habilidad_3"],
        strip_accents("Nivel"): ["Nivel", "nivel_contenido"],
        strip_accents("Nivel Cliente"): ["Nivel Cliente", "nivel_cliente"],
        strip_accents("Duración (minutos)"): ["Duración (minutos)", "duracion", "duracion_minutos"],
        strip_accents("CSAT AVG (%)"): ["CSAT AVG (%)", "csat"],
        strip_accents("NPS AVG (%)"): ["NPS AVG (%)", "nps"],
        strip_accents("codigo"): ["codigo", "cod_ruta", "route_code"],
        strip_accents("ID"): ["ID", "id_contenido", "id"],
    }
    df_recs = _norm_cols(df_recs.copy(), alias_recs)
    df_recs = ensure_scores(df_recs)

    # duración e ID
    if _col_exists(df_recs, "Duración (minutos)"):
        df_recs.loc[:, "Duración (minutos)"] = _as_numeric(df_recs["Duración (minutos)"], default=0).astype("float32")
    if _col_exists(df_recs, "ID"):
        df_recs["ID"] = df_recs["ID"].astype("string")
        df_recs["ID"] = df_recs["ID"].where(df_recs["ID"].notna(), None)

    # brechas
    alias_brechas = {strip_accents("Alumno"): ["Alumno", "alumno", "estudiante", "empleado"], strip_accents("ruta"): ["ruta", "rutas_asignadas", "itinerario"]}
    df_brechas = _norm_cols(df_brechas.copy(), alias_brechas)
    for need in ["Alumno", "ruta"]:
        if not _col_exists(df_brechas, need):
            raise ValueError(f"Falta la columna obligatoria '{need}' en la hoja de brechas.")

    # índice + matcher
    inverted_index = build_inverted_index(df_recs)
    match_fn = make_route_matcher(df_recs, inverted_index)

    tmp = df_brechas.copy()
    tmp["rutas_parsed"] = tmp["ruta"].apply(parse_rutas_cell)
    tmp["num_rutas_recomendadas"] = tmp["rutas_parsed"].apply(len)

    all_plans: list[pd.DataFrame] = []
    resumenes: list[dict] = []

    for _, row in tmp.iterrows():
        alumno = row["Alumno"]
        rutas = row["rutas_parsed"]
        # minutos dinámicos por cantidad de rutas
        num_rutas = len(rutas)
        if num_rutas <= 2:
            max_minutes = 180  # 3 horas
        elif num_rutas <= 4:
            max_minutes = 300  # 5 horas
        else:
            max_minutes = int(CONFIG["max_minutes_per_alumno"])  # 6 horas

        plan, summary = build_plan_for_alumno(df_recs, rutas, max_minutes, match_fn)
        if not plan.empty:
            plan.insert(0, "Alumno", alumno)
        else:
            plan = pd.DataFrame([
                {
                    "Alumno": alumno,
                    "ruta_objetivo": None,
                    "ID": None,
                    "Nombre del Contenido": None,
                    "Duración (minutos)": None,
                    "razon": "Sin candidatos que cumplan umbrales/duración",
                    "Level_usuario": None,
                }
            ])
        all_plans.append(plan)
        resumenes.append({"Alumno": alumno, "num_rutas_recomendadas": num_rutas, **summary})

    planes = pd.concat(all_plans, axis=0, ignore_index=True)
    resumen = pd.DataFrame(resumenes)
    brechas_out = tmp.drop(columns=["rutas_parsed"])  # conserva conteo

    t1 = time.perf_counter()
    print(f"⏱️ Tiempo total pipeline: {t1 - t0:.2f}s")
    return planes, resumen, brechas_out


# ============== Ejecución desde un único Excel ==============
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Genera planes por alumno desde un Excel maestro")
    parser.add_argument("excel", nargs="?", default="Rutas_con_Rol.xlsx", help="Workbook maestro (.xlsx)")
    parser.add_argument("--recs-sheet", default="Rutas_con_Rol", help="Hoja de catálogo de contenidos")
    parser.add_argument("--brechas-sheet", default="Brechas_con_rutas_all", help="Hoja con alumnos y rutas")
    parser.add_argument("--out", default="planes_output2.xlsx", help="Excel de salida")
    args = parser.parse_args()

    xlsx_path = Path(args.excel)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {xlsx_path}")

    try:
        df_recs = pd.read_excel(xlsx_path, sheet_name=args.recs_sheet)
        df_brechas = pd.read_excel(xlsx_path, sheet_name=args.brechas_sheet)
    except ValueError as e:
        raise RuntimeError(f"Verifica los nombres de hojas en '{xlsx_path.name}'. Esperado: recs='{args.recs_sheet}', brechas='{args.brechas_sheet}'") from e

    planes, resumen, brechas_out = construir_planes_desde_recs(df_recs, df_brechas)

    with pd.ExcelWriter(args.out, engine="xlsxwriter") as writer:
        planes.to_excel(writer, sheet_name="planes_por_alumno", index=False)
        resumen.to_excel(writer, sheet_name="resumen_planes", index=False)
        brechas_out.to_excel(writer, sheet_name="brechas_con_conteo", index=False)

    print(f"✅ Listo. Resultados en '{args.out}'. [planes_por_alumno / resumen_planes / brechas_con_conteo]")
