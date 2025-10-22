/**
 * @file Encuestas a partir de habilidades/niveles en Google Sheets usando Gemini.
 * @author Tu Nombre
 * @license MIT
 *
 * Estructura esperada:
 * - Hoja personas (por defecto "Cargos_Habilidades_Niveles"):
 *   A: Persona (opcional)
 *   B: Cargo
 *   C: Actividades
 *   D..L: 3 sugerencias, cada una como [Habilidad, Nivel, Justificación]
 *
 * - Hoja encuestas (por defecto "encuestas") con encabezados:
 *   ['Encuesta_ID','Persona','Cargo','Habilidad','Nivel','Ítem #','Enunciado','Tipo','Escala','Indicador']
 */

/** =========================
 * ======== Config ==========
 * =========================*/

/** @enum {string} */
const LEVEL = /** @type {const} */ ({
  BASICO: 'basico',
  INTERMEDIO: 'intermedio',
  AVANZADO: 'avanzado',
});

/** @type {Readonly<{
 *  MODEL: string;
 *  ENDPOINT: string;
 *  TEMPERATURE: number;
 *  MAX_OUTPUT_TOKENS: number;
 *  PERSONAS_SHEET: string;
 *  CATALOG_SHEET: string;
 *  START_ROW: number;
 *  COL_PERSONA: number;
 *  COL_CARGO: number;
 *  COL_ACTIV: number;
 *  OUT_COL_START: number;
 *  OUT_COLS_PER_SUG: number;
 *  OUT_SUG_COUNT: number;
 *  MAX_RETRIES: number;
 *  CLEAR_DEST_BEFORE: boolean;
 *  MAX_ROWS_PER_BATCH: number;
 *  SHORTLIST_K_PER_ITEM: number;
 *  SHORTLIST_MAX_UNION: number;
 *  LEVELS: readonly string[];
 *  SURVEY_SHEET: string;
 *  SURVEY_CLEAR_BEFORE: boolean;
 *  SURVEY_ITEMS_PER_SKILL: number;
 *  SURVEY_MAX_ROWS_PER_BATCH: number;
 *  SURVEY_SCALE_LABEL: string;
 *  SURVEY_ITEM_MAX_WORDS: number;
 * }>}
 */
const CFG = Object.freeze({
  // === Gemini (texto) ===
  MODEL: 'models/gemini-2.5-flash-lite',
  ENDPOINT:
    'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key=',
  TEMPERATURE: 0.4,
  MAX_OUTPUT_TOKENS: 8192,

  // === Hojas y columnas ===
  PERSONAS_SHEET: 'Cargos_Habilidades_Niveles', // entrada/salida
  CATALOG_SHEET: 'Catalogo_Habilidades', // catálogo (no usado en este archivo)
  START_ROW: 2, // encabezados en fila 1

  // Columnas de personas (1-indexed)
  COL_PERSONA: 1, // A
  COL_CARGO: 2, // B
  COL_ACTIV: 3, // C

  // Salida desde D (9 columnas: 3 sugerencias × [habilidad, nivel, justificación])
  OUT_COL_START: 4, // D
  OUT_COLS_PER_SUG: 3,
  OUT_SUG_COUNT: 3,

  // Control de ejecución
  MAX_RETRIES: 3,
  CLEAR_DEST_BEFORE: true, // limpiar D:L antes (no se usa aquí; útil en otros flujos)
  MAX_ROWS_PER_BATCH: 20, // personas por prompt
  SHORTLIST_K_PER_ITEM: 25, // (reservado)
  SHORTLIST_MAX_UNION: 120, // (reservado)

  // Niveles permitidos
  LEVELS: /** @type {const} */ ([LEVEL.BASICO, LEVEL.INTERMEDIO, LEVEL.AVANZADO]),

  // ===== Encuestas =====
  SURVEY_SHEET: 'encuestas',
  SURVEY_CLEAR_BEFORE: true,
  SURVEY_ITEMS_PER_SKILL: 4,
  SURVEY_MAX_ROWS_PER_BATCH: 2,
  SURVEY_SCALE_LABEL: 'Likert 1-5',
  SURVEY_ITEM_MAX_WORDS: 22,
});

/** =========================
 * ========= Utils ==========
 * =========================*/

/**
 * Quita acentos/diacríticos.
 * @param {string} s
 * @returns {string}
 */
function _stripAccents_(s) {
  return String(s || '').normalize('NFD').replace(/[\u0300-\u036f]/g, '');
}

/**
 * Normaliza un nivel a minúsculas sin acentos.
 * @param {string} s
 * @returns {string}
 */
function _normalizeLevel_(s) {
  return _stripAccents_(String(s || '').toLowerCase().trim());
}

/**
 * Divide un arreglo en lotes del tamaño indicado.
 * @template T
 * @param {T[]} arr
 * @param {number} size
 * @returns {T[][]}
 */
function _chunk(arr, size) {
  if (!Array.isArray(arr) || size <= 0) return [arr || []];
  const out = [];
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
  return out;
}

/**
 * Devuelve la hoja de personas por nombre (búsqueda case-insensitive de respaldo).
 * @returns {GoogleAppsScript.Spreadsheet.Sheet|null}
 */
function _getPersonasSheet_() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sh = ss.getSheetByName(CFG.PERSONAS_SHEET);
  if (sh) return sh;

  // Búsqueda case-insensitive
  const target = CFG.PERSONAS_SHEET.toLowerCase();
  const cand = ss.getSheets().find((s) => s.getName().toLowerCase() === target);
  return cand || null;
}

/**
 * Asegura la hoja de encuestas con encabezados esperados.
 * @returns {GoogleAppsScript.Spreadsheet.Sheet}
 */
function _ensureSurveySheet_() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sh = ss.getSheetByName(CFG.SURVEY_SHEET);
  if (!sh) sh = ss.insertSheet(CFG.SURVEY_SHEET);

  const headers = [
    'Encuesta_ID',
    'Persona',
    'Cargo',
    'Habilidad',
    'Nivel',
    'Ítem #',
    'Enunciado',
    'Tipo',
    'Escala',
    'Indicador',
  ];

  if (sh.getLastRow() === 0) {
    sh.appendRow(headers);
  } else {
    const h = sh.getRange(1, 1, 1, headers.length).getValues()[0];
    if (h.join('¦') !== headers.join('¦')) {
      sh.clear();
      sh.appendRow(headers);
    }
  }
  return sh;
}

/**
 * Lee de una fila las habilidades/niveles asignados en columnas D..L.
 * Acepta "básico/intermedio/avanzado" con o sin tilde y en cualquier casing.
 * @param {any[]} rowArr Vector de celdas de la fila (A..L al menos)
 * @returns {{habilidad: string, nivel: string}[]}
 */
function _readAssignedSkillsFromRow_(rowArr) {
  const out = [];
  const base = CFG.OUT_COL_START - 1; // índice base (D)
  for (let s = 0; s < CFG.OUT_SUG_COUNT; s++) {
    const idxH = base + s * CFG.OUT_COLS_PER_SUG; // habilidad
    const idxN = base + s * CFG.OUT_COLS_PER_SUG + 1; // nivel
    const h = String(rowArr[idxH] || '').trim();
    const n = _normalizeLevel_(rowArr[idxN] || '');

    if (!h) continue;
    if (CFG.LEVELS.indexOf(n) < 0) continue;

    out.push({ habilidad: h, nivel: n });
  }
  return out;
}

/** =========================
 * ====== Flujo principal ===
 * =========================*/

/**
 * Lee personas + (habilidad, nivel), construye prompt por lote, llama Gemini y escribe encuestas.
 * Muestra un alert con el total de ítems escritos.
 */
function generarEncuestasBatch() {
  const sh = _getPersonasSheet_();
  if (!sh)
    throw new Error(
      'No encuentro la hoja de personas: ' +
        CFG.PERSONAS_SHEET +
        ' (búsqueda case-insensitive)',
    );

  const lastRow = sh.getLastRow();
  if (lastRow < CFG.START_ROW) return;

  // A..L (o lo necesario para 3 sugerencias completas)
  const NUM_COLS_IN = Math.max(
    12,
    CFG.OUT_COL_START - 1 + CFG.OUT_SUG_COUNT * CFG.OUT_COLS_PER_SUG,
  );
  const data = sh
    .getRange(CFG.START_ROW, 1, lastRow - CFG.START_ROW + 1, NUM_COLS_IN)
    .getValues();

  /** @type {{
   *  row: number; persona: string; cargo: string; actividades: string;
   *  asignadas: {habilidad: string, nivel: string}[];
   * }[]} */
  const personas = [];
  for (let i = 0; i < data.length; i++) {
    const rowIndex = CFG.START_ROW + i;
    const persona = String(data[i][CFG.COL_PERSONA - 1] || '').trim();
    const cargo = String(data[i][CFG.COL_CARGO - 1] || '').trim();
    const activ = String(data[i][CFG.COL_ACTIV - 1] || '').trim();

    const suggested = _readAssignedSkillsFromRow_(data[i]);
    if (suggested.length === 0) continue;

    personas.push({
      row: rowIndex,
      persona,
      cargo,
      actividades: activ,
      asignadas: suggested,
    });
  }

  if (personas.length === 0) {
    SpreadsheetApp.getUi().alert(
      'No hay habilidades/niveles asignados (D–L) para generar encuestas.',
    );
    return;
  }

  const shOut = _ensureSurveySheet_();

  // Limpia hoja de encuestas si aplica
  if (CFG.SURVEY_CLEAR_BEFORE) {
    const lr = shOut.getLastRow();
    if (lr >= 2) shOut.getRange(2, 1, lr - 1, shOut.getLastColumn()).clearContent();
  }

  const batches = _chunk(personas, CFG.SURVEY_MAX_ROWS_PER_BATCH);
  /** @type {any[][]} */
  let allRows = [];

  batches.forEach((items) => {
    const prompt = _buildSurveyBatchPrompt_(items);

    const payload = {
      contents: [{ role: 'user', parts: [{ text: prompt }] }],
      generationConfig: {
        temperature: CFG.TEMPERATURE,
        maxOutputTokens: CFG.MAX_OUTPUT_TOKENS,
      },
    };

    const apiKey = _getApiKey_();
    const url = CFG.ENDPOINT + encodeURIComponent(apiKey);
    const options = {
      method: 'post',
      contentType: 'application/json',
      payload: JSON.stringify(payload),
      muteHttpExceptions: true,
    };

    const obj = _callGeminiWithRetries_(url, options, (json) => {
      const raw = _extractTextFromGemini_(json) || '';
      const clean = _stripCodeFences_(_normalizeQuotes_(raw)).trim();
      return _safeParseJson_(clean);
    });

    const rows = _materializeSurveyRows_(items, obj);
    if (rows.length) allRows = allRows.concat(rows);
  });

  if (allRows.length) {
    shOut
      .getRange(shOut.getLastRow() + 1, 1, allRows.length, allRows[0].length)
      .setValues(allRows);
  }

  SpreadsheetApp.getUi().alert('Encuestas generadas: ' + allRows.length + ' ítems escritos.');
}

/** =========================
 * ========= Prompt =========
 * =========================*/

/**
 * Construye el prompt para varios candidatos con sus habilidades/niveles.
 * @param {{persona:string, cargo:string, actividades:string, asignadas:{habilidad:string,nivel:string}[]}[]} items
 * @returns {string}
 */
function _buildSurveyBatchPrompt_(items) {
  const candidatos = items.map((it, idx) => ({
    idx: idx + 1,
    persona: it.persona || '',
    cargo: it.cargo || '',
    actividades: it.actividades || '',
    habilidades: it.asignadas, // [{habilidad, nivel}]
  }));

  const reglas =
    'Eres un diseñador de instrumentos de evaluación laboral. ' +
    'Para CADA candidato y CADA (habilidad, nivel), genera EXACTAMENTE ' +
    CFG.SURVEY_ITEMS_PER_SKILL +
    ' ítems tipo Likert 1–5 ' +
    '(acotados a ' +
    CFG.SURVEY_ITEM_MAX_WORDS +
    ' palabras por enunciado) y un "indicador" breve (3–8 palabras). ' +
    'Usa verbos observables. Ajusta la complejidad según el nivel (básico=aplica/identifica; intermedio=analiza/resuelve; avanzado=diseña/optimiza/lidera). ' +
    'No inventes habilidades; usa solo las provistas. Español neutro, contexto organizacional.\n\n' +
    'Devuelve EXCLUSIVAMENTE JSON válido con esta forma EXACTA:\n' +
    '{ "resultados":[\n' +
    '  { "idx": <n>, "encuesta":[\n' +
    '    { "habilidad":"...", "nivel":"basico|intermedio|avanzado", "items":[\n' +
    '      { "n":1, "enunciado":"...", "tipo":"likert", "escala":"1-5", "indicador":"..." },\n' +
    '      { "n":2, "enunciado":"...", "tipo":"likert", "escala":"1-5", "indicador":"..." }\n' +
    '    ]}\n' +
    '  ]}\n' +
    ']}\n';

  return reglas + '\nCANDIDATOS:\n' + JSON.stringify(candidatos);
}

/** =========================
 * ======= Parse & I/O ======
 * =========================*/

/**
 * Permite parseo “tolerante” intentando encontrar el primer/último bloque JSON.
 * @param {string} t
 * @returns {any|null}
 */
function _parseJsonLenient_(t) {
  const s = String(t || '').trim();
  const i = s.indexOf('{');
  const j = s.lastIndexOf('}');
  if (i >= 0 && j > i) {
    try {
      return JSON.parse(s.slice(i, j + 1));
    } catch (e) {
      // sigue
    }
  }
  return null;
}

/**
 * Normaliza comillas tipográficas a simples/dobles estándar.
 * @param {string} s
 * @returns {string}
 */
function _normalizeQuotes_(s) {
  return String(s || '')
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'");
}

/**
 * Quita fences de código tipo ```json ... ``` o ``` ... ```
 * @param {string} s
 * @returns {string}
 */
function _stripCodeFences_(s) {
  return String(s || '').replace(/```[\s\S]*?```/g, (block) =>
    block.replace(/^```[a-zA-Z]*\n?/, '').replace(/```$/, ''),
  );
}

/**
 * Intenta JSON.parse; si falla, usa parseo tolerante.
 * @param {string} s
 * @returns {any|null}
 */
function _safeParseJson_(s) {
  try {
    return JSON.parse(s);
  } catch (e) {
    return _parseJsonLenient_(s);
  }
}

/**
 * Convierte objeto de respuesta en filas para escribir.
 * Columnas destino:
 * A: Encuesta_ID  B: Persona  C: Cargo  D: Habilidad  E: Nivel  F: Ítem #
 * G: Enunciado    H: Tipo     I: Escala J: Indicador
 * @param {Array<{row:number, persona:string, cargo:string, actividades:string, asignadas:{habilidad:string,nivel:string}[]}>} items
 * @param {any} obj JSON devuelto por el modelo
 * @returns {any[][]}
 */
function _materializeSurveyRows_(items, obj) {
  const rows = [];
  if (!obj || !Array.isArray(obj.resultados)) return rows;

  /** @type {Map<number, any[]>} */
  const mapByIdx = new Map();
  obj.resultados.forEach((r) => {
    if (r && typeof r.idx === 'number' && Array.isArray(r.encuesta)) {
      mapByIdx.set(r.idx, r.encuesta);
    }
  });

  const runId = Utilities.getUuid().slice(0, 8);

  items.forEach((it, localIdx) => {
    /** @type {any[]} */
    const bloques = mapByIdx.get(localIdx + 1) || [];

    // Reordena por el orden “want” provisto en la hoja, y filtra desconocidos
    const want = it.asignadas.map((p) => (p.habilidad || '').toLowerCase().trim());
    if (bloques && want.length) {
      bloques.sort((a, b) => {
        const ai = want.indexOf(String(a.habilidad || '').toLowerCase().trim());
        const bi = want.indexOf(String(b.habilidad || '').toLowerCase().trim());
        if (ai < 0 && bi < 0) return 0;
        if (ai < 0) return 1;
        if (bi < 0) return -1;
        return ai - bi;
      });
    }

    bloques.forEach((bloque, bIdx) => {
      if (!bloque || !Array.isArray(bloque.items)) return;
      const habilidad = String(bloque.habilidad || '').trim();
      let nivel = _normalizeLevel_(bloque.nivel || '');
      if (CFG.LEVELS.indexOf(nivel) < 0) nivel = '';

      const encId = 'ENC-' + runId + '-' + it.row + '-' + (bIdx + 1);

      // Toma primeros SURVEY_ITEMS_PER_SKILL
      const kmax = Math.min(CFG.SURVEY_ITEMS_PER_SKILL, bloque.items.length);
      for (let k = 0; k < kmax; k++) {
        const item = bloque.items[k] || {};
        const n = item.n || k + 1;
        let enun = String(item.enunciado || '').trim();
        const tipo = String(item.tipo || 'likert').trim();
        const escala = String(item.escala || '1-5').trim();
        const indic = String(item.indicador || '').trim();

        // recorte suave a SURVEY_ITEM_MAX_WORDS
        const words = enun.split(/\s+/);
        if (words.length > CFG.SURVEY_ITEM_MAX_WORDS) {
          enun = words.slice(0, CFG.SURVEY_ITEM_MAX_WORDS).join(' ').replace(/[.,;:]?$/, '');
        }

        rows.push([
          encId,
          it.persona || '',
          it.cargo || '',
          habilidad,
          nivel,
          n,
          enun,
          tipo || 'likert',
          escala || '1-5',
          indic,
        ]);
      }
    });
  });

  return rows;
}

/** =========================
 * ====== Llamada Gemini ====
 * =========================*/

/**
 * Obtiene la API key desde PropertiesService.
 * Define la propiedad "GEMINI_API_KEY" en Propiedades del script.
 * @returns {string}
 */
function _getApiKey_() {
  const key = PropertiesService.getScriptProperties().getProperty('GEMINI_API_KEY');
  if (!key) {
    throw new Error(
      'No se encontró GEMINI_API_KEY en Propiedades del script. ' +
        'Ve a Editor > Proyecto > Propiedades (Script) y crea la clave.',
    );
  }
  return key;
}

/**
 * Realiza la llamada con reintentos y procesa la respuesta con un parser.
 * @template T
 * @param {string} url
 * @param {GoogleAppsScript.URL_Fetch.URLFetchRequestOptions} options
 * @param {(json:any)=>T} parseFn
 * @returns {T}
 */
function _callGeminiWithRetries_(url, options, parseFn) {
  let lastErr = null;
  for (let attempt = 1; attempt <= CFG.MAX_RETRIES; attempt++) {
    try {
      const resp = UrlFetchApp.fetch(url, options);
      const code = resp.getResponseCode();
      const body = resp.getContentText();

      if (code >= 200 && code < 300) {
        const json = JSON.parse(body);
        return parseFn(json);
      }

      lastErr = new Error('HTTP ' + code + ': ' + body);
      Utilities.sleep(300 * attempt); // backoff lineal suave
    } catch (e) {
      lastErr = e;
      Utilities.sleep(300 * attempt);
    }
  }
  throw lastErr || new Error('Error desconocido al llamar a Gemini.');
}

/**
 * Extrae el texto “principal” desde la respuesta de Gemini.
 * @param {any} json
 * @returns {string}
 */
function _extractTextFromGemini_(json) {
  try {
    // Formato típico: { candidates: [{ content: { parts: [{ text: "..." }] } }] }
    const cand = json && json.candidates && json.candidates[0];
    const parts = cand && cand.content && cand.content.parts;
    if (Array.isArray(parts) && parts.length && parts[0].text) {
      return String(parts[0].text);
    }
  } catch (e) {
    // Ignora y retorna vacío
  }
  return '';
}

/** =========================
 * ====== Comandos extra ====
 * =========================*/

/**
 * (Opcional) Configura la API Key desde un prompt UI (una vez).
 * Luego revisa con: PropertiesService.getScriptProperties().getProperty('GEMINI_API_KEY')
 */
function configurarApiKey() {
  const ui = SpreadsheetApp.getUi();
  const res = ui.prompt('Configurar GEMINI_API_KEY', 'Pega tu API key:', ui.ButtonSet.OK_CANCEL);
  if (res.getSelectedButton() === ui.Button.OK) {
    PropertiesService.getScriptProperties().setProperty('GEMINI_API_KEY', res.getResponseText());
    ui.alert('GEMINI_API_KEY guardada.');
  }
}
