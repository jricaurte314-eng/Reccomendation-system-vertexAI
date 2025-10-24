/**
 * @file Generación de embeddings para habilidades del cliente.
 * Lee "Habilidades_Mejorada_Cliente" y escribe en "Embeddings cliente".
 * Requiere funciones externas: getGeminiApiKey(), batchEmbedContents_(), opcionalmente getSpreadsheet_().
 * @license MIT
 */

/** Debug global (true = log en Logger + hoja "Logs Embeddings Cliente") */
var DEBUG_EMB = true;

/** =========================
 * ======= Logging ==========
 * =========================*/

/**
 * Stringify seguro.
 * @param {any} x
 * @returns {string}
 */
function _safeStringify(x) {
  try { return (typeof x === 'string') ? x : JSON.stringify(x); }
  catch (e) { return String(x); }
}

/**
 * Log a consola y hoja dedicada (si existe Spreadsheet).
 * No detiene la ejecución si falla el log.
 * @param {string} msg
 * @param {any=} data
 */
function _logEmb(msg, data) {
  var line = '[' + new Date().toISOString() + '] ' + msg +
             (data !== undefined ? ' ' + _safeStringify(data) : '');
  if (DEBUG_EMB) Logger.log(line);
  try {
    var ss = (typeof getSpreadsheet_ === 'function')
      ? getSpreadsheet_()
      : SpreadsheetApp.getActiveSpreadsheet();
    var shName = 'Logs Embeddings Cliente';
    var sh = ss.getSheetByName(shName) || ss.insertSheet(shName);
    sh.appendRow([line]);
  } catch (e) {
    // Ignorar fallos de logging
  }
}

/**
 * Asserts con mensaje.
 * @param {boolean} cond
 * @param {string} msg
 */
function _assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

/** =========================
 * ====== Reintentos ========
 * =========================*/

function _sleep(ms) { Utilities.sleep(ms); }

/**
 * Envuelve batchEmbedContents_ con reintentos y backoff exponencial.
 * @param {string} apiKey
 * @param {string[]} texts
 * @param {number} outDim
 * @param {number} tries
 * @param {number} baseSleepMs
 * @returns {number[][]}
 */
function _tryEmbedWithRetry(apiKey, texts, outDim, tries, baseSleepMs) {
  var attempt = 0, lastErr;
  while (attempt < tries) {
    try {
      _logEmb('Llamando batchEmbedContents_', {
        attempt: attempt + 1, textos: texts.length, outDim: outDim,
      });
      var vecs = batchEmbedContents_(apiKey, texts, outDim);
      _assert(Array.isArray(vecs), 'batchEmbedContents_ no devolvió un arreglo.');
      _assert(vecs.length === texts.length, 'Cantidad de vectores != cantidad de textos.');
      return vecs;
    } catch (e) {
      lastErr = e;
      var wait = baseSleepMs * Math.pow(2, attempt);
      _logEmb('Error en batchEmbedContents_ (reintentando)', {
        attempt: attempt + 1, waitMs: wait, error: String(e),
      });
      _sleep(wait);
      attempt++;
    }
  }
  throw lastErr;
}

/** =========================
 * ====== Principal =========
 * =========================*/

/**
 * Genera embeddings desde "Habilidades_Mejorada_Cliente" y los escribe en "Embeddings cliente".
 * Dependencias requeridas:
 *   - getGeminiApiKey(): string
 *   - batchEmbedContents_(apiKey:string, texts:string[], outDim:number): number[][]
 * Dependencia opcional:
 *   - getSpreadsheet_(): Spreadsheet (si no, se usa SpreadsheetApp.getActiveSpreadsheet()).
 * @returns {string} resumen
 */
function generarEmbeddingsCliente() {
  var startedAt = new Date();
  _logEmb('=== INICIO generarEmbeddingsCliente ===');

  // 0) Verificaciones y defaults
  _assert(typeof getGeminiApiKey === 'function', 'Falta getGeminiApiKey().');
  if (typeof getSpreadsheet_ !== 'function') {
    _logEmb('Aviso: getSpreadsheet_() no existe; se usará SpreadsheetApp.getActiveSpreadsheet()');
  }
  if (typeof clearEmbClientIndex !== 'function') {
    _logEmb('Aviso: clearEmbClientIndex() no existe (continuará sin checkpoint).');
  }
  if (typeof setEmbClientIndex !== 'function') {
    _logEmb('Aviso: setEmbClientIndex() no existe (continuará sin checkpoint).');
  }

  var BATCH_LIMIT        = (typeof this.BATCH_LIMIT === 'number' && this.BATCH_LIMIT > 0)
    ? this.BATCH_LIMIT : 50;
  var DEFAULT_OUTPUT_DIM = (typeof this.DEFAULT_OUTPUT_DIM === 'number' && this.DEFAULT_OUTPUT_DIM > 0)
    ? this.DEFAULT_OUTPUT_DIM : 3072;
  var BATCH_SLEEP_MS     = (typeof this.BATCH_SLEEP_MS === 'number' && this.BATCH_SLEEP_MS >= 0)
    ? this.BATCH_SLEEP_MS : 400;

  _logEmb('Constantes', {
    BATCH_LIMIT: BATCH_LIMIT,
    DEFAULT_OUTPUT_DIM: DEFAULT_OUTPUT_DIM,
    BATCH_SLEEP_MS: BATCH_SLEEP_MS,
  });

  try {
    // 1) API key
    var apiKey = getGeminiApiKey();
    _assert(apiKey && typeof apiKey === 'string', 'API key inválida.');
    _logEmb('API key OK (oculta)');

    // 2) Spreadsheet
    var ss = (typeof getSpreadsheet_ === 'function')
      ? getSpreadsheet_()
      : SpreadsheetApp.getActiveSpreadsheet();
    _assert(ss && ss.getId, 'No se pudo obtener el Spreadsheet.');
    _logEmb('Spreadsheet OK', { id: ss.getId(), name: ss.getName() });

    // 3) Entrada
    var inSh = ss.getSheetByName('Habilidades_Mejorada_Cliente');
    _assert(inSh, 'No existe la hoja "Habilidades_Mejorada_Cliente".');

    var vals = inSh.getDataRange().getValues();
    _assert(vals.length >= 2, 'La hoja debe tener encabezado y al menos 1 fila de datos.');
    _logEmb('Lectura entrada OK', { rows: vals.length, cols: vals[0].length });

    // Detectar columnas por encabezado (tolerante a variantes)
    var headerRaw = vals[0];
    var header = headerRaw.map(function (c) { return String(c || '').toLowerCase().trim(); });
    _logEmb('Header detectado', headerRaw);

    var iName = header.findIndex(function (h) {
      return h.includes('habilidad (a)') || h === 'habilidad' || h.includes('competencia') || h.includes('habilidad');
    });
    var iDef = header.findIndex(function (h) {
      return h.includes('definición mejorada (ia)') ||
             h.includes('definicion mejorada (ia)') ||
             h === 'definicion' || h === 'definición' || h === 'definition';
    });
    _logEmb('Índices de columnas', { iName: iName, iDef: iDef });

    _assert(iName !== -1 && iDef !== -1,
      'Encabezados no detectados. Debe haber columnas "Habilidad (A)" y "Definición mejorada (IA)" en la fila 1.');

    // Filtrar filas válidas
    var rowsAll = vals.slice(1);
    var rows = [];
    for (var r = 0; r < rowsAll.length; r++) {
      var row = rowsAll[r];
      if (row[iName] && row[iDef]) rows.push(row);
    }
    _assert(rows.length > 0, 'No hay filas válidas con Habilidad y Definición.');
    _logEmb('Filas válidas', { valid: rows.length, total: rowsAll.length });

    // 4) Hoja salida (preparada pronto para facilitar debugging)
    var outName = 'Embeddings cliente';
    var outSh = ss.getSheetByName(outName) || ss.insertSheet(outName);
    _logEmb('Hoja salida preparada', { sheet: outName, sheetId: outSh.getSheetId() });

    // Limpieza y reinicio checkpoint
    outSh.clearContents();
    if (typeof clearEmbClientIndex === 'function') {
      try { clearEmbClientIndex(); } catch (e) { _logEmb('clearEmbClientIndex() falló', String(e)); }
    }

    // Header de salida
    var headerOut = ['Habilidad', 'Definición', 'Embedding (JSON)'];
    outSh.getRange(1, 1, 1, headerOut.length).setValues([headerOut]);
    outSh.setFrozenRows(1);
    outSh.getRange(1, 3, outSh.getMaxRows(), 1).setNumberFormat('@'); // col C como texto
    _logEmb('Header de salida escrito');

    // 5) Textos a embeber
    var textos = rows.map(function (rr) {
      return 'Habilidad: ' + rr[iName] + '. Definición: ' + rr[iDef];
    });
    _logEmb('Textos construidos', { count: textos.length, sample: textos[0] });

    // 6) Procesamiento por lotes con reintentos
    var processed = 0;
    for (var start = 0; start < textos.length; start += BATCH_LIMIT) {
      var end = Math.min(start + BATCH_LIMIT, textos.length);
      var chunkTexts = textos.slice(start, end);
      _logEmb('Procesando lote', { start: start, end: end, size: chunkTexts.length });

      var vectors = _tryEmbedWithRetry(apiKey, chunkTexts, DEFAULT_OUTPUT_DIM, /*tries*/ 3, /*baseSleepMs*/ 1500);

      // Construir filas salida
      var rowsOut = [];
      for (var i = 0; i < vectors.length; i++) {
        var orig = rows[start + i];
        var rowToWrite = [String(orig[iName]), String(orig[iDef]), _safeStringify(vectors[i])];
        rowsOut.push(rowToWrite);
      }

      // Escribir bloque
      if (rowsOut.length > 0) {
        outSh.getRange(2 + processed, 1, rowsOut.length, 3).setValues(rowsOut);
        outSh.getRange(2 + processed, 3, rowsOut.length, 1).setNumberFormat('@');
      }

      processed += vectors.length;
      if (typeof setEmbClientIndex === 'function') {
        try { setEmbClientIndex(processed); } catch (e) { _logEmb('setEmbClientIndex() falló', String(e)); }
      }
      _logEmb('Lote completado', { processed: processed });

      _sleep(BATCH_SLEEP_MS);
    }

    var elapsed = ((new Date()) - startedAt) / 1000;
    var msg = 'Embeddings cliente generados. Filas procesadas: ' + processed + ' (en ' + elapsed + 's)';
    _logEmb('=== FIN OK ===', { processed: processed, elapsed_s: elapsed });
    return msg;

  } catch (err) {
    _logEmb('*** ERROR FATAL ***', {
      message: String(err),
      stack: (err && err.stack) ? String(err.stack) : 'no stack',
    });
    throw err;
  }
}
