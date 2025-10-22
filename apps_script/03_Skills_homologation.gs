/**
 * @file Comparación de embeddings: cliente vs. catálogo (UBITS).
 * Escribe Top-K en "Catalogo_Habilidades" (formato largo).
 * Requiere hoja "Embeddings cliente" y hoja "Catalogo" con embeddings JSON en columna G (configurable).
 * @license MIT
 */

// ===== CONFIG =====
var TOP_K = 1000;                 // Top-N por habilidad cliente
var MIN_SIMILARITY = 0.65;        // Umbral mínimo (0.65 = 65%)
var CLIENT_EMB_SHEET = 'Embeddings cliente';
var UBITS_EMB_SHEET = 'Catalogo';
var UBITS_EMB_COL = 'G';          // Embeddings JSON en col G
var OUTPUT_SHEET = 'Catalogo_Habilidades';
var PERCENT_DECIMALS = 2;         // Decimales en porcentaje

// Lotes (anti-timeout)
var BATCH_SIZE = 1000;            // Filas cliente por ejecución del batch
var SCRIPT_PROP = PropertiesService.getScriptProperties();
var PROP_LAST_INDEX = 'lastIndex';

// ============ MODO NORMAL (todo de una) ============
/**
 * Ejecuta comparación completa de una vez y escribe resultados.
 * @returns {string}
 */
function compararClienteUBITS() {
  var ss = (typeof getSpreadsheet_ === 'function')
    ? getSpreadsheet_()
    : SpreadsheetApp.getActiveSpreadsheet();

  var clientSh = ss.getSheetByName(CLIENT_EMB_SHEET);
  var ubitsSh  = ss.getSheetByName(UBITS_EMB_SHEET);
  if (!clientSh) throw new Error('No se encontró la hoja "' + CLIENT_EMB_SHEET + '".');
  if (!ubitsSh)  throw new Error('No se encontró la hoja "' + UBITS_EMB_SHEET + '".');

  var client = readEmbeddingsSheet_(clientSh);                  // cliente (JSON en C o d1..dN)
  var ubits  = readEmbeddingsFromColumn_(ubitsSh, UBITS_EMB_COL); // UBITS (JSON en G)

  if (client.rows === 0) throw new Error('La hoja "' + CLIENT_EMB_SHEET + '" no tiene filas válidas.');
  if (ubits.rows  === 0) throw new Error('La hoja "' + UBITS_EMB_SHEET  + '" no tiene filas válidas (o columna ' + UBITS_EMB_COL + ' vacía).');

  // Alinear dimensiones
  var dimUsed = Math.min(client.dim, ubits.dim);
  if (client.dim !== ubits.dim) {
    client.vecs = client.vecs.map(function (v) { return v.slice(0, dimUsed); });
    ubits.vecs  = ubits.vecs.map(function (v) { return v.slice(0, dimUsed); });
  }

  // Precalcular normas de UBITS
  var ubitsNorms = ubits.vecs.map(L2norm_);
  var ubCount = ubits.vecs.length;

  // Hoja salida (formato largo)
  var outSh = ss.getSheetByName(OUTPUT_SHEET) || ss.insertSheet(OUTPUT_SHEET);
  outSh.clearContents();

  var headers = [
    'Habilidad cliente', 'Definición cliente',
    'Habilidad UBITS', 'Definición UBITS',
    'Similitud (%)', 'Ranking',
  ];
  outSh.getRange(1, 1, 1, headers.length).setValues([headers]);
  outSh.setFrozenRows(1);

  var results = [];

  for (var i = 0; i < client.rows; i++) {
    var cName = client.names[i];
    var cDef  = client.defs[i];
    var cVec  = client.vecs[i];
    var cNorm = L2norm_(cVec);

    // Similitudes contra UBITS
    var sims = new Array(ubCount);
    for (var j = 0; j < ubCount; j++) {
      var uVec  = ubits.vecs[j];
      var uNorm = ubitsNorms[j];
      var sim = (cNorm === 0 || uNorm === 0) ? 0 : (dot_(cVec, uVec) / (cNorm * uNorm));
      if (sim > 1) sim = 1;
      if (sim < -1) sim = -1;
      sims[j] = { name: ubits.names[j], def: ubits.defs[j], sim: sim };
    }

    // Ordenar y tomar Top-K con umbral
    sims.sort(function (a, b) { return b.sim - a.sim; });

    var picked = 0;
    for (var k = 0; k < sims.length && picked < TOP_K; k++) {
      var cand = sims[k];
      if (cand.sim >= MIN_SIMILARITY) {
        picked++;
        results.push([
          cName,
          cDef,
          (cand.name || '-'),
          (cand.def || '-'),
          (cand.sim * 100).toFixed(PERCENT_DECIMALS) + '%',
          picked,
        ]);
      }
    }
  }

  if (results.length) {
    outSh.getRange(2, 1, results.length, headers.length).setValues(results);
  }
  try { outSh.autoResizeColumns(1, headers.length); } catch (e) {}

  return 'Top-' + TOP_K + ' generado. Filas escritas: ' + results.length +
         '. Dimensión usada: ' + dimUsed + '. Umbral: ' + (MIN_SIMILARITY * 100).toFixed(0) + '%.';
}

// ============ MODO BATCH (anti-timeout) ============
/**
 * Ejecuta comparación por lotes, usando triggers cada minuto.
 * Estado en ScriptProperties (clave PROP_LAST_INDEX).
 */
function compararClienteUBITS_batch() {
  var ss = (typeof getSpreadsheet_ === 'function')
    ? getSpreadsheet_()
    : SpreadsheetApp.getActiveSpreadsheet();

  var clientSh = ss.getSheetByName(CLIENT_EMB_SHEET);
  var ubitsSh  = ss.getSheetByName(UBITS_EMB_SHEET);
  var outSh    = ss.getSheetByName(OUTPUT_SHEET) || ss.insertSheet(OUTPUT_SHEET);

  if (!clientSh) throw new Error('No se encontró la hoja "' + CLIENT_EMB_SHEET + '".');
  if (!ubitsSh)  throw new Error('No se encontró la hoja "' + UBITS_EMB_SHEET + '".');

  var client = readEmbeddingsSheet_(clientSh);
  var ubits  = readEmbeddingsFromColumn_(ubitsSh, UBITS_EMB_COL);

  if (client.rows === 0) throw new Error('La hoja "' + CLIENT_EMB_SHEET + '" no tiene filas válidas.');
  if (ubits.rows  === 0) throw new Error('La hoja "' + UBITS_EMB_SHEET + '" no tiene filas válidas (o columna ' + UBITS_EMB_COL + ' vacía).');

  var dimUsed = Math.min(client.dim, ubits.dim);
  if (client.dim !== ubits.dim) {
    client.vecs = client.vecs.map(function (v) { return v.slice(0, dimUsed); });
    ubits.vecs  = ubits.vecs.map(function (v) { return v.slice(0, dimUsed); });
  }

  var ubitsNorms = ubits.vecs.map(L2norm_);

  var start = Number(SCRIPT_PROP.getProperty(PROP_LAST_INDEX)) || 0;
  var end   = Math.min(start + BATCH_SIZE, client.rows);

  if (start === 0) {
    outSh.clearContents();
    var headers = [
      'Habilidad cliente', 'Definición cliente',
      'Habilidad UBITS', 'Definición UBITS',
      'Similitud (%)', 'Ranking',
    ];
    outSh.getRange(1, 1, 1, headers.length).setValues([headers]);
    outSh.setFrozenRows(1);
  }

  var results = [];

  for (var i = start; i < end; i++) {
    var cName = client.names[i];
    var cDef  = client.defs[i];
    var cVec  = client.vecs[i];
    var cNorm = L2norm_(cVec);

    var sims = new Array(ubits.vecs.length);
    for (var j = 0; j < ubits.vecs.length; j++) {
      var uVec  = ubits.vecs[j];
      var uNorm = ubitsNorms[j];
      var sim = (cNorm === 0 || uNorm === 0) ? 0 : (dot_(cVec, uVec) / (cNorm * uNorm));
      if (sim > 1) sim = 1;
      if (sim < -1) sim = -1;
      sims[j] = { name: ubits.names[j], def: ubits.defs[j], sim: sim };
    }

    sims.sort(function (a, b) { return b.sim - a.sim; });

    var picked = 0;
    for (var k = 0; k < sims.length && picked < TOP_K; k++) {
      var cand = sims[k];
      if (cand.sim >= MIN_SIMILARITY) {
        picked++;
        results.push([
          cName,
          cDef,
          (cand.name || '-'),
          (cand.def || '-'),
          (cand.sim * 100).toFixed(PERCENT_DECIMALS) + '%',
          picked,
        ]);
      }
    }
  }

  if (results.length > 0) {
    var destRow = outSh.getLastRow() + 1;
    outSh.getRange(destRow, 1, results.length, results[0].length).setValues(results);
  }

  if (end >= client.rows) {
    SCRIPT_PROP.deleteProperty(PROP_LAST_INDEX);
    stopBatchProcessIfDone_(end, client.rows);
    Logger.log('✅ Proceso completado. Filas cliente: ' + client.rows + '. Dimensión usada: ' + dimUsed + '.');
  } else {
    SCRIPT_PROP.setProperty(PROP_LAST_INDEX, end.toString());
    Logger.log('⏳ Avance hasta fila ' + end + ' de ' + client.rows + '. Dimensión usada: ' + dimUsed + '.');
  }
}

// ===== Triggers de batch =====
function startBatchProcess() {
  deleteBatchTriggers_();
  SCRIPT_PROP.deleteProperty(PROP_LAST_INDEX);

  ScriptApp.newTrigger('compararClienteUBITS_batch')
    .timeBased()
    .everyMinutes(1) // ajusta a 5/10 si quieres
    .create();

  Logger.log('⏳ Trigger creado: correrá cada minuto hasta terminar.');
}

function deleteBatchTriggers_() {
  var triggers = ScriptApp.getProjectTriggers();
  for (var i = 0; i < triggers.length; i++) {
    if (triggers[i].getHandlerFunction() === 'compararClienteUBITS_batch') {
      ScriptApp.deleteTrigger(triggers[i]);
    }
  }
  Logger.log('🧹 Triggers de batch eliminados.');
}

function stopBatchProcessIfDone_(end, total) {
  if (end >= total) {
    deleteBatchTriggers_();
    Logger.log('✅ Proceso completado. Trigger eliminado.');
  }
}

// ===== Helpers de lectura/álgebra =====

/**
 * Hoja de cliente: soporta
 *  (a) JSON en col C → [Habilidad, Definición, "[...vector...]"]
 *  (b) columnas d1..dN → [Habilidad, Definición, d1, d2, ..., dN]
 * @param {GoogleAppsScript.Spreadsheet.Sheet} sheet
 * @returns {{rows:number, dim:number, names:string[], defs:string[], vecs:number[][]}}
 */
function readEmbeddingsSheet_(sheet) {
  var values = sheet.getDataRange().getValues();
  if (!values || values.length < 2) {
    return { rows: 0, dim: 0, names: [], defs: [], vecs: [] };
  }

  var body = values.slice(1).filter(function (r) { return r[0] && r[1]; });
  var names = new Array(body.length);
  var defs  = new Array(body.length);
  var vecs  = new Array(body.length);
  var dim   = 0;

  // ¿JSON en C?
  var looksJson = false;
  if (values[0].length >= 3 && body.length > 0) {
    var c0 = body[0][2];
    looksJson = (typeof c0 === 'string' && String(c0).trim().charAt(0) === '[') || Array.isArray(c0);
  }

  if (looksJson) {
    for (var i = 0; i < body.length; i++) {
      names[i] = String(body[i][0]);
      defs[i]  = String(body[i][1]);

      var raw = body[i][2];
      var arr;
      if (Array.isArray(raw)) {
        arr = raw;
      } else {
        try { arr = JSON.parse(String(raw)); } catch (e) { arr = []; }
      }

      var v = new Array(arr.length);
      for (var d = 0; d < arr.length; d++) {
        var num = Number(arr[d]);
        v[d] = isNaN(num) ? 0 : num;
      }
      vecs[i] = v;
      if (v.length > dim) dim = v.length;
    }
  } else {
    // Formato d1..dN
    var numCols = values[0].length;
    dim = Math.max(0, numCols - 2);
    for (var j = 0; j < body.length; j++) {
      names[j] = String(body[j][0]);
      defs[j]  = String(body[j][1]);
      var v2 = new Array(dim);
      for (var dd = 0; dd < dim; dd++) {
        var val = body[j][2 + dd];
        var num2 = (typeof val === 'number') ? val : parseFloat(String(val).replace(',', '.'));
        v2[dd] = isNaN(num2) ? 0 : num2;
      }
      vecs[j] = v2;
    }
  }

  return { rows: body.length, dim: dim, names: names, defs: defs, vecs: vecs };
}

/**
 * Lee embeddings desde una columna fija (letra), asumiendo:
 *  A: Habilidad, B: Definición, <col> : JSON del vector "[...]"
 * @param {GoogleAppsScript.Spreadsheet.Sheet} sheet
 * @param {string} colLetter
 * @returns {{rows:number, dim:number, names:string[], defs:string[], vecs:number[][]}}
 */
function readEmbeddingsFromColumn_(sheet, colLetter) {
  var values = sheet.getDataRange().getValues();
  if (!values || values.length < 2) {
    return { rows: 0, dim: 0, names: [], defs: [], vecs: [] };
  }

  var embIdx = letterToIndex_(colLetter); // 0-based
  var body = values.slice(1).filter(function (r) { return r[0] && r[1] && r[embIdx]; });

  var names = new Array(body.length);
  var defs  = new Array(body.length);
  var vecs  = new Array(body.length);
  var dim   = 0;

  for (var i = 0; i < body.length; i++) {
    names[i] = String(body[i][0]);
    defs[i]  = String(body[i][1]);

    var raw = body[i][embIdx];
    var arr;
    if (Array.isArray(raw)) {
      arr = raw;
    } else {
      try { arr = JSON.parse(String(raw)); } catch (e) { arr = []; }
    }

    var v = new Array(arr.length);
    for (var d = 0; d < arr.length; d++) {
      var num = Number(arr[d]);
      v[d] = isNaN(num) ? 0 : num;
    }
    vecs[i] = v;
    if (v.length > dim) dim = v.length;
  }

  return { rows: body.length, dim: dim, names: names, defs: defs, vecs: vecs };
}

/**
 * Convierte letra de columna (e.g., "G") a índice 0-based.
 * @param {string} letter
 * @returns {number}
 */
function letterToIndex_(letter) {
  letter = String(letter || '').toUpperCase().trim();
  if (!/^[A-Z]+$/.test(letter)) throw new Error('Columna inválida: ' + letter);
  var n = 0;
  for (var i = 0; i < letter.length; i++) n = n * 26 + (letter.charCodeAt(i) - 64);
  return n - 1;
}

/**
 * Producto punto (maneja tamaños distintos tomando el mínimo).
 * @param {number[]} a
 * @param {number[]} b
 * @returns {number}
 */
function dot_(a, b) {
  var s = 0;
  var len = Math.min(a.length, b.length);
  for (var i = 0; i < len; i++) {
    var ai = Number(a[i]) || 0;
    var bi = Number(b[i]) || 0;
    s += ai * bi;
  }
  return s;
}

/**
 * Norma L2 de un vector.
 * @param {number[]} v
 * @returns {number}
 */
function L2norm_(v) {
  var s = 0;
  for (var i = 0; i < v.length; i++) {
    var x = Number(v[i]) || 0;
    s += x * x;
  }
  return Math.sqrt(s);
}

// ===== Menú opcional =====
/*
function onOpen() {
  var ui = SpreadsheetApp.getUi();
  ui.createMenu('Catálogo Embeddings')
    .addItem('Iniciar batch', 'startBatchProcess')
    .addItem('Detener batch', 'deleteBatchTriggers_')
    .addItem('Ejecutar una vez (normal)', 'compararClienteUBITS')
    .addToUi();
}
*/
