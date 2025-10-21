// Reranker.gs
// Re-ordenamiento de candidatos usando modelos cross-encoder externos.
// Soportados: Cohere Rerank v3.5+ y Jina Reranker v2 Multilingüe.
// Uso esperado: pasar un query (string) y una lista de candidatos (strings).
// Devuelve: array de objetos { index, score } ordenados por score desc.

function rerankCandidates(query, docs, options) {
  options = options || {};
  const provider = (options.provider || getPreferredReranker_()).toUpperCase(); // 'COHERE' | 'JINA'
  const topN = Math.max(1, Math.min(Number(options.topN || docs.length), docs.length));

  if (!Array.isArray(docs) || docs.length === 0) return [];

  if (provider === 'COHERE') {
    const key = getCohereApiKey();
    if (!key) {
      // Si no hay clave para Cohere, intenta Jina
      const alt = getJinaApiKey();
      if (!alt) return fallbackIdentity_(docs.length);
      return callJinaRerank_(alt, query, docs, topN);
    }
    return callCohereRerank_(key, query, docs, topN);
  } else {
    const key = getJinaApiKey();
    if (!key) {
      const alt = getCohereApiKey();
      if (!alt) return fallbackIdentity_(docs.length);
      return callCohereRerank_(alt, query, docs, topN);
    }
    return callJinaRerank_(key, query, docs, topN);
  }
}

// Preferencia simple: si hay Cohere, usarlo; si no, Jina; si no, fallback.
function getPreferredReranker_() {
  return getCohereApiKey() ? 'COHERE' : (getJinaApiKey() ? 'JINA' : 'NONE');
}

// --- Cohere Rerank ---
// Docs: https://docs.cohere.com/reference/rerank
function callCohereRerank_(apiKey, query, docs, topN) {
  const url = 'https://api.cohere.com/v1/rerank';
  const headers = {
    'Authorization': 'Bearer ' + apiKey,
    'Content-Type': 'application/json'
  };
  const payload = {
    model: 'rerank-english-v3.0', // o 'rerank-multilingual-v3.0' / 'rerank-v3.5' si tu cuenta lo tiene
    query: query,
    documents: docs.map(t => ({ text: String(t || '') })),
    top_n: topN,
    return_documents: false
  };

  const json = httpJson_(url, { headers, payload, retries: 3, timeoutMs: 60000 });
  // Respuesta: { results: [{index, relevance_score, ...}, ...] }
  if (!json || !json.results) return fallbackIdentity_(docs.length);

  // Ordena por score descendente
  const ordered = json.results.slice().sort((a,b)=> (b.relevance_score - a.relevance_score));
  return ordered.map(r => ({ index: r.index, score: r.relevance_score }));
}

// --- Jina Reranker ---
// Docs: https://jina.ai/reranker/
function callJinaRerank_(apiKey, query, docs, topN) {
  const url = 'https://api.jina.ai/v1/reranker';
  const headers = {
    'Authorization': 'Bearer ' + apiKey,
    'Content-Type': 'application/json'
  };
  const payload = {
    model: 'jina-reranker-v2-base-multilingual',
    query: query,
    documents: docs,
    top_n: topN
  };

  const json = httpJson_(url, { headers, payload, retries: 3, timeoutMs: 60000 });
  // Esperado: { data: [{ index, score }...] } (la forma real puede variar; normalizamos)
  if (!json) return fallbackIdentity_(docs.length);

  let items = [];
  if (Array.isArray(json.data)) {
    items = json.data;
  } else if (Array.isArray(json.results)) {
    items = json.results;
  } else if (Array.isArray(json.reranked)) {
    items = json.reranked;
  }
  if (!items.length) return fallbackIdentity_(docs.length);

  const ordered = items.slice().sort((a,b)=> (Number(b.score||0) - Number(a.score||0)));
  return ordered.map(r => ({ index: Number(r.index), score: Number(r.score) }));
}

function fallbackIdentity_(n) {
  const arr = [];
  for (let i=0;i<n;i++) arr.push({ index: i, score: 0 });
  return arr;
}
