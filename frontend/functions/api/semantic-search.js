const MODEL = "@cf/baai/bge-m3";
const MAX_REQUEST_BYTES = 12_000_000;
const MAX_DOCUMENTS = 10_000;
const MAX_CONTEXT_LENGTH = 400;
const CONTEXT_OVERLAP = 80;
const MAX_CONTEXTS_PER_BATCH = 1_200;
const MAX_CONTEXT_TOTAL = 150_000;
const AI_CONCURRENCY = 4;

function splitDocument(document) {
  const text = document.text;
  const contexts = [];
  let start = 0;
  while (start < text.length) {
    let end = Math.min(start + MAX_CONTEXT_LENGTH, text.length);
    if (
      end < text.length
      && text.charCodeAt(end - 1) >= 0xd800
      && text.charCodeAt(end - 1) <= 0xdbff
      && text.charCodeAt(end) >= 0xdc00
      && text.charCodeAt(end) <= 0xdfff
    ) end += 1;
    const context = text.slice(start, end).trim();
    if (context) contexts.push({ id: document.id, text: context });
    if (end === text.length) break;
    start = end - CONTEXT_OVERLAP;
    if (
      start > 0
      && text.charCodeAt(start - 1) >= 0xd800
      && text.charCodeAt(start - 1) <= 0xdbff
      && text.charCodeAt(start) >= 0xdc00
      && text.charCodeAt(start) <= 0xdfff
    ) start -= 1;
  }
  return contexts;
}

function makeBatches(documents) {
  const batches = [];
  let batch = [];
  let batchCharacters = 0;

  function flush() {
    if (batch.length) batches.push(batch);
    batch = [];
    batchCharacters = 0;
  }

  for (const document of documents) {
    for (const context of splitDocument(document)) {
      if (
        batch.length >= MAX_CONTEXTS_PER_BATCH
        || batchCharacters + context.text.length > MAX_CONTEXT_TOTAL
      ) flush();
      batch.push(context);
      batchCharacters += context.text.length;
    }
  }
  flush();
  return batches;
}

export async function onRequestPost({ request, env }) {
  if (!env.AI) {
    return Response.json(
      { error: "Multilingual search is not configured. Add a Workers AI binding named AI to this Pages project." },
      { status: 503 },
    );
  }

  const rawBody = await request.text();
  if (new TextEncoder().encode(rawBody).byteLength > MAX_REQUEST_BYTES) {
    return Response.json({ error: "The conversation data is too large for one search request." }, { status: 413 });
  }

  let body;
  try {
    body = JSON.parse(rawBody);
  } catch {
    return Response.json({ error: "Invalid search request." }, { status: 400 });
  }

  const query = typeof body.query === "string" ? body.query.trim().slice(0, 200) : "";
  const documents = Array.isArray(body.documents) ? body.documents : [];
  if (query.length < 2 || !documents.length || documents.length > MAX_DOCUMENTS) {
    return Response.json({ error: "Enter a search phrase and try again." }, { status: 400 });
  }

  const validDocuments = documents.filter(
    (document) => typeof document?.id === "string" && typeof document?.text === "string" && document.text.length,
  );
  if (!validDocuments.length) return Response.json({ matches: [] });

  const batches = makeBatches(validDocuments);
  const bestScoreById = new Map();

  try {
    for (let offset = 0; offset < batches.length; offset += AI_CONCURRENCY) {
      const currentBatches = batches.slice(offset, offset + AI_CONCURRENCY);
      const results = await Promise.all(currentBatches.map((batch) => env.AI.run(MODEL, {
        query,
        contexts: batch.map((context) => ({ text: context.text })),
        truncate_inputs: true,
      })));

      results.forEach((result, batchIndex) => {
        const batch = currentBatches[batchIndex];
        for (const item of result?.response || []) {
          if (!Number.isInteger(item.id) || !Number.isFinite(item.score) || !batch[item.id]) continue;
          const id = batch[item.id].id;
          const previous = bestScoreById.get(id);
          if (previous == null || item.score > previous) bestScoreById.set(id, item.score);
        }
      });
    }

    const matches = [...bestScoreById.entries()]
      .map(([id, score]) => ({ id, score }))
      .sort((a, b) => b.score - a.score);
    return Response.json({ matches, conversationsSearched: validDocuments.length, batchesSearched: batches.length });
  } catch (error) {
    console.error("Workers AI semantic search failed", error);
    const details = String(error?.message || error || "Unknown Workers AI error")
      .replace(/\s+/g, " ")
      .slice(0, 240);
    return Response.json({
      error: "Multilingual search could not complete across all conversation data. Try again shortly.",
      details: `Cloudflare AI: ${details}`,
    }, { status: 502 });
  }
}
