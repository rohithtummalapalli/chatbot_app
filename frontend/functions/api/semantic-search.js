const MODEL = "@cf/baai/bge-m3";
const MAX_REQUEST_BYTES = 300_000;
const MAX_CONTEXTS = 1_200;
const MAX_CONTEXT_LENGTH = 400;
const MAX_CONTEXT_TOTAL = 180_000;

export async function onRequestPost({ request, env }) {
  if (!env.AI) {
    return Response.json(
      { error: "Multilingual search is not configured. Add a Workers AI binding named AI to this Pages project." },
      { status: 503 },
    );
  }

  const rawBody = await request.text();
  if (rawBody.length > MAX_REQUEST_BYTES) {
    return Response.json({ error: "The search request is too large." }, { status: 413 });
  }

  let body;
  try {
    body = JSON.parse(rawBody);
  } catch {
    return Response.json({ error: "Invalid search request." }, { status: 400 });
  }

  const query = typeof body.query === "string" ? body.query.trim().slice(0, 200) : "";
  const documents = Array.isArray(body.documents) ? body.documents : [];
  if (query.length < 2 || !documents.length || documents.length > MAX_CONTEXTS) {
    return Response.json({ error: "Enter a search phrase and try again." }, { status: 400 });
  }

  let totalCharacters = 0;
  const ids = [];
  const contexts = [];
  for (const document of documents) {
    if (typeof document?.id !== "string" || typeof document?.text !== "string") continue;
    const context = document.text.trim().slice(0, MAX_CONTEXT_LENGTH);
    if (!context) continue;
    totalCharacters += context.length;
    if (totalCharacters > MAX_CONTEXT_TOTAL) {
      return Response.json({ error: "The conversation export is too large for semantic search." }, { status: 413 });
    }
    ids.push(document.id);
    contexts.push({ text: context });
  }

  if (!contexts.length) return Response.json({ matches: [] });

  try {
    const result = await env.AI.run(MODEL, {
      query,
      contexts,
      truncate_inputs: true,
    });
    const matches = (result?.response || [])
      .filter((item) => Number.isInteger(item.id) && Number.isFinite(item.score) && ids[item.id])
      .map((item) => ({ id: ids[item.id], score: item.score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 100);
    return Response.json({ matches });
  } catch (error) {
    console.error("Workers AI semantic search failed", error);
    const details = String(error?.message || error || "Unknown Workers AI error")
      .replace(/\s+/g, " ")
      .slice(0, 240);
    return Response.json({
      error: "Multilingual search could not reach the language model. Try again shortly.",
      details: `Cloudflare AI: ${details}`,
    }, { status: 502 });
  }
}
