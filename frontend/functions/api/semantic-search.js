const EMBEDDING_DIMENSIONS = 512;
const MAX_QUERY_LENGTH = 500;

function errorResponse(message, status = 400) {
  return Response.json({ error: message }, {
    status,
    headers: { "Cache-Control": "no-store", Pragma: "no-cache" },
  });
}

function base64ToFloat32(base64, expectedDimensions) {
  const binary = atob(base64);
  if (binary.length !== expectedDimensions * Float32Array.BYTES_PER_ELEMENT) return null;
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return new Float32Array(bytes.buffer);
}

async function createQueryEmbedding(query, env) {
  const endpoint = String(env.AZURE_OPENAI_ENDPOINT || "").trim().replace(/\/+$/, "");
  const apiKey = String(env.AZURE_OPENAI_API_KEY || "");
  const apiVersion = String(env.AZURE_OPENAI_API_VERSION || "").trim();
  const deployment = String(env.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME || "").trim();
  if (!endpoint || !apiKey || !apiVersion || !deployment) {
    throw new Error("Azure embeddings are not configured. Add the embedding deployment name and Azure credentials to Pages settings.");
  }

  let url;
  try {
    url = new URL(`${endpoint}/openai/deployments/${encodeURIComponent(deployment)}/embeddings?api-version=${encodeURIComponent(apiVersion)}`);
    if (url.protocol !== "https:") throw new Error("HTTPS required");
  } catch {
    throw new Error("Azure OpenAI endpoint configuration is invalid.");
  }

  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", "api-key": apiKey },
    body: JSON.stringify({ input: query, dimensions: EMBEDDING_DIMENSIONS }),
  });
  if (!response.ok) {
    throw new Error(`Azure embeddings returned HTTP ${response.status}. Check the embedding deployment, API version, and quota.`);
  }
  const result = await response.json();
  const embedding = result?.data?.[0]?.embedding;
  if (!Array.isArray(embedding) || embedding.length !== EMBEDDING_DIMENSIONS || !embedding.every(Number.isFinite)) {
    throw new Error("Azure embeddings returned a vector with an unexpected size.");
  }
  return embedding;
}

function cosineSimilarity(left, right) {
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < left.length; index += 1) {
    const a = left[index];
    const b = right[index];
    dot += a * b;
    leftNorm += a * a;
    rightNorm += b * b;
  }
  if (!leftNorm || !rightNorm) return -1;
  return dot / Math.sqrt(leftNorm * rightNorm);
}

export async function onRequestPost({ request, env }) {
  let body;
  try {
    body = await request.json();
  } catch {
    return errorResponse("Invalid search request.");
  }

  const query = typeof body?.query === "string" ? body.query.trim().slice(0, MAX_QUERY_LENGTH) : "";
  if (query.length < 2) return errorResponse("Enter at least two characters to search.");
  if (!env.DB) return errorResponse("Semantic search is not configured. Add the D1 binding named DB.", 503);

  try {
    const meta = await env.DB.prepare(
      "SELECT model, dimensions, conversation_count, chunk_count FROM semantic_index_meta WHERE id = 1",
    ).first();
    if (!meta) return errorResponse("The semantic index is not ready. Generate and load the conversation embeddings first.", 503);
    if (Number(meta.dimensions) !== EMBEDDING_DIMENSIONS) {
      return errorResponse("The semantic index uses a different vector size. Rebuild it with the current embedding settings.", 503);
    }
    const deployment = String(env.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME || "").trim();
    if (String(meta.model) !== deployment) {
      return errorResponse("The semantic index was generated with a different embedding deployment. Rebuild it with the configured deployment.", 503);
    }

    const embedding = await createQueryEmbedding(query, env);
    const { results = [] } = await env.DB.prepare(
      "SELECT conversation_id, vector FROM semantic_embeddings",
    ).all();
    if (!results.length) return errorResponse("The semantic index is empty. Generate and load the conversation embeddings first.", 503);

    const bestByConversation = new Map();
    for (const row of results) {
      const vector = base64ToFloat32(row.vector, EMBEDDING_DIMENSIONS);
      if (!vector) continue;
      const cosine = cosineSimilarity(embedding, vector);
      const previous = bestByConversation.get(row.conversation_id);
      if (previous == null || cosine > previous) bestByConversation.set(row.conversation_id, cosine);
    }

    const matches = [...bestByConversation.entries()]
      .map(([id, cosine]) => ({ id, score: Math.max(0, Math.min(1, (cosine + 1) / 2)) }))
      .sort((a, b) => b.score - a.score);
    return Response.json({
      matches,
      conversationsSearched: Number(meta.conversation_count),
      chunksSearched: results.length,
      model: meta.model,
    }, { headers: { "Cache-Control": "no-store", Pragma: "no-cache" } });
  } catch (error) {
    console.error("Azure semantic search failed", String(error?.message || error).slice(0, 240));
    return errorResponse(error?.message || "Semantic search could not reach Azure. Check the deployment configuration and try again.", 502);
  }
}
