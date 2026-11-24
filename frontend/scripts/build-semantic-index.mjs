import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const frontendRoot = path.resolve(here, "..");
const inputPath = path.join(frontendRoot, "public", "data", "conversations.json");
const outputPath = path.join(frontendRoot, "semantic-index.sql");
const model = process.env.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME || "text-embedding-3-small";
const dimensions = 512;
const chunkSize = 3_500;
const chunkOverlap = 400;
const batchSize = 32;
const rowsPerInsert = 20;

const endpoint = String(process.env.AZURE_OPENAI_ENDPOINT || "").trim().replace(/\/+$/, "");
const apiKey = String(process.env.AZURE_OPENAI_API_KEY || "");
const apiVersion = String(process.env.AZURE_OPENAI_API_VERSION || "").trim();
if (!endpoint || !apiKey || !apiVersion) {
  throw new Error("Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_API_VERSION in this terminal before indexing.");
}

const exported = JSON.parse(await readFile(inputPath, "utf8"));
const conversations = Array.isArray(exported.conversations) ? exported.conversations : [];
if (!conversations.length) throw new Error(`No conversations found in ${inputPath}. Run npm run build first.`);
const { conversations: ignoredConversations, ...datasetMetadata } = exported;

function splitText(text) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    let end = Math.min(start + chunkSize, text.length);
    if (end < text.length && /[\uD800-\uDBFF]/.test(text[end - 1]) && /[\uDC00-\uDFFF]/.test(text[end])) end += 1;
    const chunk = text.slice(start, end).trim();
    if (chunk) chunks.push(chunk);
    if (end >= text.length) break;
    start = end - chunkOverlap;
    if (start > 0 && /[\uD800-\uDBFF]/.test(text[start - 1]) && /[\uDC00-\uDFFF]/.test(text[start])) start -= 1;
  }
  return chunks;
}

const chunks = [];
for (const conversation of conversations) {
  const id = String(conversation?.id || "");
  if (!id) continue;
  const searchableDocument = JSON.stringify({ dataset: datasetMetadata, conversation });
  splitText(searchableDocument).forEach((text, chunkIndex) => chunks.push({ id, chunkIndex, text }));
}
if (!chunks.length) throw new Error("No searchable conversation text was found.");

const deploymentUrl = new URL(
  `${endpoint}/openai/deployments/${encodeURIComponent(model)}/embeddings?api-version=${encodeURIComponent(apiVersion)}`,
);
if (deploymentUrl.protocol !== "https:") throw new Error("AZURE_OPENAI_ENDPOINT must use HTTPS.");

async function embedBatch(batch, attempt = 0) {
  const response = await fetch(deploymentUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json", "api-key": apiKey },
    body: JSON.stringify({ input: batch.map((item) => item.text), dimensions }),
  });
  if ((response.status === 429 || response.status >= 500) && attempt < 4) {
    const retryAfter = Number(response.headers.get("retry-after"));
    const delayMs = Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter * 1000 : 1_000 * (2 ** attempt);
    await new Promise((resolve) => setTimeout(resolve, Math.min(delayMs, 20_000)));
    return embedBatch(batch, attempt + 1);
  }
  if (!response.ok) {
    throw new Error(`Azure embeddings returned HTTP ${response.status} for a batch. Check the deployment name, API version, and quota.`);
  }
  const result = await response.json();
  const vectors = result?.data;
  if (!Array.isArray(vectors) || vectors.length !== batch.length) throw new Error("Azure returned an unexpected number of vectors.");
  for (const item of vectors) {
    if (!Array.isArray(item.embedding) || item.embedding.length !== dimensions || !item.embedding.every(Number.isFinite)) {
      throw new Error("Azure returned a vector with an unexpected size. Confirm the deployment supports the dimensions parameter.");
    }
  }
  return { vectors, tokens: Number(result?.usage?.total_tokens || 0) };
}

function floatVectorBase64(values) {
  const bytes = new ArrayBuffer(values.length * Float32Array.BYTES_PER_ELEMENT);
  const view = new DataView(bytes);
  values.forEach((value, index) => view.setFloat32(index * 4, value, true));
  return Buffer.from(bytes).toString("base64");
}

function sqlString(value) {
  return `'${String(value).replaceAll("'", "''")}'`;
}

const vectorRows = [];
let totalTokens = 0;
for (let offset = 0; offset < chunks.length; offset += batchSize) {
  const batch = chunks.slice(offset, offset + batchSize);
  const { vectors, tokens } = await embedBatch(batch);
  totalTokens += tokens;
  for (const item of vectors) {
    const source = batch[item.index];
    vectorRows.push(`(${sqlString(source.id)}, ${source.chunkIndex}, ${sqlString(floatVectorBase64(item.embedding))})`);
  }
  console.log(`Embedded ${Math.min(offset + batch.length, chunks.length)} / ${chunks.length} chunks`);
}

const dataHash = createHash("sha256")
  .update(JSON.stringify({ data: exported, model, dimensions, chunkSize, chunkOverlap }))
  .digest("hex");
const sql = [
  "-- Generated locally by npm run semantic:index. Contains vectors only; do not commit or share this file.",
  "DELETE FROM semantic_embeddings;",
  "DELETE FROM semantic_index_meta;",
];
for (let offset = 0; offset < vectorRows.length; offset += rowsPerInsert) {
  sql.push(`INSERT INTO semantic_embeddings (conversation_id, chunk_index, vector) VALUES\n${vectorRows.slice(offset, offset + rowsPerInsert).join(",\n")};`);
}
sql.push(
  `INSERT INTO semantic_index_meta (id, model, dimensions, data_hash, conversation_count, chunk_count, indexed_at) VALUES (1, ${sqlString(model)}, ${dimensions}, ${sqlString(dataHash)}, ${conversations.length}, ${chunks.length}, ${sqlString(new Date().toISOString())});`,
  "",
);
await writeFile(outputPath, sql.join("\n"), { encoding: "utf8", mode: 0o600 });
console.log(`Created ${path.relative(frontendRoot, outputPath)} with ${vectorRows.length} vectors for ${conversations.length} conversations.`);
console.log(`Azure reported ${totalTokens.toLocaleString()} input tokens; the estimate uses ${dimensions} dimensions.`);
console.log("Load this file into the chatbot D1 database with Wrangler. Keep the generated SQL file private and delete it after loading.");
