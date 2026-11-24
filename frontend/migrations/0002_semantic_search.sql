CREATE TABLE IF NOT EXISTS semantic_embeddings (
  conversation_id TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  vector TEXT NOT NULL,
  PRIMARY KEY (conversation_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS semantic_embeddings_conversation_idx
  ON semantic_embeddings(conversation_id);

CREATE TABLE IF NOT EXISTS semantic_index_meta (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  model TEXT NOT NULL,
  dimensions INTEGER NOT NULL,
  data_hash TEXT NOT NULL,
  conversation_count INTEGER NOT NULL,
  chunk_count INTEGER NOT NULL,
  indexed_at TEXT NOT NULL
);
