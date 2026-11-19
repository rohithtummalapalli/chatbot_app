CREATE TABLE IF NOT EXISTS chat_summaries (
  id TEXT PRIMARY KEY,
  start_date TEXT NOT NULL,
  end_date TEXT NOT NULL,
  language TEXT NOT NULL,
  scope TEXT NOT NULL,
  prompt_version INTEGER NOT NULL,
  conversation_count INTEGER NOT NULL,
  low_confidence_count INTEGER NOT NULL,
  report_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (start_date, end_date, language, scope, prompt_version)
);

CREATE INDEX IF NOT EXISTS chat_summaries_created_at_idx ON chat_summaries(created_at DESC);
