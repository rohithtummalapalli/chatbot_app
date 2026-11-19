import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { randomUUID } from "node:crypto";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "../..");
const source = path.join(repoRoot, "data", "chats_new(19-11-2026).json");
const destinationDirectory = path.join(here, "../public/data");

await mkdir(destinationDirectory, { recursive: true });
const exported = JSON.parse(await readFile(source, "utf8"));
const userKeys = new Map();
const conversations = (exported.conversations || []).map((conversation) => {
  const originalUserKey = conversation.user_id || conversation.anonymous_id;
  let analyticsUserId = null;
  if (originalUserKey) {
    if (!userKeys.has(originalUserKey)) userKeys.set(originalUserKey, randomUUID());
    analyticsUserId = userKeys.get(originalUserKey);
  }
  const { account_id, user_id, anonymous_id, ...publicConversation } = conversation;
  return {
    ...publicConversation,
    analytics_user_id: analyticsUserId,
    form_submission: Boolean(conversation.form_submission && Object.keys(conversation.form_submission).length),
  };
});
await writeFile(
  path.join(destinationDirectory, "conversations.json"),
  JSON.stringify({ ...exported, conversations }),
  "utf8",
);
