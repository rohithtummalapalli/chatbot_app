import { copyFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "../..");
const source = path.join(repoRoot, "data", "chats_new(19-11-2026).json");
const destinationDirectory = path.join(here, "../public/data");

await mkdir(destinationDirectory, { recursive: true });
await copyFile(source, path.join(destinationDirectory, "conversations.json"));
