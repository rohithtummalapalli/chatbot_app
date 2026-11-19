# Chatbot Analytics UI

React dashboard for the existing conversation export. The build copies the latest bundled export from `../data` into the static app, so it does not need a Python service at runtime.

## Run locally

```sh
npm install
npm run dev
```

## Deploy on Cloudflare Pages

- Connect this repository to a Pages project.
- Set the project root to `frontend`.
- Build command: `npm run build`
- Build output directory: `dist`
- Use the `feature` branch for the new dashboard deployment.
- Set the root directory to `frontend`.
- Set the build output directory to `dist`.

Each build prepares a sanitized copy of `data/chats_new(19-11-2026).json` in the static output. Direct account/user identifiers are removed, user IDs are replaced with random per-export IDs for repeat-visit metrics, and contact form values are reduced to a submitted/not-submitted flag.

### Enable multilingual semantic search

The search box uses Cloudflare Workers AI's multilingual BGE-M3 model through a Pages Function. In the Cloudflare dashboard, open this Pages project and go to **Settings → Bindings → Add → Workers AI**. Set the variable name to `AI`, save, then redeploy. Until the binding is enabled, the dashboard still supports exact text search; Semantic mode displays a setup message.

Workers AI has a free daily allocation; usage above that allocation requires a Workers Paid plan. Search sends conversation titles and user messages to the BGE-M3 model to rank across languages. See [Cloudflare Workers AI pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/) and [BGE-M3 model details](https://developers.cloudflare.com/workers-ai/models/bge-m3/).
