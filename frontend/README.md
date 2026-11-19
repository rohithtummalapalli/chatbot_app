# Chatbot Analytics UI

React dashboard for the existing conversation export. The build sanitizes the export and copies it into the static app. Cloudflare Pages Functions provide password sign-in, semantic search, and Chat Summary APIs.

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
- Select the `feature` branch for this dashboard project.

Each build prepares a sanitized copy of `data/chats_new(19-11-2026).json`. Direct account/user identifiers are removed, user IDs are replaced with random per-export IDs for repeat-visit metrics, and contact form values are reduced to a submitted/not-submitted flag.

### Dashboard password

The whole dashboard uses an app-level shared password; Cloudflare Zero Trust / Access is not required. In the Pages project, open **Settings → Variables and Secrets** and set these as secrets for every environment you deploy (production and any previews):

- `DASHBOARD_ACCESS_PASSWORD` — a randomly generated password of at least 16 characters. Use a long password and share it only with dashboard users.
- `DASHBOARD_SESSION_SECRET` — a separate random secret of at least 32 characters, used to sign secure, HTTP-only, eight-hour session cookies.

For example, generate independent random values locally with `openssl rand -base64 32`. Do not commit these values or add them as `VITE_` variables. Without both secrets, sign-in is disabled. The dashboard UI's static files are public, but the conversation export and all API routes are blocked unless the visitor has a valid signed-in session. Sign out or wait eight hours to end a session.

### Enable multilingual semantic search

The search box uses Cloudflare Workers AI's multilingual BGE-M3 model through a Pages Function. In the Cloudflare dashboard, open this Pages project and go to **Settings → Bindings → Add → Workers AI**. Set the variable name to `AI`, save, then redeploy. Until the binding is enabled, the dashboard still supports exact text search; Semantic mode displays a setup message.

Workers AI has a free daily allocation; usage above that allocation requires a Workers Paid plan. Search sends conversation titles and user messages to the BGE-M3 model to rank across languages. See [Workers AI pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/) and [BGE-M3 model details](https://developers.cloudflare.com/workers-ai/models/bge-m3/).

### Chat Summary setup

Chat Summary analyses only `Chat Bubble/Iframe` conversations. It sends chat batches to your Azure OpenAI deployment using JSON mode, creates a report in the selected language, and saves the report JSON and metadata in D1. The source conversation text is not saved in D1. Azure credentials are read only by the Pages Function and are never sent to the browser.

1. Create a Cloudflare D1 database for the Pages project.
2. Run the schema in `frontend/migrations/0001_chat_summaries.sql` against that database. From the repository root, Wrangler can apply it with `npx wrangler d1 execute <database-name> --remote --file=frontend/migrations/0001_chat_summaries.sql`.
3. In the Pages project, open **Settings → Bindings**, add the D1 database with the binding name `DB`, then save.
4. Under **Settings → Variables and Secrets**, add the following for each deployed environment. Store `AZURE_OPENAI_API_KEY` as a secret; the other Azure settings are non-secret configuration values.
   - `AZURE_OPENAI_ENDPOINT` — Azure resource endpoint, for example `https://<resource>.openai.azure.com`
   - `AZURE_OPENAI_API_VERSION` — API version enabled for your Azure deployment
   - `AZURE_OPENAI_DEPLOYMENT_NAME` — the Azure deployment name passed as `model` in the Python SDK
   - `AZURE_OPENAI_API_KEY` — Azure key; secret, never a Vite/client variable
5. Redeploy the `feature` branch.

Chat Summary needs the Azure settings and D1 binding. The dashboard password gate works independently of D1, while saved summaries are unavailable until D1 is configured. Keep all credentials in Cloudflare Pages secrets and never prefix them with `VITE_`. See [Pages Function bindings](https://developers.cloudflare.com/pages/functions/bindings/) and [D1 limits](https://developers.cloudflare.com/d1/platform/limits/).
