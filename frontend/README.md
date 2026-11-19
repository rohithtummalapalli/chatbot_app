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

Each build copies `data/chats_new(19-11-2026).json` into the static output. The deployed dashboard and export therefore use the same bundled dataset.
