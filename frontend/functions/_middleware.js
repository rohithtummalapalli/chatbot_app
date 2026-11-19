import { hasValidSession } from "./lib/session.js";

export async function onRequest({ request, env, next }) {
  const pathname = new URL(request.url).pathname;
  const needsSession = pathname.startsWith("/api/") && pathname !== "/api/session"
    || pathname === "/data/conversations.json";
  if (!needsSession || await hasValidSession(request, env)) return next();
  return Response.json({ error: "Sign in to access the dashboard." }, { status: 401, headers: { "Cache-Control": "no-store" } });
}
