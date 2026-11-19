import { clearSessionCookie, createSessionCookie, hasValidSession } from "../lib/session.js";

function response(body, status = 200, cookie = "") {
  const headers = new Headers({ "Content-Type": "application/json", "Cache-Control": "no-store", "Pragma": "no-cache" });
  if (cookie) headers.set("Set-Cookie", cookie);
  return new Response(JSON.stringify(body), { status, headers });
}

function sameOrigin(request) {
  const origin = request.headers.get("Origin");
  if (!origin) return false;
  try { return new URL(origin).origin === new URL(request.url).origin; }
  catch { return false; }
}

async function digest(value) {
  return new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)));
}

function constantTimeEqual(left, right) {
  if (left.length !== right.length) return false;
  let difference = 0;
  for (let index = 0; index < left.length; index += 1) difference |= left[index] ^ right[index];
  return difference === 0;
}

export async function onRequestGet({ request, env }) {
  if (String(env.DASHBOARD_SESSION_SECRET || "").length < 32 || String(env.DASHBOARD_ACCESS_PASSWORD || "").length < 16) {
    return response({ error: "Dashboard sign-in is not configured. Add the required Pages secrets." }, 503);
  }
  return response({ authenticated: await hasValidSession(request, env) });
}

export async function onRequestPost({ request, env }) {
  if (!sameOrigin(request)) return response({ error: "Invalid sign-in request." }, 403);
  if (String(env.DASHBOARD_SESSION_SECRET || "").length < 32 || String(env.DASHBOARD_ACCESS_PASSWORD || "").length < 16) {
    return response({ error: "Dashboard sign-in is not configured. Add the required Pages secrets." }, 503);
  }
  const raw = await request.text();
  if (raw.length > 2_000) return response({ error: "Invalid sign-in request." }, 413);
  let body;
  try { body = JSON.parse(raw); } catch { return response({ error: "Invalid sign-in request." }, 400); }
  if (body.action === "logout") return response({ authenticated: false }, 200, clearSessionCookie());
  if (body.action !== "login" || typeof body.password !== "string" || body.password.length > 500) return response({ error: "Enter the dashboard password." }, 400);
  const expected = await digest(String(env.DASHBOARD_ACCESS_PASSWORD));
  const submitted = await digest(body.password);
  if (!constantTimeEqual(expected, submitted)) return response({ error: "That password did not match." }, 401);
  try { return response({ authenticated: true }, 200, await createSessionCookie(env)); }
  catch { return response({ error: "Could not create a secure dashboard session." }, 503); }
}
