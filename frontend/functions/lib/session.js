const SESSION_LIFETIME_SECONDS = 8 * 60 * 60;
const COOKIE_NAME = "chatbot_dashboard_session";

function encodeBase64Url(bytes) {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
}

function decodeBase64Url(value) {
  const base64 = value.replaceAll("-", "+").replaceAll("_", "/");
  const binary = atob(base64.padEnd(Math.ceil(base64.length / 4) * 4, "="));
  return Uint8Array.from(binary, (character) => character.charCodeAt(0));
}

function getCookie(request) {
  const cookie = request.headers.get("Cookie") || "";
  const entry = cookie.split(";").map((part) => part.trim()).find((part) => part.startsWith(`${COOKIE_NAME}=`));
  return entry ? entry.slice(COOKIE_NAME.length + 1) : "";
}

async function signingKey(env) {
  const secret = String(env.DASHBOARD_SESSION_SECRET || "");
  if (secret.length < 32) return null;
  return crypto.subtle.importKey("raw", new TextEncoder().encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign", "verify"]);
}

export async function createSessionCookie(env) {
  const key = await signingKey(env);
  if (!key) throw new Error("Dashboard session signing secret is not configured.");
  const issuedAt = Math.floor(Date.now() / 1000);
  const payload = encodeBase64Url(new TextEncoder().encode(JSON.stringify({ iat: issuedAt, exp: issuedAt + SESSION_LIFETIME_SECONDS })));
  const signature = new Uint8Array(await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(payload)));
  return `${COOKIE_NAME}=${payload}.${encodeBase64Url(signature)}; Path=/; Max-Age=${SESSION_LIFETIME_SECONDS}; HttpOnly; Secure; SameSite=Strict`;
}

export function clearSessionCookie() {
  return `${COOKIE_NAME}=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Strict`;
}

export async function hasValidSession(request, env) {
  const token = getCookie(request);
  const [payloadPart, signaturePart, extra] = token.split(".");
  if (!payloadPart || !signaturePart || extra) return false;
  try {
    const payload = JSON.parse(new TextDecoder().decode(decodeBase64Url(payloadPart)));
    const now = Math.floor(Date.now() / 1000);
    if (!Number.isFinite(payload.iat) || !Number.isFinite(payload.exp) || payload.exp <= now || payload.iat > now + 60 || payload.exp - payload.iat > SESSION_LIFETIME_SECONDS) return false;
    const key = await signingKey(env);
    if (!key) return false;
    return crypto.subtle.verify("HMAC", key, decodeBase64Url(signaturePart), new TextEncoder().encode(payloadPart));
  } catch {
    return false;
  }
}
