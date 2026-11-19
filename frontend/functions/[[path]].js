export async function onRequest({ request, next }) {
  const response = await next();
  if (new URL(request.url).pathname !== "/data/conversations.json") return response;
  const headers = new Headers(response.headers);
  headers.set("Cache-Control", "private, no-store");
  headers.set("Pragma", "no-cache");
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}
