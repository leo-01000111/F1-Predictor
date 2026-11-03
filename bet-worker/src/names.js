// Shared by the Worker and scripts/seed-friends.mjs, so both hash names the same way.

// "Michał " -> "michal", "STÉPHANE" -> "stephane". Case, accents and anything
// that is not a letter are ignored, so friends only need nicknames as aliases.
export function normalizeName(s) {
  return String(s || "")
    .trim()
    .toLowerCase()
    .replace(/ł/g, "l").replace(/ß/g, "ss").replace(/æ/g, "ae").replace(/œ/g, "oe").replace(/ø/g, "o")
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/[^a-z]/g, "");
}

export async function hmacHex(secret, message) {
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey("raw", enc.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(message));
  return [...new Uint8Array(sig)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

// KV key for a first name. The pepper is a secret, so the keys can't be brute-forced offline.
export async function friendKey(pepper, firstName) {
  return "friend:" + (await hmacHex(pepper, "name:" + normalizeName(firstName)));
}
