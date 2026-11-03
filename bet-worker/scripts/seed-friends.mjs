// Hash friends.local.json into KV entries: friend:<hmac(pepper, first name)> -> {"name": ...}
//
//   node scripts/seed-friends.mjs            writes .friends-kv.json
//   npx wrangler kv bulk put .friends-kv.json --binding BETS --local    (or --remote)
//
// friends.local.json: [{"name": "Michał", "aliases": ["misiek", "michu"]}, ...]
// Case and accents are ignored when matching, so aliases only need nicknames and typos.
// The pepper comes from PEPPER in .dev.vars; production must use the same value.

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { friendKey, normalizeName } from "../src/names.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const devVars = readFileSync(join(root, ".dev.vars"), "utf8");
const pepper = (devVars.match(/^PEPPER\s*=\s*"?([^"\r\n]+)"?/m) || [])[1];
if (!pepper) throw new Error("PEPPER missing from .dev.vars");

const friends = JSON.parse(readFileSync(join(root, "friends.local.json"), "utf8"));
const entries = new Map();
for (const f of friends) {
  for (const alias of new Set([f.name, ...(f.aliases || [])].map(normalizeName).filter(Boolean))) {
    const key = await friendKey(pepper, alias);
    const prev = entries.get(key);
    if (prev && prev.value !== JSON.stringify({ name: f.name })) {
      console.warn(`alias "${alias}" is claimed by ${JSON.parse(prev.value).name} and ${f.name}; keeping the first`);
      continue;
    }
    entries.set(key, { key, value: JSON.stringify({ name: f.name }) });
  }
}
writeFileSync(join(root, ".friends-kv.json"), JSON.stringify([...entries.values()]));
console.log(`${friends.length} friends, ${entries.size} name keys -> .friends-kv.json`);
