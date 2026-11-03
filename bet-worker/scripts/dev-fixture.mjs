// Local testing only: fake an open betting window (or a finished race) from the real
// site JSON, written to .dev-race.json for the Worker to read from KV key dev:race.
//
//   node scripts/dev-fixture.mjs open      next race "predicted", lights out in 2 days
//   node scripts/dev-fixture.mjs settled   that race finished (finishes copied from the last race)
//   npx wrangler kv key put dev:race --path .dev-race.json --binding BETS --local

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const d = JSON.parse(readFileSync(join(root, "..", "site", "f1-predictions.json"), "utf8"));
const mode = process.argv[2] || "open";
const meta = { round: d.next.round, name: d.next.name, location: d.next.location, quali_utc: d.next.quali_utc };
const drivers = d.last.drivers.map(({ finish, ...rest }) => rest);

let fixture;
if (mode === "open") {
  const race_utc = new Date(Date.now() + 2 * 86400000).toISOString().replace(/\.\d+Z$/, "Z");
  fixture = { ...d, next: { status: "predicted", ...meta, race_utc, drivers } };
} else if (mode === "settled") {
  fixture = { ...d, next: { status: "awaiting_qualifying", round: meta.round + 1 }, last: { ...d.last, ...meta, race_utc: d.next.race_utc } };
} else {
  throw new Error("mode: open | settled");
}
writeFileSync(join(root, ".dev-race.json"), JSON.stringify(fixture));
console.log(`${mode}: round ${meta.round} ${meta.name} -> .dev-race.json`);
