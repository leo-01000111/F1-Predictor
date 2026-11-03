// f1-bets: friends bet against the race model at bet.leongorecki.eu.
//
//   GET  /api/race        current race, whether betting is open, driver list (no odds)
//   POST /api/login       {first, last?} -> signed token if the first name is a known friend
//   POST /api/bet         {token, driver, event, negate, wager} -> ticket or refusal
//   GET  /api/ticket/:id  one ticket (for the receipt page)
//   GET  /r/:id           receipt page (public/receipt.html renders the ticket)
//   cron                  settles open tickets once the race result is published

import { friendKey, hmacHex, normalizeName } from "./names.js";
import { WAGERS, claimProbability, decide, settleClaim } from "./rules.js";

const LOGIN_FAILS_PER_HOUR = 8;
const BETS_PER_HOUR = 40;
const TOKEN_TTL_MS = 2 * 3600 * 1000;

const json = (data, status = 200) =>
  new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });

export default {
  async fetch(req, env, ctx) {
    const url = new URL(req.url);
    const path = url.pathname;
    try {
      if (path === "/api/race" && req.method === "GET") return json(await raceInfo(env));
      if (path === "/api/login" && req.method === "POST") return await login(req, env);
      if (path === "/api/bet" && req.method === "POST") return await placeBet(req, env, ctx);
      let m = path.match(/^\/api\/ticket\/([a-z0-9-]{6,40})$/);
      if (m && req.method === "GET") {
        const t = await env.BETS.get("ticket:" + m[1], "json");
        return t ? json(t) : json({ error: "not_found" }, 404);
      }
      if (/^\/r\/[a-z0-9-]{6,40}$/.test(path)) return env.ASSETS.fetch(new URL("/receipt", url));
      if (path.startsWith("/api/")) return json({ error: "not_found" }, 404);
      return env.ASSETS.fetch(req);
    } catch (err) {
      console.error(err && err.stack ? err.stack : err);
      return json({ error: "server_error" }, 500);
    }
  },

  async scheduled(event, env, ctx) {
    ctx.waitUntil(settle(env));
  },
};

// ---------------------------------------------------------------- race data

async function getRace(env) {
  // Local dev: a fixture in KV (scripts/dev-fixture.mjs) instead of the live JSON
  if (env.DEV === "1") {
    const fixture = await env.BETS.get("dev:race", "json");
    if (fixture) return fixture;
  }
  const r = await fetch(env.RACE_URL, { cf: { cacheTtl: 120, cacheEverything: true } });
  if (!r.ok) throw new Error(`race json: HTTP ${r.status}`);
  return r.json();
}

// Bets open once the post-qualifying prediction is out and close at lights out
function bettingWindow(d) {
  const n = d.next || {};
  const open = n.status === "predicted" && !!n.race_utc && Date.now() < Date.parse(n.race_utc);
  return { open, n };
}

async function raceInfo(env) {
  const d = await getRace(env);
  const { open, n } = bettingWindow(d);
  const drivers = (n.drivers || [])
    .map((r) => ({ driver: r.driver, team: r.team, grid: r.grid }))
    .sort((a, b) => (a.grid || 99) - (b.grid || 99));
  return {
    season: d.season,
    open,
    status: n.status,
    round: n.round,
    name: n.name,
    location: n.location,
    quali_utc: n.quali_utc,
    race_utc: n.race_utc,
    drivers: open ? drivers : [],
    wagers: WAGERS,
  };
}

// ---------------------------------------------------------------- tokens, limits

const b64url = (s) => btoa(unescape(encodeURIComponent(s))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
const unb64url = (s) => decodeURIComponent(escape(atob(s.replace(/-/g, "+").replace(/_/g, "/"))));

async function signToken(env, payload) {
  const body = b64url(JSON.stringify(payload));
  return body + "." + (await hmacHex(env.PEPPER, "tok:" + body));
}

async function verifyToken(env, token) {
  const [body, sig] = String(token || "").split(".");
  if (!body || !sig || sig !== (await hmacHex(env.PEPPER, "tok:" + body))) return null;
  const p = JSON.parse(unb64url(body));
  return p.exp > Date.now() ? p : null;
}

const clientIp = (req) => req.headers.get("cf-connecting-ip") || "local";

async function counter(env, kind, ip) {
  const key = `rl:${kind}:${ip}:${Math.floor(Date.now() / 3600000)}`;
  const n = parseInt((await env.BETS.get(key)) || "0", 10);
  return { n, bump: () => env.BETS.put(key, String(n + 1), { expirationTtl: 3700 }) };
}

// Free-text surname: printed on the ticket, not checked
function cleanSurname(s) {
  return String(s || "").replace(/[^\p{L}\s'-]/gu, "").replace(/\s+/g, " ").trim().slice(0, 30);
}

// ---------------------------------------------------------------- login

async function login(req, env) {
  const ip = clientIp(req);
  const fails = await counter(env, "login", ip);
  if (fails.n >= LOGIN_FAILS_PER_HOUR) {
    return json({ error: "rate_limited", message: "Too many wrong names. Try again in an hour." }, 429);
  }
  const body = await req.json().catch(() => ({}));
  const first = String(body.first || "").trim().split(/\s+/)[0];
  if (!normalizeName(first) || first.length > 40) return json({ error: "bad_name", message: "Type your first name." }, 400);

  const friend = await env.BETS.get(await friendKey(env.PEPPER, first), "json");
  if (!friend) {
    await fails.bump();
    return json({ error: "unknown", message: "I don't know you. Friends only." }, 403);
  }

  const surname = cleanSurname(body.last);
  const display = (friend.name + (surname ? " " + surname : "")).toUpperCase();
  const key = normalizeName(friend.name) + (surname ? "-" + normalizeName(surname) : "");
  const token = await signToken(env, { name: display, key, exp: Date.now() + TOKEN_TTL_MS });

  // Already holding a ticket for this race? Hand it back instead of a new bet.
  const d = await getRace(env);
  const { n } = bettingWindow(d);
  let ticket = null;
  if (n.round) {
    const id = await env.BETS.get(`slot:${d.season}-${n.round}:${key}`);
    if (id) ticket = await env.BETS.get("ticket:" + id, "json");
  }
  return json({ token, name: display, first: friend.name, ticket });
}

// ---------------------------------------------------------------- bet

async function placeBet(req, env, ctx) {
  const ip = clientIp(req);
  const tries = await counter(env, "bet", ip);
  if (tries.n >= BETS_PER_HOUR) return json({ error: "rate_limited", message: "Easy there. Try again in an hour." }, 429);
  await tries.bump();

  const body = await req.json().catch(() => ({}));
  const who = await verifyToken(env, body.token);
  if (!who) return json({ error: "auth", message: "Session expired. Type your name again." }, 401);

  const d = await getRace(env);
  const { open, n } = bettingWindow(d);
  if (!open) return json({ error: "closed", message: "Betting is closed for this race." }, 409);

  const row = (n.drivers || []).find((r) => r.driver === body.driver);
  const event = body.event === "win" ? "win" : body.event === "podium" ? "podium" : null;
  const wager = WAGERS.includes(body.wager) ? body.wager : null;
  if (!row || !event || !wager) return json({ error: "bad_bet", message: "Pick a driver, an outcome and a wager." }, 400);
  const negate = body.negate === true;

  const slot = `slot:${d.season}-${n.round}:${who.key}`;
  const existing = await env.BETS.get(slot);
  if (existing) {
    return json({ error: "already", message: "One ticket per race.", ticket: await env.BETS.get("ticket:" + existing, "json") }, 409);
  }

  const p = claimProbability(row, event, negate);
  const verdict = decide(p);
  const claim = { bettor: who.name, driver: row.driver, team: row.team, event, negate, p: round3(p), wager };
  if (!verdict.accepted) return json({ accepted: false, line: verdict.line, claim, race: n.name });

  const seqKey = `seq:${d.season}-${n.round}`;
  const seq = parseInt((await env.BETS.get(seqKey)) || "0", 10) + 1;
  await env.BETS.put(seqKey, String(seq));
  const suffix = (await hmacHex(env.PEPPER, `${slot}:${Date.now()}:${Math.random()}`)).slice(0, 4);
  const id = `${d.season}-${pad(n.round, 2)}-${pad(seq, 3)}-${suffix}`;

  const ticket = {
    id,
    ...claim,
    season: d.season,
    round: n.round,
    race: n.name,
    location: n.location,
    race_utc: n.race_utc,
    issued_at: new Date().toISOString(),
    line: verdict.line,
    status: "open",
  };
  await Promise.all([
    env.BETS.put("ticket:" + id, JSON.stringify(ticket)),
    env.BETS.put(slot, id),
    env.BETS.put(`open:${d.season}-${n.round}:${id}`, "1"),
  ]);

  const link = new URL("/r/" + id, req.url).toString();
  ctx.waitUntil(
    sendMail(
      env,
      `New bet: ${who.name} says ${claimText(ticket)}`,
      [
        `${who.name} placed a bet on the ${n.name}.`,
        "",
        `Claim:  ${claimText(ticket)} (${row.team})`,
        `Model:  ${Math.round(p * 100)}% chance it happens`,
        `Wager:  ${wager}`,
        `Ticket: ${id}`,
        "",
        link,
      ].join("\n"),
    ),
  );
  return json({ accepted: true, ticket });
}

// ---------------------------------------------------------------- settlement

async function settle(env) {
  const d = await getRace(env);
  const last = d.last;
  if (!last || !last.round) return;
  const prefix = `open:${d.season}-${last.round}:`;
  const settled = [];
  let cursor;
  do {
    const page = await env.BETS.list({ prefix, cursor });
    for (const k of page.keys) {
      const id = k.name.slice(prefix.length);
      const t = await env.BETS.get("ticket:" + id, "json");
      if (t && t.status === "open") {
        Object.assign(t, settleClaim(t, last.drivers), { settled_at: new Date().toISOString() });
        await env.BETS.put("ticket:" + id, JSON.stringify(t));
        settled.push(t);
      }
      await env.BETS.delete(k.name);
    }
    cursor = page.list_complete ? undefined : page.cursor;
  } while (cursor);

  if (!settled.length) return;
  const lines = settled.map((t) => {
    const who = t.bettor;
    const what = t.status === "won" ? `you owe ${who}: ${t.wager}` : t.status === "lost" ? `${who} owes you: ${t.wager}` : "void";
    return `${t.id}  ${who}: ${claimText(t)} -> ${t.status.toUpperCase()} (${what})`;
  });
  const won = settled.filter((t) => t.status === "lost").length;
  await sendMail(env, `Bets settled: ${last.name} (you won ${won}/${settled.length})`, lines.join("\n"));
}

// ---------------------------------------------------------------- helpers

function claimText(t) {
  const verb = t.event === "win" ? (t.negate ? "will not win" : "will win") : t.negate ? "will not podium" : "will podium";
  return `${t.driver} ${verb}`;
}

const pad = (n, w) => String(n).padStart(w, "0");
const round3 = (x) => Math.round(x * 1000) / 1000;

async function sendMail(env, subject, text) {
  if (!env.RESEND_API_KEY || !env.MAIL_TO) {
    console.log(`[mail not sent: no RESEND_API_KEY/MAIL_TO]\nSubject: ${subject}\n${text}`);
    return;
  }
  const r = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: { authorization: `Bearer ${env.RESEND_API_KEY}`, "content-type": "application/json" },
    body: JSON.stringify({ from: env.MAIL_FROM, to: [env.MAIL_TO], subject, text }),
  });
  if (!r.ok) console.error("resend", r.status, await r.text());
}
