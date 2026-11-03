# f1-bets (bet.leongorecki.eu)

Friends bet against the race model. A Cloudflare Worker (static page + API + KV + hourly cron).

- **Who:** first name checked against salted hashes in KV (`friend:<hmac>`); last name is free text for the ticket.
  The plaintext list is `friends.local.json` (gitignored, never committed).
- **What:** driver + will / will not + win / podium + a wager from `src/rules.js` (`WAGERS`).
- **When:** opens once `next.status == "predicted"` in `site/f1-predictions.json` (after qualifying), closes at `race_utc`.
- **Accept:** the bookie takes the other side when the model gives the claim at most 35% (`ACCEPT_MAX`).
  One accepted ticket per person per race. Refusals store nothing and send no email.
- **Ticket:** printed on the page with a QR code to `/r/<id>`; the server copy is the proof.
- **Email:** one per accepted bet, one summary when the cron settles a race (Resend).
- **Settle:** hourly cron reads `last.drivers[].finish` for the ticket's round. No finish = no podium/no win;
  driver missing from the results = void.
- **Limits:** 8 wrong names per IP per hour, 40 bet attempts per IP per hour.

## Deploy (once)

```
cd bet-worker
npx wrangler kv namespace create BETS        # paste the id into wrangler.toml
npx wrangler secret put PEPPER               # paste PEPPER from .dev.vars (seeded hashes depend on it)
npx wrangler secret put RESEND_API_KEY
npx wrangler secret put MAIL_TO              # where bet emails go
node scripts/seed-friends.mjs
npx wrangler kv bulk put .friends-kv.json --binding BETS --remote
npx wrangler deploy                          # creates bet.leongorecki.eu (custom domain)
```

`MAIL_FROM` defaults to Resend's `onboarding@resend.dev`, which only delivers to the Resend account's own
address. To send from the domain, verify leongorecki.eu in Resend and change `MAIL_FROM` in `wrangler.toml`.

## Add or change friends

Edit `friends.local.json` (`{"name": "Michał", "aliases": ["misiek"]}`; case and accents are ignored), then
re-run `seed-friends.mjs` and `kv bulk put ... --remote`. Removing someone: `npx wrangler kv key delete` their keys,
or wipe `friend:*` and reseed.

## Local test

`.dev.vars` holds `DEV="1"` and a `PEPPER`; with no `RESEND_API_KEY` emails print to the wrangler log.

```
node scripts/seed-friends.mjs && npx wrangler kv bulk put .friends-kv.json --binding BETS --local
node scripts/dev-fixture.mjs open && npx wrangler kv key put dev:race --path .dev-race.json --binding BETS --local
npx wrangler dev --test-scheduled
# place bets at http://localhost:8787, then fake the result and settle:
node scripts/dev-fixture.mjs settled && npx wrangler kv key put dev:race --path .dev-race.json --binding BETS --local
curl "http://localhost:8787/__scheduled?cron=17+*+*+*+*"
```
