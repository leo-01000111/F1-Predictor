// Receipt markup, shared by the bet page (printer) and /r/:id (receipt viewer).
/* global qrcode, html2canvas */

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function claimLines(t) {
  const outcome = t.event === "win" ? "WIN THE RACE" : "FINISH ON THE PODIUM";
  return [`${t.driver} · ${t.team || ""}`.trim(), `${t.negate ? "WILL NOT" : "WILL"} ${outcome}`];
}

const fmtDate = (iso, withTime) =>
  new Date(iso)
    .toLocaleString("en-GB", {
      day: "2-digit", month: "short", year: "numeric",
      ...(withTime ? { hour: "2-digit", minute: "2-digit" } : {}),
    })
    .replace(",", " ")
    .toUpperCase();

const row = (k, v, cls = "") => `<div class="r-row ${cls}"><span>${esc(k)}</span><span>${v}</span></div>`;

// "São Paulo Grand Prix", round 20, 2026-11-08T17:00Z -> "São Paulo GP · R20 · 08.11.2026"
function raceLine(t) {
  const d = new Date(t.race_utc);
  const date = [d.getDate(), d.getMonth() + 1].map((x) => String(x).padStart(2, "0")).join(".") + "." + d.getFullYear();
  return `${esc(String(t.race).replace(/Grand Prix/i, "GP"))} · R${esc(t.round)} · ${date}`;
}

const mark = (label, d) =>
  `<svg class="r-mark" viewBox="0 0 16 16" role="img" aria-label="${label}"><path d="${d}"/></svg>`;
const CROSS = mark("no", "M3 3L13 13M13 3L3 13");
const TICK = mark("yes", "M2.5 8.5L6.5 12.5L13.5 3.5");

// The model's percentage, then the house's verdict on the claim: ✗ = "you're wrong" (bet taken)
const modelSays = (p, taken) =>
  row("MODEL SAYS", `${Math.round(p * 100)}% chance <b class="${taken ? "r-cross" : "r-tick"}">${taken ? CROSS : TICK}</b>`);
const rule = '<div class="r-rule"></div>';

function header() {
  return `<div class="r-head"><div class="r-title">LEON'S PIT WALL BOOKIE</div><div class="r-sub">bet.leongorecki.eu</div></div>`;
}

function qrSvg(text) {
  if (typeof qrcode === "undefined") return "";
  const qr = qrcode(0, "M");
  qr.addData(text);
  qr.make();
  return qr.createSvgTag({ cellSize: 4, margin: 0, scalable: true });
}

const STAMPS = {
  open: { text: "ACCEPTED", cls: "stamp-open" },
  won: { text: "BETTOR WON", cls: "stamp-won" },
  lost: { text: "HOUSE WON", cls: "stamp-lost" },
  void: { text: "VOID", cls: "stamp-void" },
};

function settlement(t) {
  if (t.status === "won") return `Leon owes ${esc(t.bettor)}<br>${esc(t.wager.toLowerCase())}`;
  if (t.status === "lost") return `${esc(t.bettor)} owes Leon<br>${esc(t.wager.toLowerCase())}`;
  if (t.status === "void") return "driver not classified · no bet";
  return "";
}

export function ticketHtml(t, origin = location.origin) {
  const [c1, c2] = claimLines(t);
  const stamp = STAMPS[t.status] || STAMPS.open;
  const link = `${origin}/r/${t.id}`;
  const result =
    t.status !== "open"
      ? `${rule}${row("RESULT", t.finish ? "P" + t.finish : "NO FINISH")}<div class="r-settle">${settlement(t)}</div>`
      : "";
  return `
    ${header()}
    ${rule}
    ${row("TICKET", "#" + esc(t.id.split("-").slice(0, 3).join("-")))}
    ${row("ISSUED", esc(fmtDate(t.issued_at, true)))}
    ${rule}
    <div class="r-race">${raceLine(t)}</div>
    ${rule}
    ${row("BETTOR", esc(t.bettor))}
    <div class="r-claim"><span>CLAIM</span><div><b>${esc(c1)}</b><br>${esc(c2)}</div></div>
    ${rule}
    ${modelSays(t.p, true)}
    ${row("WAGER", esc(t.wager.toUpperCase()))}
    ${rule}
    <div class="r-stamp ${stamp.cls}">${stamp.text}</div>
    ${result}
    ${rule}
    <div class="r-terms">Settled on the official result.<br>No finish counts as no podium.<br>No refunds. No crying.</div>
    <a class="r-qr" href="${esc(link)}" aria-label="Ticket page">${qrSvg(link)}</a>
    <div class="r-small">scan to verify this ticket</div>
    ${rule}
    <div class="r-quote">"${esc(t.line)}"</div>
  `;
}

export function refusalHtml(r) {
  const [c1, c2] = claimLines(r.claim);
  return `
    ${header()}
    ${rule}
    ${row("BETTOR", esc(r.claim.bettor))}
    <div class="r-claim"><span>CLAIM</span><div><b>${esc(c1)}</b><br>${esc(c2)}</div></div>
    ${modelSays(r.claim.p, false)}
    ${rule}
    <div class="r-stamp stamp-refused">BET REFUSED</div>
    <div class="r-quote">"${esc(r.line)}"</div>
    ${rule}
    <div class="r-small">no ticket issued · try again</div>
  `;
}

// Save the paper as a PNG: share sheet on phones (save to photos), download elsewhere
export async function saveReceiptImage(el, name) {
  // html2canvas spreads the paper's inset top shadow over the whole sheet, so the saved copy drops it
  const canvas = await html2canvas(el, {
    backgroundColor: null,
    scale: 2,
    onclone: (doc) => { doc.getElementById(el.id).style.boxShadow = "none"; },
  });
  const blob = await new Promise((res) => canvas.toBlob(res, "image/png"));
  const file = new File([blob], `${name}.png`, { type: "image/png" });
  if (navigator.canShare && navigator.canShare({ files: [file] })) {
    try {
      await navigator.share({ files: [file], title: "Bet ticket" });
      return;
    } catch (e) {
      if (e && e.name === "AbortError") return;
    }
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = file.name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}
