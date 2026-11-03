import { refusalHtml, saveReceiptImage, ticketHtml } from "/receipt.js";

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const store = {
  get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

let race = null;
let session = null; // {token, name, first}

const STEPS = ["step-loading", "step-closed", "step-name", "step-bet", "step-done"];
function show(step) {
  for (const s of STEPS) $(s).hidden = s !== step;
}

function screen(text, lamp = "", busy = false) {
  $("screen").innerHTML = text;
  $("lamp").className = "lamp " + lamp;
  $("machine").classList.toggle("busy", busy);
}

async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) } : {});
  const data = await r.json().catch(() => ({ message: "Something broke. Try again." }));
  return { ok: r.ok, status: r.status, data };
}

const localTime = (iso) =>
  new Date(iso).toLocaleString(undefined, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

// ---------------------------------------------------------------- printer

async function feed(html) {
  const clip = $("clip");
  const paper = $("paper");
  clip.classList.remove("torn");
  clip.style.height = "0px";
  paper.innerHTML = html;
  const target = paper.offsetHeight;
  if (reduced) {
    clip.style.height = target + "px";
    return;
  }
  $("machine").classList.add("busy");
  let h = 0;
  while (h < target) {
    h = Math.min(target, h + 9 + Math.random() * 13);
    clip.style.height = h + "px";
    await sleep(45 + Math.random() * 45);
  }
  $("machine").classList.remove("busy");
}

async function tearOff() {
  const clip = $("clip");
  if (!clip.offsetHeight) return;
  clip.classList.add("torn");
  await sleep(reduced ? 0 : 450);
  clip.classList.remove("torn");
  clip.style.height = "0px";
}

function afterButtons(buttons) {
  const box = $("after");
  box.innerHTML = "";
  for (const b of buttons) box.append(b);
  box.hidden = !buttons.length;
}

function button(label, onClick, ghost = false) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "btn" + (ghost ? " ghost" : "");
  b.textContent = label;
  b.addEventListener("click", onClick);
  return b;
}

function link(label, href) {
  const a = document.createElement("a");
  a.className = "btn";
  a.href = href;
  a.textContent = label;
  return a;
}

async function printTicket(t, title, msg) {
  screen("Printing ticket<span class='dots'></span>", "wait", true);
  // Single-column layout: the printer sits under the form
  if (matchMedia("(max-width: 820px)").matches) {
    document.querySelector(".printer").scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
  }
  await feed(ticketHtml(t));
  screen("Bet accepted · good luck", "go");
  $("done-title").textContent = title;
  $("done-msg").textContent = msg;
  show("step-done");
  afterButtons([
    link("Open ticket page", "/r/" + t.id),
    button("Save image", (e) => saveImage(e.currentTarget, t.id), true),
  ]);
}

async function saveImage(btn, id) {
  btn.disabled = true;
  try {
    await saveReceiptImage($("paper"), "bet-" + id);
  } catch (err) {
    console.error(err);
    btn.textContent = "Screenshot it instead";
  } finally {
    btn.disabled = false;
  }
}

// ---------------------------------------------------------------- flow

async function init() {
  const { ok, data } = await api("/api/race");
  if (!ok) {
    screen("Out of paper", "hold");
    $("closed-race").textContent = "";
    $("closed-msg").textContent = "Couldn't load the race. Try again in a minute.";
    show("step-closed");
    return;
  }
  race = data;
  const raceLine = `${race.name}`;
  if (!race.open) {
    screen("Bets closed", "hold");
    $("closed-race").textContent = race.status === "season_over" ? "" : raceLine;
    $("closed-msg").textContent =
      race.status === "season_over" ? "Season's over. See you in March."
      : race.status === "awaiting_qualifying" ? `Betting opens once the model has seen qualifying (${localTime(race.quali_utc)}) and closes at lights out (${localTime(race.race_utc)}).`
      : "Lights are out. Betting is closed for this race.";
    show("step-closed");
    return;
  }
  for (const id of ["name-race", "bet-race"]) $(id).textContent = raceLine;
  $("first").value = store.get("bet:first") || "";
  $("last").value = store.get("bet:last") || "";
  screen("Insert name");
  show("step-name");

  $("driver").innerHTML = race.drivers
    .map((d) => `<option value="${d.driver}">${d.driver} · ${d.team}${d.grid ? " · P" + d.grid : ""}</option>`)
    .join("");
  $("wager").innerHTML = race.wagers.map((w) => `<option>${w}</option>`).join("");
  updatePreview();
}

function currentBet() {
  return {
    driver: $("driver").value,
    event: document.querySelector("input[name=event]:checked").value,
    negate: document.querySelector("input[name=negate]:checked").value === "1",
    wager: $("wager").value,
  };
}

function updatePreview() {
  const b = currentBet();
  const outcome = b.event === "win" ? "win the race" : "finish on the podium";
  $("preview").textContent = `I bet ${b.driver} ${b.negate ? "will not" : "will"} ${outcome}. Loser: ${b.wager.toLowerCase()}.`;
}

$("step-bet").addEventListener("change", updatePreview);

$("step-name").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = e.submitter;
  btn.disabled = true;
  $("name-error").textContent = "";
  screen("Checking the guest list<span class='dots'></span>", "wait", true);
  const { ok, data } = await api("/api/login", { first: $("first").value, last: $("last").value });
  btn.disabled = false;
  if (!ok) {
    screen(data.error === "unknown" ? "Access denied" : "Error", "hold");
    $("name-error").textContent = data.message || "Something broke.";
    return;
  }
  store.set("bet:first", $("first").value.trim());
  store.set("bet:last", $("last").value.trim());
  session = data;
  if (data.ticket) {
    await printTicket(data.ticket, `You're already in, ${data.first}`, "One ticket per race. Here's yours again.");
    return;
  }
  $("hello").textContent = `Alright ${data.first}, what's the bet?`;
  screen("Awaiting bet");
  show("step-bet");
});

$("step-bet").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("place");
  btn.disabled = true;
  $("bet-error").textContent = "";
  await tearOff();
  afterButtons([]);
  screen("Checking the odds<span class='dots'></span>", "wait", true);
  const [{ ok, status, data }] = await Promise.all([api("/api/bet", { token: session.token, ...currentBet() }), sleep(reduced ? 0 : 1100)]);
  btn.disabled = false;

  if (ok && data.accepted) {
    await printTicket(data.ticket, "Bet accepted", "Leon has been notified. Keep the ticket: scan the QR code to check the result after the race.");
    return;
  }
  if (ok && !data.accepted) {
    screen("Bet refused", "hold", true);
    await feed(refusalHtml(data));
    screen("Bet refused · try another", "hold");
    afterButtons([]);
    return;
  }
  if (status === 409 && data.ticket) {
    await printTicket(data.ticket, "You already have a ticket", "One ticket per race. Here's yours again.");
    return;
  }
  if (status === 401) {
    screen("Insert name");
    show("step-name");
    $("name-error").textContent = data.message;
    return;
  }
  screen("Error", "hold");
  $("bet-error").textContent = data.message || "Something broke. Try again.";
});

init();
