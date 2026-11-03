// What the bookie takes, what it says, and how a ticket settles.

export const WAGERS = ["1x round of drinks", "1x coffee", "1x Happy Kebab"];

// Leon takes the other side of every claim. Even stakes, so he wants P(claim) well
// under 0.5: at most 35% means he wins at least ~65% of accepted bets if the model
// is calibrated.
export const ACCEPT_MAX = 0.35;

const pick = (lines) => lines[Math.floor(Math.random() * lines.length)];

export function decide(p) {
  if (p <= ACCEPT_MAX) {
    const line =
      p < 0.05 ? pick(["thanks for the free drink.", "this one's charity.", "I'll take that. obviously."])
      : p < 0.2 ? pick(["the model is rarely wrong. prove it.", "bold. wrong, but bold.", "brave pick. see you at the bar."])
      : pick(["fair fight. the model still likes its chances.", "close one. the house is in.", "you're on."]);
    return { accepted: true, line };
  }
  const line =
    p <= 0.55 ? pick(["coin flip. I don't do coin flips.", "too close to call. find some guts."])
    : p <= 0.8 ? pick(["that's a bit too probable for me.", "nice try. the model isn't that generous."])
    : pick(["yeah, of course that's gonna happen. try harder.", "you want me to bet against THAT?", "absolutely not. next."]);
  return { accepted: false, line };
}

// Probability of the friend's claim coming true, from a next.drivers row
export function claimProbability(row, event, negate) {
  const pe = event === "win" ? row.p_win : row.p_podium;
  return negate ? 1 - pe : pe;
}

// Settle against last.drivers[].finish. No finish (DNF/DNS) counts as no podium and no win.
// Returns "won" / "lost" from the bettor's side, or "void" if the driver is missing.
export function settleClaim(ticket, lastDrivers) {
  const row = (lastDrivers || []).find((r) => r.driver === ticket.driver);
  if (!row) return { status: "void", finish: null };
  const f = row.finish ?? null;
  const happened = ticket.event === "win" ? f === 1 : f != null && f <= 3;
  const claimTrue = ticket.negate ? !happened : happened;
  return { status: claimTrue ? "won" : "lost", finish: f };
}
