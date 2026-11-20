/* Coordinating Crowds: one page per tab, every number from data.js (written by experiments.py). */
"use strict";
const D = window.DATA, S = D.summary;
const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : Number(x).toFixed(d);
const NAME = { road: "Road", network: "Network", restaurant: "Restaurant", gym: "Gym", mess: "Mess", jaiswal: "Jaiswal dinner" };
const ORDER = D.order;
const POLICY_COLOR = { "no information": "--muted", "public display": "--ne", "display for 30%": "--gold",
  "CE advice": "--ce", "optimum as advice": "--pink", "CE advice + reward": "--opt" };
const TEAM = [
  { name: "Harsh Kumar Jha", roll: "24110130", branch: "Artificial Intelligence", email: "harsh.jha@iitgn.ac.in", role: "Mess case study (synthetic QR-scan data, calibration, forecasting); small crowds solved exactly, noisy users, robustness checks", c: "--ne" },
  { name: "Yash Goyal", roll: "24110399", branch: "Computer Science and Engineering", email: "yash.g@iitgn.ac.in", role: "Project lead and integration: solvers, queue lemma, information design, learning; report, website and lab", c: "--ce" },
  { name: "Piyush Kumar Keshri", roll: "24110260", branch: "Chemical Engineering", email: "piyush.keshri@iitgn.ac.in", role: "Equilibria: Nash and optimum solvers, correlated-equilibrium LPs, rewards", c: "--opt" },
  { name: "Garv Singhal", roll: "24110119", branch: "Computer Science and Engineering", email: "garv.singhal@iitgn.ac.in", role: "Day-to-day simulator, display policies, cross-facility experiments", c: "--violet" },
];

/* ------------------------------------------------------------------ tooltips and SVG helpers */
const tt = $("tt");
function tip(el, text) {
  el.addEventListener("mousemove", e => { tt.textContent = text; tt.style.opacity = 1; tt.style.left = (e.clientX + 12) + "px"; tt.style.top = (e.clientY - 30) + "px"; });
  el.addEventListener("mouseleave", () => tt.style.opacity = 0);
}
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
function txt(parent, x, y, s, o = {}) { const t = el("text", { x, y, "font-size": o.size || 11, fill: o.fill || css("--text-2"), "text-anchor": o.anchor || "middle", transform: o.rot || "" }, parent); t.textContent = s; return t; }
function niceMax(v) { if (v <= 0) return 1; const p = Math.pow(10, Math.floor(Math.log10(v))); const m = v / p; return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p; }

function bars(target, o) {
  const box = $(target); box.innerHTML = "";
  const W = o.width || 560, H = o.height || 250, m = { l: 46, r: 10, t: 14, b: o.rotate ? 58 : 30 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.aria || "bar chart" }, box);
  const all = o.series.flatMap(s => s.values).concat((o.ref || []).map(r => r.value));
  const ymax = o.ymax || niceMax(Math.max(...all) * 1.08);
  const x0 = m.l, x1 = W - m.r, y0 = H - m.b, y1 = m.t, y = v => y0 - Math.max(0, Math.min(v, ymax)) / ymax * (y0 - y1);
  for (let i = 0; i <= 4; i++) { const v = ymax * i / 4; el("line", { x1: x0, x2: x1, y1: y(v), y2: y(v), stroke: css("--line") }, svg); txt(svg, x0 - 6, y(v) + 4, +v.toFixed(2), { anchor: "end", fill: css("--muted") }); }
  const n = o.labels.length, band = (x1 - x0) / n, k = o.series.length, gap = 2, bw = Math.max(2, (band * 0.78 - gap * (k - 1)) / k);
  o.series.forEach((s, j) => s.values.forEach((v, i) => {
    const bx = x0 + i * band + band * 0.11 + j * (bw + gap), color = typeof s.color === "function" ? s.color(i) : s.color;
    const r = el("rect", { x: bx, y: y(v), width: bw, height: Math.max(0, y0 - y(v)), rx: Math.min(4, bw / 2), fill: color }, svg);
    if (o.labelsOnBars) txt(svg, bx + bw / 2, y(v) - 5, fmt(v, o.dec ?? 2), { size: 10.5, fill: css("--text-2") });
    tip(r, `${o.labels[i]} · ${s.name}: ${fmt(v, o.dec ?? 2)}${o.unit || ""}`);
  }));
  (o.ref || []).forEach(r => { el("line", { x1: x0, x2: x1, y1: y(r.value), y2: y(r.value), stroke: r.color || css("--muted"), "stroke-width": 1.6, "stroke-dasharray": "5 4" }, svg); if (r.label) txt(svg, x1, y(r.value) - 5, r.label, { anchor: "end", fill: r.color || css("--muted") }); });
  const step = Math.ceil(n / (o.maxTicks || 12));
  o.labels.forEach((lab, i) => { if (i % step) return; const cx = x0 + i * band + band / 2; txt(svg, cx, y0 + 16, lab, { anchor: o.rotate ? "end" : "middle", rot: o.rotate ? `rotate(-30 ${cx} ${y0 + 16})` : "" }); });
  if (o.yTitle) txt(svg, 12, (y0 + y1) / 2, o.yTitle, { rot: `rotate(-90 12 ${(y0 + y1) / 2})`, fill: css("--muted") });
}
function lines(target, o) {
  const box = $(target); box.innerHTML = "";
  const W = o.width || 560, H = o.height || 240, m = { l: 46, r: 12, t: 14, b: o.xTitle ? 38 : 28 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.aria || "line chart" }, box);
  const xs = o.x, xmin = o.xmin ?? Math.min(...xs), xmax = o.xmax ?? Math.max(...xs);
  const vals = o.series.flatMap(s => s.values.filter(v => v !== null)).concat((o.ref || []).map(r => r.value));
  const log = !!o.log, ymin = o.ymin ?? (log ? Math.min(...vals) * 0.8 : 0), ymax = o.ymax || (log ? Math.max(...vals) * 1.2 : niceMax(Math.max(...vals) * 1.05));
  const x0 = m.l, x1 = W - m.r, y0 = H - m.b, y1 = m.t;
  const X = v => x0 + (v - xmin) / (xmax - xmin || 1) * (x1 - x0);
  const Y = v => log ? y0 - (Math.log(Math.max(v, 1e-9)) - Math.log(ymin)) / (Math.log(ymax) - Math.log(ymin)) * (y0 - y1) : y0 - (v - ymin) / (ymax - ymin) * (y0 - y1);
  const ticks = log ? (o.yticks || [ymin, ymax]) : Array.from({ length: 5 }, (_, i) => ymin + (ymax - ymin) * i / 4);
  ticks.forEach(v => { el("line", { x1: x0, x2: x1, y1: Y(v), y2: Y(v), stroke: css("--line") }, svg); txt(svg, x0 - 6, Y(v) + 4, +(+v).toFixed(2), { anchor: "end", fill: css("--muted") }); });
  (o.ref || []).forEach(r => { el("line", { x1: x0, x2: x1, y1: Y(r.value), y2: Y(r.value), stroke: r.color, "stroke-width": 1.4, "stroke-dasharray": "5 4" }, svg); if (r.label) txt(svg, x1, Y(r.value) - 5, r.label, { anchor: "end", fill: r.color }); });
  o.series.forEach(s => {
    let d = ""; s.values.forEach((v, i) => { if (v === null) return; d += (d ? "L" : "M") + X(xs[i]).toFixed(1) + "," + Y(Math.min(v, ymax)).toFixed(1); });
    el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2, "stroke-linejoin": "round", "stroke-dasharray": s.dash || "" }, svg);
    if (s.dots) s.values.forEach((v, i) => { if (v === null) return; const c = el("circle", { cx: X(xs[i]), cy: Y(v), r: 4.2, fill: s.color, stroke: css("--surface"), "stroke-width": 2 }, svg); tip(c, `${s.name}: ${fmt(v)}`); });
  });
  (o.marks || []).forEach(p => { const c = el("circle", { cx: X(p.x), cy: Y(p.y), r: 6, fill: p.color, stroke: css("--surface"), "stroke-width": 2 }, svg); tip(c, `${p.label}: ${fmt(p.y)}`); });
  (o.xticks || [xmin, (xmin + xmax) / 2, xmax]).forEach(v => txt(svg, X(v), y0 + 16, o.xfmt ? o.xfmt(v) : v));
  if (o.xTitle) txt(svg, (x0 + x1) / 2, H - 4, o.xTitle, { fill: css("--muted") });
  if (o.yTitle) txt(svg, 12, (y0 + y1) / 2, o.yTitle, { rot: `rotate(-90 12 ${(y0 + y1) / 2})`, fill: css("--muted") });
  if (o.legend) { let lx = x0 + 6; o.series.filter(s => s.name).forEach(s => { el("rect", { x: lx, y: y1 + 2, width: 12, height: 3, fill: s.color }, svg); txt(svg, lx + 16, y1 + 7, s.name, { anchor: "start", size: 10.5 }); lx += 20 + s.name.length * 5.6; }); }
  const hit = el("rect", { x: x0, y: y1, width: x1 - x0, height: y0 - y1, fill: "transparent" }, svg);
  hit.addEventListener("mousemove", e => { const r = svg.getBoundingClientRect(), fx = (e.clientX - r.left) / r.width * W; let i = 0, best = 1e9; xs.forEach((v, j) => { const dd = Math.abs(X(v) - fx); if (dd < best) { best = dd; i = j; } }); tt.textContent = `${o.xfmt ? o.xfmt(xs[i]) : xs[i]} · ` + o.series.filter(s => s.name).map(s => `${s.name}: ${fmt(s.values[i])}`).join(" · "); tt.style.opacity = 1; tt.style.left = (e.clientX + 12) + "px"; tt.style.top = (e.clientY - 30) + "px"; });
  hit.addEventListener("mouseleave", () => tt.style.opacity = 0);
}
function table(id, head, rows, numCols = [], cls = () => "") {
  $(id).innerHTML = "<thead><tr>" + head.map((h, i) => `<th class="${numCols.includes(i) ? "num" : ""}">${h}</th>`).join("") + "</tr></thead><tbody>" +
    rows.map((r, ri) => "<tr>" + r.map((c, i) => `<td class="${numCols.includes(i) ? "num" : ""} ${cls(ri, i)}">${c}</td>`).join("") + "</tr>").join("") + "</tbody>";
}
function seg(id, items, current, onPick) {
  const box = $(id);
  box.innerHTML = items.map(([k, label]) => `<button class="btn small" data-k="${k}" aria-pressed="${k === current}">${label}</button>`).join("");
  box.querySelectorAll("button").forEach(b => b.onclick = () => { box.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", x === b)); onPick(b.dataset.k); });
}
function press(id, k) { const b = $(id).querySelector(`[data-k="${k}"]`); if (b) b.click(); }

/* ------------------------------------------------------------------ numbers used in several places */
const e2 = Object.fromEntries(S.e2.domains.map(r => [r.domain, r]));
const e3 = (dom, reg, pol) => S.e3.find(r => r.domain === dom && r.regime === reg && r.policy === pol);
const e4 = Object.fromEntries(S.e4.table.map(r => [r.domain, r]));
const poas = ORDER.map(k => e2[k].poa), poaMin = Math.min(...poas), poaMax = Math.max(...poas);
const herd = ORDER.map(k => e3(k, "reactive", "public display").vs_nash);

/* ------------------------------------------------------------------ OVERVIEW */
function drawOverview() {
  $("lnkReport").href = D.links.report; $("lnkApp").href = D.links.app; $("lnkRepo").href = D.links.repo;
  const e7 = S.e7;
  $("stats").innerHTML = [
    [`${fmt(poaMin, 1)}–${fmt(poaMax, 1)}×`, "selfish timing costs this many times the planner's schedule, across five facilities"],
    [`up to ${fmt(Math.max(...herd), 0)}×`, "the Nash cost when a reactive crowd follows a public display"],
    ["0%", "of the crowd switches slot each day with private correlated advice"],
    [`${Math.round(e7.hours_lost_per_day)} h`, "student-hours lost to selfish timing on a normal day at the four IITGN messes"],
  ].map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
  const steps = [
    ["--ne", "Everyone comes at once", "People choose when to use a shared bottleneck. Each one who joins a queue delays everyone behind.", `${Math.round(e7.hours_lost_per_day)} h`, "lost per day at the messes", "problem"],
    ["--ne", "Selfish timing is expensive", "The Nash equilibrium wastes most of the time a planner could save.", `${fmt(poaMin, 1)}–${fmt(poaMax, 1)}×`, "price of anarchy", "problem"],
    ["--gold", "A public display backfires", "If everyone reacts to the same numbers, yesterday's quiet slot is today's rush.", `${fmt(Math.min(...herd), 1)}–${fmt(Math.max(...herd), 0)}×`, "Nash cost, reactive crowd", "display"],
    ["--ce", "Private advice is stable", "Shuffled recommendations from a Nash schedule are a correlated equilibrium: nobody wants to disobey.", "0%", "daily switching, every crowd", "advice"],
    ["--opt", "A small reward reaches the optimum", "Correlation alone cannot beat Nash in a large crowd; a reward priced by the CE conditions can.", "0", "people worse off than selfish", "advice"],
  ];
  $("story").innerHTML = steps.map(([c, h, p, big, unit, page], i) => `<div class="st" data-go="${page}"><span class="num" style="background:var(${c})">${i + 1}</span><h4>${h}</h4><p>${p}</p><div class="big" style="color:var(${c})">${big}</div><div class="unit">${unit}</div></div>`).join("");
  $("story").querySelectorAll(".st").forEach(s => s.onclick = () => go(s.dataset.go));
  const ia = ORDER.map(k => S.e5[k]);
  const rel = k => S.e5[k].expected;
  const road = rel("road");
  // the exact small-crowd result at 10 people (E8)
  const c10 = S.e8.crowds.find(r => r.n === 10);
  const gapTen = fmt(100 * (c10.best_pure_nash - c10.best_ce) / (c10.best_pure_nash - c10.optimum), 0);
  document.querySelectorAll(".gapTen").forEach(e => e.textContent = gapTen + "%");
  // smallest share of app users at which each facility is back within 10% of Nash
  const enough = ORDER.map(k => Math.min(...S.e4.compliance.filter(r => r.domain === k && r.vs_nash <= 1.1).map(r => r.share)));
  $("findings").innerHTML = [
    `<b>Selfish timing is costly everywhere.</b> The Nash equilibrium costs ${fmt(poaMin, 1)}–${fmt(poaMax, 1)} times the planner's schedule in all five facilities (less, but still a lot, when arrivals are random).`,
    `<b>A shared display can multiply the cost.</b> With a reactive crowd it costs ${fmt(Math.min(...herd), 1)}–${fmt(Math.max(...herd), 0)} times Nash; showing it to only 30% of people removes most of the damage.`,
    `<b>Private advice is a correlated equilibrium.</b> Everyone obeys and nothing swings. Once ${fmt(Math.min(...enough) * 100, 0)}–${fmt(Math.max(...enough) * 100, 0)}% of people use the app, the herding of the others stops too.`,
    `<b>Correlation beats Nash in small crowds, not in large ones.</b> Ten people solved exactly: the best correlated equilibrium closes ${gapTen}% of the gap to the planner; in a large crowd we find no correlated equilibrium cheaper than Nash. Under uncertainty, what you announce matters far more than whether advice is private (road: announcing an accident costs ${fmt(road["announce the state"] / road["tell nobody"] * 100 - 100, 0)}% more than silence; a vague hint is best).`,
    `<b>A small reward reaches the optimum.</b> The reward the CE conditions require is a fraction of the time saved, and nobody ends up worse off than under selfish timing.`,
  ].map(s => `<li>${s}</li>`).join("");
}

/* ------------------------------------------------------------------ PROBLEM */
const PHOTOS = [
  ["road.jpg", "Roads", "Morning rush on a highway", "Photo: Countryball mys123, CC BY 4.0, via Wikimedia Commons"],
  ["canteen.jpg", "Canteens", "A university canteen queue", "Photo: Freooudzao, CC BY-SA 4.0, via Wikimedia Commons"],
  ["network.jpg", "Networks", "Jobs sharing one data link", "Photo: BalticServers.com, CC BY-SA 3.0, via Wikimedia Commons"],
  ["gym.jpg", "Gyms", "Evening crowd for machines", "Photo: U.S. Air Force / A. Alarcon, public domain"],
];
let poaDom = "road";
function drawProblem() {
  $("photos").innerHTML = PHOTOS.map(([f, h, s, c]) => `<figure><img src="img/${f}" alt="${s}" loading="lazy"><figcaption><b>${h}</b>${s}<small>${c}</small></figcaption></figure>`).join("");
  table("domTable", ["Facility", "Who", "Slot", "Serves", "Selfish (Nash)", "Planner", "Price of anarchy"],
    ORDER.map(k => { const r = e2[k], d = D.domains[k]; return [d.title, `${r.people} ${r.unit}`, `${r.slot_min} min`, `${r.service_rate} / min`, fmt(r.nash), fmt(r.opt), `<b>${fmt(r.poa, 1)}×</b>`]; }), [4, 5, 6]);
  seg("poaDomains", ORDER.map(k => [k, NAME[k]]), poaDom, k => { poaDom = k; drawPoa(); });
  drawPoa();
}
function drawPoa() {
  const d = D.domains[poaDom], r = e2[poaDom];
  bars("poaChart", { labels: d.labels, series: [{ name: "selfish", values: d.nash, color: css("--ne") }, { name: "planner", values: d.opt, color: css("--opt") }], ref: [{ value: d.capacity, label: "capacity", color: css("--muted") }], dec: 0, maxTicks: 6, yTitle: d.unit, height: 260 });
  $("poaNote").textContent = `${d.story} Selfish timing costs ${fmt(r.nash)} minutes per ${d.unit.replace(/s$/, "")}, the planner ${fmt(r.opt)}: a price of anarchy of ${fmt(r.poa, 1)}. The bumps are real: the slot right after a rush inherits its queue.`;
}

/* ------------------------------------------------------------------ SMALL GAMES */
function pigou(p) { return 1 / (1 - p * Math.pow(p + 1, -(p + 1) / p)); }
function drawGames() {
  const drawP = () => {
    const p = +$("pSl").value; $("pVal").textContent = p; $("pPoa").textContent = fmt(pigou(p)); $("pShare").textContent = Math.round(100 * Math.pow(p + 1, -1 / p)) + "%";
    const xs = Array.from({ length: 45 }, (_, i) => 1 + i * 0.25);
    lines("pigouChart", { x: xs, series: [{ name: "PoA", values: xs.map(pigou), color: css("--ne") }], marks: [{ x: p, y: pigou(p), color: css("--ne"), label: "you" }], ymin: 1, ymax: 4, width: 330, height: 200, xTitle: "steepness p", xticks: [1, 4, 8, 12] });
  };
  $("pSl").oninput = drawP; drawP();
  const drawL = () => {
    const L = +$("lSl").value; $("lVal").textContent = L.toFixed(1);
    const counter = 2 + L, mixed = 2 * L, ce = 1 + L, opt = Math.min(4, 1 + L);
    $("dinnerMatrix").innerHTML = `<tr><th></th><th>Peak</th><th>Late</th></tr><tr><th>Peak</th><td>(2, 2)</td><td>(1, ${L.toFixed(1)})</td></tr><tr><th>Late</th><td>(${L.toFixed(1)}, 1)</td><td>(${L.toFixed(1)}, ${L.toFixed(1)})</td></tr>`;
    bars("dinnerChart", { labels: ["shared counter", "no information", "private coin"], series: [{ name: "total cost", values: [counter, mixed, ce], color: i => [css("--ne"), css("--muted"), css("--ce")][i] }], ref: [{ value: opt, label: "optimum", color: css("--opt") }], width: 330, height: 210, ymax: 4.5, labelsOnBars: true, dec: 1 });
    $("dinnerNote").textContent = `Reacting to yesterday's counter, both students jump together (${counter.toFixed(1)}). With no information each goes Peak with probability ${(L - 1).toFixed(1)} (${mixed.toFixed(1)}). A fair private coin ("you Peak, you Late") is a correlated equilibrium: both obey, and it reaches the optimum (${ce.toFixed(1)}).`;
  };
  $("lSl").oninput = drawL; drawL();
  const I = S.e1.intersection;
  bars("lightChart", { labels: ["both stop", "one goes", "crash"], series: [{ name: "best CE", values: [0.5, 0.5, 0], color: css("--ce") }, { name: "mixed Nash", values: [4 / 9, 4 / 9, 1 / 9], color: css("--ne") }], width: 330, height: 190, ymax: 0.6, dec: 2 });
  $("lightNote").innerHTML = `The best correlated equilibrium (a linear program) never recommends a crash. Each driver gets <b>${fmt(I.best_ce_each)}</b>, more than any mixture of Nash equilibria (<b>${fmt(I.best_symmetric_nash)}</b>). That is what a traffic light does.`;
  $("learnNote").innerHTML = `Yes. Players using <b>regret matching</b> (move away from actions they regret) end up playing a correlated equilibrium: after 6000 rounds of the traffic-light game the largest gain from disobeying is ${fmt(S.e1.learning.regret_matching_ce_gap, 4)}. Players using <b>Hedge</b> reach the larger coarse-CE set. A mediator simply gets there on the first day. At scale (all five facilities, 300 days), learners who see every slot's wait settle near Nash with little regret; learners who see only their own wait learn far worse.`;
}

/* ------------------------------------------------------------------ PUBLIC DISPLAY */
const POLICIES = ["no information", "public display", "display for 30%", "CE advice", "CE advice + reward"];
const NOTES = { "no information": "People only remember their own past waits.", "public display": "Everyone sees yesterday's exact crowd and reacts: the rush jumps from slot to slot.", "display for 30%": "Only 30% see the display: too few move together to shift the rush.", "CE advice": "Private, shuffled slots from a Nash schedule: nobody wants to disobey, nothing moves.", "CE advice + reward": "The planner's schedule, made self-enforcing by a small reward: queues vanish." };
let dDom = "mess", dPol = "public display", day = 0, timer = null;
function drawDisplay() {
  seg("dispDomains", ORDER.map(k => [k, NAME[k]]), dDom, k => { dDom = k; day = 0; drawPlay(); drawDispTable(); });
  seg("dispPolicies", POLICIES.map(p => [p, p[0].toUpperCase() + p.slice(1)]), dPol, k => { dPol = k; drawPlay(); });
  $("daySl").oninput = e => { day = +e.target.value; drawPlay(); };
  $("playBtn").onclick = () => togglePlay();
  drawPlay(); drawDispTable();
}
function togglePlay(force) {
  const on = force !== undefined ? force : !timer;
  if (!on) { clearInterval(timer); timer = null; $("playBtn").textContent = "▶ Play"; return; }
  if (timer) return;
  if (day >= 119) day = 0;
  $("playBtn").textContent = "❚❚ Pause";
  timer = setInterval(() => { day++; if (day >= 119) togglePlay(false); drawPlay(); }, 130);
}
function drawPlay() {
  const P = D.playback[dDom].policies[dPol], cap = D.playback[dDom].capacity, labels = D.playback[dDom].labels;
  const nash = e4[dDom].nash, opt = e4[dDom].opt;
  $("dayVal").textContent = day + 1; $("daySl").value = day;
  const peak = Math.max(...Object.values(D.playback[dDom].policies).flatMap(p => p.loads.slice(40).flat()));
  bars("playChart", { labels, series: [{ name: "arrivals", values: P.loads[day], color: i => P.loads[day][i] > cap ? css("--ne") : css("--ce") }], ref: [{ value: cap, label: "capacity", color: css("--muted") }], ymax: niceMax(peak * 1.05), dec: 0, maxTicks: 6, yTitle: D.playback[dDom].unit });
  $("polNote").textContent = NOTES[dPol];
  const xs = P.cost.map((_, i) => i + 1), top = Math.max(...Object.values(D.playback[dDom].policies).flatMap(p => p.cost.slice(10)));
  lines("costChart", { x: xs, series: [{ name: "this policy", values: P.cost.map((v, i) => i <= day ? v : null), color: css("--ce") }], ref: [{ value: nash, label: "Nash", color: css("--ne") }, { value: opt, label: "planner", color: css("--opt") }], ymax: niceMax(top * 1.05), xTitle: "day", xticks: [1, 40, 80, 120], yTitle: "minutes" });
  $("pCost").textContent = fmt(P.mean) + " min"; $("pSwing").textContent = Math.round(100 * P.swing) + "%";
}
function drawDispTable() {
  const pols = ["no information", "public display", "display for 30%", "CE advice", "optimum as advice", "CE advice + reward"];
  table("dispTable", [`${NAME[dDom]}: policy`, "Calm", "Reactive", "Noisy", "Worst case"], pols.map(p => {
    const v = ["calm", "reactive", "noisy"].map(r => e3(dDom, r, p).vs_nash);
    return [p, ...v.map(x => fmt(x)), `<b>${fmt(Math.max(...v))}</b>`];
  }), [1, 2, 3, 4], (ri, ci) => ci > 0 && ci < 4 && ["public display"].includes(pols[ri]) && e3(dDom, ["calm", "reactive", "noisy"][ci - 1], pols[ri]).vs_nash > 2 ? "bad" : "");
}

/* ------------------------------------------------------------------ CORRELATED ADVICE */
let ph = "nash", bDom = "road";
function drawAdvice() {
  seg("phoneBtns", [["nash", "Nash schedule (a CE)"], ["opt", "Planner's optimum"], ["reward", "Optimum + reward"]], ph, k => { ph = k; drawPhone(); });
  drawPhone();
  const comp = S.e4.compliance, shares = [...new Set(comp.map(r => r.share))];
  const cols = ["--ce", "--ne", "--opt", "--gold", "--violet"];
  lines("compChart", { x: shares, series: ORDER.map((k, i) => ({ name: NAME[k], values: shares.map(s => comp.find(r => r.domain === k && r.share === s).vs_nash), color: css(cols[i]), dots: true })), ref: [{ value: 1, label: "Nash", color: css("--text-2") }], log: true, ymin: 0.8, ymax: Math.max(...comp.map(r => r.vs_nash)) * 1.2, yticks: [1, 2, 5, 10], xticks: [0, 0.5, 1], xfmt: v => Math.round(v * 100) + "%", xTitle: "share of people using the app", legend: true, height: 260 });
  bars("rewardChart", { labels: ORDER.map(k => NAME[k]), series: [{ name: "reward per person", values: ORDER.map(k => e4[k].reward_pp), color: css("--gold") }, { name: "time saved per person", values: ORDER.map(k => e4[k].saving_pp), color: css("--opt") }], unit: " min", yTitle: "minutes", height: 260, labelsOnBars: true, dec: 2 });
  table("lpTable", ["Facility", "Nash", "Best correlated eq.", "Planner", "Gain from correlation", "Equilibrium accuracy"],
    ORDER.map(k => { const r = e4[k]; return [NAME[k], fmt(r.large_nash, 3), fmt(r.large_best_ce, 3), fmt(r.opt, 3), fmt(r.large_nash - r.large_best_ce, 3), `${fmt(r.large_gap_sec, 3)} s`]; }), [1, 2, 3, 4, 5]);
  const pols = ["tell nobody", "announce the state", "best public hint", "private advice"], pc = ["--muted", "--ne", "--gold", "--ce"];
  $("infoLegend").innerHTML = pols.map((p, i) => `<span><i style="background:var(${pc[i]})"></i>${p}</span>`).join("");
  bars("infoChart", { labels: ORDER.map(k => NAME[k]), series: pols.map((p, i) => ({ name: p, values: ORDER.map(k => S.e5[k].expected[p] / S.e5[k].expected["tell nobody"]), color: css(pc[i]) })), ymax: 1.4, height: 250, dec: 2, ref: [{ value: 1, color: css("--text-2") }] });
  seg("beliefDomains", ORDER.map(k => [k, NAME[k]]), bDom, k => { bDom = k; drawBelief(); });
  drawBelief();
  const road = S.e5.road.expected;
  $("infoNote").innerHTML = `<b>What to announce matters far more than whether advice is private.</b> Telling everyone about a bad day helps a lot at the mess, restaurant and gym, but on the <b>road</b> it costs ${fmt(road["announce the state"])} min per car against ${fmt(road["tell nobody"])} with silence: everyone leaves at the same "safe" time. A vague hint does better than both (${fmt(road["best public hint"])}). Private correlated advice is never worse, but in a large crowd it adds at most ${fmt(Math.max(...ORDER.map(k => 100 * (1 - S.e5[k].expected["private advice"] / S.e5[k].expected["best public hint"]))), 1)}%. Where correlation pays off is coordination (no herding) and carrying rewards.`;
}
function drawPhone() {
  const p = D.phones[ph], labels = D.mess_schedule.labels, best = Math.min(...p.costs), obey = p.costs[p.rec] <= best + 0.01;
  $("phSlot").textContent = labels[p.rec];
  $("phPref").textContent = `you like ${labels[p.tau]} · flexibility ${p.beta} min per slot`;
  $("phReward").innerHTML = ph === "reward" ? `🎟️ Come at ${labels[p.rec]} and get <b>${fmt(D.phones.opt.costs[p.rec] - p.costs[p.rec])} min</b> of priority` : "";
  $("phVerdict").innerHTML = obey ? `<span style="color:var(--opt)">✓ Following the app is your best choice</span>` : `<span style="color:var(--ne)">✗ You would rather go at ${labels[p.costs.indexOf(best)]}</span>`;
  bars("phoneChart", { labels, series: [{ name: "your cost", values: p.costs, color: i => i === p.rec ? css("--ce") : css("--line") }], unit: " min", yTitle: "minutes", maxTicks: 6 });
}
function drawBelief() {
  const r = S.e5[bDom], q = r.beliefs, n = q.length;
  lines("beliefChart", { x: q, series: [{ name: "cost if everyone believes q", values: r.value, color: css("--text-2") }, { name: "best public hint", values: r.envelope, color: css("--gold"), dash: "6 4" }, { name: "announce", values: q.map(x => (1 - x) * r.value[0] + x * r.value[n - 1]), color: css("--ne"), dash: "2 4", width: 1.5 }],
    marks: [{ x: 0.3, y: r.expected["tell nobody"], color: css("--muted"), label: "tell nobody" }, { x: 0.3, y: r.expected["announce the state"], color: css("--ne"), label: "announce" }, { x: 0.3, y: r.expected["best public hint"], color: css("--gold"), label: "best hint" }],
    xticks: [0, 0.3, 0.5, 1], xTitle: "belief that today is a bad day (prior 0.3)", yTitle: "minutes", height: 250, legend: true });
  $("beliefNote").textContent = `${NAME[bDom]}: a bad day means ${r.bad_state}. Any public message moves the crowd's belief; the best one is read off the lower convex envelope at 0.3 (Bayesian persuasion). Equilibria accurate to ${fmt(r.max_gap_sec, 1)} s.`;
}

/* ------------------------------------------------------------------ MESS */
let mPol = "public display", mDay = 40, mTimer = null;
function drawMess() {
  const e7 = S.e7, j = e7.jaiswal;
  $("messStats").innerHTML = [
    ["3000", "students in four messes (Jaiswal, Mohani, Argoras, Bhopal)"],
    [`${(e7.rows / 1000).toFixed(0)}k`, "synthetic QR scans over eight weeks"],
    [`${Math.round(e7.hours_lost_per_day)} h`, "student-hours lost to selfish timing on a normal day"],
    [`${fmt(j.nash)} → ${fmt(j.opt)}`, "minutes per student at Jaiswal dinner: Nash → CE advice + reward"],
  ].map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
  drawHeat();
  const MEAL_ORDER = ["breakfast", "lunch", "dinner"];
  const agg = [...e7.by_meal].sort((a, b) => MEAL_ORDER.indexOf(a.meal) - MEAL_ORDER.indexOf(b.meal) || (a.day === "normal" ? -1 : 1));
  bars("mealChart", { labels: agg.map(r => `${r.meal} ${r.day}`), series: [{ name: "selfish", values: agg.map(r => r.nash), color: css("--ne") }, { name: "planner", values: agg.map(r => r.opt), color: css("--opt") }], unit: " min", rotate: true, height: 250, labelsOnBars: true, dec: 1 });
  seg("messPolicies", [["no information", "No information"], ["public display", "Public display"], ["CE advice", "CE advice"], ["CE advice + reward", "CE advice + reward"]], mPol, k => { mPol = k; drawMessPlay(); });
  $("messPlay").onclick = () => toggleMess();
  drawMessPlay();
  const pols = ["no information", "public display", "display 3 dinners old", "display + private noise", "display in 3 levels", "display for 30%", "moving-average forecast", "CE advice", "optimum as advice", "CE advice + reward"];
  table("messTable", ["Policy", "Calm", "Reactive", "Noisy"], pols.map(p => [p, ...["calm", "reactive", "noisy"].map(r => fmt(j.policies.find(x => x.policy === p && x.regime === r).cost))]), [1, 2, 3],
    (ri, ci) => ci > 0 && j.policies.find(x => x.policy === pols[ri] && x.regime === ["calm", "reactive", "noisy"][ci - 1]).cost > 2 * j.nash ? "bad" : "");
  const pr = e7.preferences.dinner, cal = e7.calibration.filter(r => r.meal === "dinner");
  const info = e7.information.expected;
  $("calibList").innerHTML = [
    `<b>Serving rate from scans alone.</b> While a queue exists, plates leave at the serving rate, so the busiest stretch of scans reveals it: ${cal.map(r => `${r.menu} dinner ${r.estimated_mu} (true ${r.true_mu})`).join(", ")} students per minute.`,
    `<b>Preferred times from scans.</b> A student's usual scan time on normal days recovers their favourite slot for ${Math.round(pr.within_1_slot * 100)}% of students within one slot. That is what the app needs to give advice.`,
    `<b>Who gains from the reward.</b> Without it, flexible students would carry the inconvenience; with it, ${e7.worse_off.with_reward} of the diners are worse off than under selfish timing. The reward costs ${fmt(e7.budget_pp)} min per student on average.`,
    `<b>Slow nights.</b> If the counter is short-staffed 30% of the time, announcing it lowers the expected cost from ${fmt(info["tell nobody"])} to ${fmt(info["announce the state"])} minutes; private advice adds almost nothing (${fmt(info["private advice"])}).`,
  ].map(s => `<li>${s}</li>`).join("");
  const sw = e7.sweeps.filter(r => r.sweep === "share");
  $("officeList").innerHTML = [
    `<b>Do not put yesterday's exact queue on a screen for everyone.</b> In a reactive crowd it raises the dinner cost from ${fmt(j.nash)} to ${fmt(j.policies.find(x => x.policy === "public display" && x.regime === "reactive").cost)} minutes.`,
    `<b>If you show crowd numbers, show them to a few or make them fresh.</b> Shown to 30% of students the cost is ${fmt(sw.find(r => Math.abs(r.value - 0.3) < 1e-9).cost)}; shown to everyone ${fmt(sw.find(r => r.value === 1).cost)}.`,
    `<b>Send private slot suggestions from a Nash schedule.</b> Everyone wants to follow them, nobody swings, and it works for every kind of crowd.`,
    `<b>To go further, reward following the advice</b> (priority token, dosa coupon): about ${fmt(e7.budget_pp)} min per student buys the planner's ${fmt(j.opt)}-minute schedule.`,
    `<b>On slow nights, tell students.</b> Announcing a short-staffed counter helps here (unlike the road, where it backfires).`,
  ].map(s => `<li>${s}</li>`).join("");
}
function drawHeat() {
  const H = D.heatmap, box = $("heatChart"); box.innerHTML = "";
  const W = 560, Hh = 250, m = { l: 40, r: 10, t: 8, b: 28 }, rows = H.days.length, cols = H.times.length;
  const svg = el("svg", { viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": "scan heatmap" }, box);
  const cw = (W - m.l - m.r) / cols, ch = (Hh - m.t - m.b) / rows, vmax = Math.max(...H.values.flat());
  const ramp = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
  H.values.forEach((row, i) => row.forEach((v, j) => { const r = el("rect", { x: m.l + j * cw, y: m.t + i * ch, width: cw - 1.5, height: ch - 1.5, rx: 3, fill: ramp[Math.min(7, Math.floor(v / vmax * 7.999))] }, svg); tip(r, `${H.days[i]} ${H.times[j]}: ${fmt(v, 0)} scans`); }));
  H.days.forEach((d, i) => txt(svg, m.l - 6, m.t + i * ch + ch / 2 + 4, d, { anchor: "end" }));
  H.times.forEach((t, j) => { if (j % 2 === 0) txt(svg, m.l + j * cw + cw / 2, Hh - 8, t); });
}
function toggleMess(force) {
  const on = force !== undefined ? force : !mTimer;
  if (!on) { clearInterval(mTimer); mTimer = null; $("messPlay").textContent = "▶ Play"; return; }
  if (mTimer) return;
  $("messPlay").textContent = "❚❚ Pause";
  mTimer = setInterval(() => { mDay++; if (mDay >= 119) { mDay = 40; } drawMessPlay(); }, 140);
}
function drawMessPlay() {
  const P = D.playback.jaiswal.policies[mPol], cap = D.playback.jaiswal.capacity, labels = D.playback.jaiswal.labels;
  const peak = Math.max(...Object.values(D.playback.jaiswal.policies).flatMap(p => p.loads.slice(40).flat()));
  bars("messPlayChart", { labels, series: [{ name: "arrivals", values: P.loads[mDay], color: i => P.loads[mDay][i] > cap ? css("--ne") : css("--ce") }], ref: [{ value: cap, label: "capacity", color: css("--muted") }], ymax: niceMax(peak * 1.05), dec: 0, maxTicks: 6, yTitle: `students, dinner ${mDay + 1}` });
  const xs = P.cost.map((_, i) => i + 1);
  lines("messCostChart", { x: xs, series: [{ name: "cost per student", values: P.cost.map((v, i) => i <= mDay ? v : null), color: css("--ce") }], ref: [{ value: S.e7.jaiswal.nash, label: "Nash", color: css("--ne") }, { value: S.e7.jaiswal.opt, label: "planner", color: css("--opt") }], ymax: 16, xTitle: "dinner", xticks: [1, 40, 80, 120], yTitle: "minutes" });
}

/* ------------------------------------------------------------------ TEAM */
function drawTeam() {
  $("team").innerHTML = TEAM.map(t => `<div class="member"><div class="av" style="background:var(${t.c})">${t.name.split(" ").map(w => w[0]).slice(0, 2).join("")}</div><h3>${t.name}</h3><div class="meta">${t.roll} · ${t.branch}</div><a href="mailto:${t.email}">${t.email}</a><div class="role">${t.role}</div></div>`).join("");
  $("credits").innerHTML = `Photos via Wikimedia Commons: ${PHOTOS.map(p => p[3].replace("Photo: ", "")).join("; ")}.`;
}

/* ------------------------------------------------------------------ router */
const DRAW = { overview: drawOverview, problem: drawProblem, games: drawGames, display: drawDisplay, advice: drawAdvice, mess: drawMess, team: drawTeam };
const drawn = {};
function go(page, scroll = true) {
  if (!DRAW[page]) page = "overview";
  document.querySelectorAll(".page").forEach(p => p.classList.toggle("on", p.id === "page-" + page));
  document.querySelectorAll(".tabs a").forEach(a => a.classList.toggle("on", a.dataset.page === page));
  if (location.hash !== "#" + page) history.replaceState(null, "", "#" + page);
  if (!drawn[page]) { DRAW[page](); drawn[page] = true; }
  if (page !== "display") togglePlay(false);
  if (page !== "mess") toggleMess(false);
  if (scroll) window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
}
document.querySelectorAll(".tabs a, a.brand").forEach(a => a.addEventListener("click", e => { e.preventDefault(); go((a.dataset.page || "overview")); }));
window.addEventListener("hashchange", () => go(location.hash.slice(1)));

/* ------------------------------------------------------------------ guided demo */
const TOUR = [
  { page: "overview", el: ".hero", t: "The problem", x: "Many people share one bottleneck and each picks when to use it. Everyone wants a short queue, so everyone comes at once. The question: what should we tell people, and who should be told what?" },
  { page: "overview", el: "#story", t: "The story in five steps", x: "Selfish timing is expensive; a public display can make it worse; private advice that people want to follow is stable; and a small reward reaches the planner's optimum." },
  { page: "problem", el: "#photos", t: "Congestion is everywhere", x: "Roads, canteens, network links, gyms: different places, one game. Our model covers all of them with the same few numbers." },
  { page: "problem", el: "#poaChart", t: "Selfish timing vs the planner", x: "Orange is what selfish people do (a Nash equilibrium); green is the planner's schedule, which feeds the bottleneck exactly at capacity.", act: () => press("poaDomains", "road") },
  { page: "games", el: "#cardDinner", t: "Two students, Peak or Late", x: "A shared counter makes both jump together; no information wastes time by chance; a private fair coin (a correlated equilibrium) reaches the optimum." },
  { page: "games", el: "#cardLight", t: "A traffic light", x: "The best correlated equilibrium never recommends a crash and gives each driver more than any mixture of Nash equilibria." },
  { page: "display", el: "#dispCharts", t: "A public display, reactive crowd", x: "Everyone sees yesterday's crowd and reacts the same way: the rush jumps from slot to slot and the cost explodes.", act: () => { press("dispDomains", "mess"); press("dispPolicies", "public display"); day = 0; togglePlay(true); } },
  { page: "display", el: "#dispCharts", t: "Private correlated advice", x: "Each person gets a private slot from a Nash schedule. Nobody wants to disobey, so nothing moves, for every kind of crowd.", act: () => { togglePlay(false); press("dispPolicies", "CE advice"); day = 0; togglePlay(true); } },
  { page: "advice", el: "#phoneRow", t: "Why people obey", x: "With Nash advice the recommended slot is the student's cheapest option. With the planner's schedule some students are tempted to leave. A reward priced by the CE conditions removes the temptation.", act: () => { press("phoneBtns", "nash"); setTimeout(() => tourOn && press("phoneBtns", "opt"), 3000); setTimeout(() => tourOn && press("phoneBtns", "reward"), 6000); } },
  { page: "advice", el: "#infoRow", t: "Who should be told what", x: "When the operator knows today is a bad day: announcing can backfire (road), a vague hint can beat both silence and announcing, and private advice adds very little on top.", act: () => press("beliefDomains", "road") },
  { page: "mess", el: "#messDispCard", t: "The IITGN mess", x: "3000 students, four messes, synthetic QR scans. The same picture: a public display makes dinner chaotic; private advice keeps it calm.", act: () => { press("messPolicies", "public display"); toggleMess(true); } },
  { page: "mess", el: "#officeCard", t: "What the mess office should do", x: "No public exact-queue screen for everyone; private slot suggestions from a Nash schedule; a small reward to reach the planner's schedule.", act: () => toggleMess(false) },
  { page: "team", el: "#team", t: "The team", x: "Thanks for watching. Every number on this site comes from the project's code; the report has the full method." },
];
let tourIdx = 0, tourOn = false, tourAuto = true, tourTimer = null;
function tourShow(i) {
  tourIdx = Math.max(0, Math.min(TOUR.length - 1, i));
  const s = TOUR[tourIdx];
  go(s.page, false);
  document.querySelectorAll(".focus").forEach(e => e.classList.remove("focus"));
  const target = document.querySelector(s.el);
  if (target) { target.classList.add("focus"); target.scrollIntoView({ behavior: "smooth", block: "center" }); }
  if (s.act) s.act();
  $("tourStep").textContent = `Step ${tourIdx + 1} of ${TOUR.length}`;
  $("tourTitle").textContent = s.t; $("tourText").textContent = s.x;
  $("tourDots").innerHTML = TOUR.map((_, j) => `<i class="${j === tourIdx ? "on" : j < tourIdx ? "done" : ""}" data-j="${j}"></i>`).join("");
  $("tourDots").querySelectorAll("i").forEach(d => d.onclick = () => tourShow(+d.dataset.j));
  $("tourNext").textContent = tourIdx === TOUR.length - 1 ? "Finish" : "Next";
  clearTimeout(tourTimer);
  if (tourAuto) tourTimer = setTimeout(() => tourIdx < TOUR.length - 1 ? tourShow(tourIdx + 1) : null, 9000);
}
function tourStart() { tourOn = true; tourAuto = true; $("tourAuto").textContent = "❚❚ pause"; $("tour").hidden = false; tourShow(0); }
function tourEnd() { tourOn = false; clearTimeout(tourTimer); $("tour").hidden = true; document.querySelectorAll(".focus").forEach(e => e.classList.remove("focus")); togglePlay(false); toggleMess(false); }
$("tourBtn").onclick = tourStart; $("tourBtn2").onclick = tourStart;
$("tourExit").onclick = tourEnd;
$("tourBack").onclick = () => tourShow(tourIdx - 1);
$("tourNext").onclick = () => tourIdx === TOUR.length - 1 ? tourEnd() : tourShow(tourIdx + 1);
$("tourAuto").onclick = () => { tourAuto = !tourAuto; $("tourAuto").textContent = tourAuto ? "❚❚ pause" : "▶ auto-play"; if (tourAuto) tourShow(tourIdx); else clearTimeout(tourTimer); };
document.addEventListener("keydown", e => {
  if (!tourOn) return;
  if (e.key === "ArrowRight") tourShow(tourIdx + 1);
  else if (e.key === "ArrowLeft") tourShow(tourIdx - 1);
  else if (e.key === "Escape") tourEnd();
  else if (e.key === " ") { e.preventDefault(); $("tourAuto").click(); }
});

/* ------------------------------------------------------------------ theme + start */
$("themeBtn").onclick = () => { const r = document.documentElement; const dark = r.dataset.theme ? r.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches; r.dataset.theme = dark ? "light" : "dark"; for (const k in drawn) delete drawn[k]; go(location.hash.slice(1) || "overview", false); };
go(location.hash.slice(1) || "overview");
if (new URLSearchParams(location.search).has("tour")) tourStart();
