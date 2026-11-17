// Receipts web client. Everything that came from the web (titles, quotes, URLs) is untrusted:
// it is HTML-escaped, and only http(s) links are rendered.
"use strict";

const $ = (sel) => document.querySelector(sel);
const sheet = $("#sheet");
const logList = $("#log");
const logState = $("#log-state");
const meter = $("#meter");
const form = $("#ask-form");
const input = $("#q");
const go = $("#go");

const EXAMPLES = {
  research: [
    "Which open models does Nebius Token Factory serve right now, and what do they cost per million tokens?",
    "Did Nebius acquire Tavily, and when did the deal close?",
    "What changed in Python 3.14 that affects performance?",
  ],
  audit: [
    "Nebius acquired Tavily in 2024 for $1 billion. Tavily's API has exactly two endpoints, search and extract.",
  ],
};

const STATUS = {
  answered: "Answered",
  partial: "Partial answer",
  insufficient_evidence: "Not enough evidence",
  tool_failure: "Search failed",
};

let busy = false;
let live = null; // state for the in-progress run

// ------------------------------------------------------------------ helpers
function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
function safeUrl(url) {
  try {
    const u = new URL(url);
    return u.protocol === "http:" || u.protocol === "https:" ? u.href : null;
  } catch { return null; }
}
function link(url, text, cls = "") {
  const href = safeUrl(url);
  if (!href) return esc(text);
  return `<a ${cls ? `class="${cls}"` : ""} href="${esc(href)}" target="_blank" rel="noopener noreferrer">${esc(text)}</a>`;
}
const shortModel = (m) => String(m || "").split("/").pop();
const fmtInt = (n) => Number(n || 0).toLocaleString("en-US");
const mode = () => document.querySelector('input[name="mode"]:checked').value;

// ------------------------------------------------------------------- health
async function checkHealth() {
  try {
    const h = await (await fetch("/api/health")).json();
    const el = $("#health");
    if (!h.ready) {
      el.hidden = false;
      el.textContent = `${h.missing.join(" and ")} not set. Add it to .env and restart the server. Replay still works.`;
    }
  } catch { /* server unreachable: the run itself will report it */ }
}

// -------------------------------------------------------------------- input
function setMode() {
  const audit = mode() === "audit";
  go.textContent = audit ? "Check text" : "Research";
  input.placeholder = audit
    ? "Paste a paragraph, an article or another AI's answer to check claim by claim"
    : "Ask something where a wrong answer would cost you";
  input.rows = audit ? 5 : 2;
  $("#examples").innerHTML = EXAMPLES[audit ? "audit" : "research"]
    .map((q) => `<button type="button" data-q="${esc(q)}">${esc(q)}</button>`).join("");
}
document.querySelectorAll('input[name="mode"]').forEach((r) => r.addEventListener("change", setMode));
$("#examples").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-q]");
  if (!b) return;
  input.value = b.dataset.q;
  input.focus();
});
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) form.requestSubmit();
});

// ---------------------------------------------------------------- streaming
async function consume(response, question) {
  if (!response.ok) {
    let msg = `Server returned ${response.status}.`;
    try { msg = (await response.json()).error || msg; } catch {}
    showError(msg);
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (!line) continue;
      const msg = JSON.parse(line);
      if (msg.replay) live.replay = msg.replay;
      if (msg.event) onEvent(msg.event, question);
      if (msg.result) renderResult(msg.result, live.replay);
      if (msg.error) showError(msg.error);
    }
  }
}

function start(question) {
  busy = true;
  go.disabled = true;
  live = { question, plan: null, checked: 0, verified: 0, replay: null };
  logList.innerHTML = "";
  meter.innerHTML = "";
  logState.textContent = "Working";
  logState.classList.add("running");
  renderWorking();
}
function finish() {
  busy = false;
  go.disabled = false;
  logState.textContent = "Done";
  logState.classList.remove("running");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (busy || text.length < 3) return;
  start(mode() === "audit" ? "Checking the pasted text" : text);
  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: mode(), input: text }),
    });
    await consume(res, text);
  } catch (err) {
    showError(`Could not reach the server: ${err.message}. Is "python src/app.py" running?`);
  } finally { finish(); }
});

$("#replay").addEventListener("click", async () => {
  if (busy) return;
  start("Replaying the last run");
  try { await consume(await fetch("/api/replay"), ""); }
  catch (err) { showError(`Replay failed: ${err.message}`); }
  finally { finish(); }
});

// --------------------------------------------------------------------- log
function addLog(t, html, cls = "") {
  logList.querySelector(".log-empty")?.remove();
  const li = document.createElement("li");
  if (cls) li.className = cls;
  li.innerHTML = `<span class="t">${Number(t || 0).toFixed(1)}s</span><span class="what">${html}</span>`;
  logList.appendChild(li);
  logList.scrollTop = logList.scrollHeight;
}

const VERDICT_WORD = {
  supported: "verified", partial: "partly verified", conflicting: "sources disagree",
  contradicted: "contradicted", unsupported: "not verified",
};

function onEvent(e, question) {
  const tav = '<span class="svc tavily">Tavily</span>';
  const tf = '<span class="svc tf">Token Factory</span>';
  switch (e.type) {
    case "start":
      if (e.question && !live.question.startsWith("Checking")) live.question = e.question;
      addLog(e.t, e.mode === "audit" ? "Started checking the text" : "Started research");
      renderWorking();
      break;
    case "plan":
      live.plan = e.subquestions;
      addLog(e.t, `Planned ${e.subquestions.length} sub-question${e.subquestions.length === 1 ? "" : "s"}` +
        (e.time_range ? `, recent results only (${esc(e.time_range)})` : "") +
        e.subquestions.map((s) => `<span class="detail">${esc(s.id)} ${esc(s.text)}</span>`).join(""));
      renderWorking();
      break;
    case "round":
      if (e.n > 1) addLog(e.t, `Round ${e.n}: ${e.queries.length} new search${e.queries.length === 1 ? "" : "es"}`);
      break;
    case "search":
      addLog(e.t, `${tav} search “${esc(e.query)}”<span class="detail">${e.results} results, ${e.credits} credit${e.credits == 1 ? "" : "s"}, ${(e.ms / 1000).toFixed(1)} s</span>`);
      break;
    case "empty_results":
      addLog(e.t, "Searches returned nothing usable", "bad");
      break;
    case "select":
      addLog(e.t, `Kept ${e.kept_new} new source${e.kept_new === 1 ? "" : "s"}, set aside ${e.dropped} (repeats, copies, same site)`);
      break;
    case "extract":
      addLog(e.t, `${tav} extract: read ${e.ok} of ${e.urls} cited page${e.urls === 1 ? "" : "s"}` + (e.failed ? `<span class="detail bad">${e.failed} could not be read</span>` : ""));
      break;
    case "llm":
      addLog(e.t, `${tf} ${esc(e.role)} with ${esc(shortModel(e.model))}${e.fallback ? " (fallback)" : ""}<span class="detail">${fmtInt(e.prompt_tokens + e.completion_tokens)} tokens, ${(e.ms / 1000).toFixed(1)} s</span>`);
      break;
    case "llm_error":
      addLog(e.t, `${tf} ${esc(shortModel(e.model))} failed${e.action === "fallback" ? ", switching to the fallback model" : ""}<span class="detail">${esc(e.error)}</span>`, "bad");
      break;
    case "llm_note":
      addLog(e.t, `${tf} ${esc(e.note)}`);
      break;
    case "tool_error":
      addLog(e.t, `${tav} ${esc(e.tool)} error${e.action === "retry" ? ", retrying" : ""}<span class="detail">${esc(e.error)}</span>`, "bad");
      break;
    case "verify": {
      live.checked += 1;
      const ok = ["supported", "partial", "conflicting"].includes(e.status);
      if (ok) live.verified += 1;
      addLog(e.t, `Checked ${esc(e.claim)}: <span class="${ok ? "good" : "bad"}">${VERDICT_WORD[e.status] || esc(e.status)}</span>` +
        (e.text ? `<span class="detail">${esc(e.text)}</span>` : ""));
      renderWorking();
      break;
    }
    case "injection_flag":
      addLog(e.t, `${esc(e.source)} contains instructions aimed at AI agents. Treated as data and ignored.<span class="detail">“${esc((e.matches || [])[0])}”</span>`, "bad");
      break;
    case "invalid_citation":
      addLog(e.t, `Dropped citations to sources that don't exist: ${esc((e.ids || []).join(", "))}`, "bad");
      break;
    case "refine":
      addLog(e.t, `Searching again: ${esc(e.reason)}` + (e.new_queries || []).map((q) => `<span class="detail">“${esc(q)}”</span>`).join(""));
      break;
    case "skip_claim":
      addLog(e.t, `Skipped checking ${esc(e.claim)} (${esc(e.reason)})`, "bad");
      break;
    case "plan_fallback": case "draft_failed": case "refine_failed":
      addLog(e.t, `${esc(e.type.replace("_", " "))}: ${esc(e.error)}`, "bad");
      break;
    case "done":
      addLog(e.t, `Finished: ${esc(STATUS[e.status] || e.status)}, ${e.verified} of ${e.total} claims verified`, e.verified ? "good" : "bad");
      break;
    default:
      break;
  }
}

function renderMeter(r) {
  const u = r.usage || {};
  const items = [
    ["Token Factory calls", fmtInt(u.llm_calls)],
    ["Tokens", fmtInt((u.prompt_tokens || 0) + (u.completion_tokens || 0))],
    ["Tavily credits", String(u.tavily_credits ?? 0)],
    ["Time", `${Number(r.elapsed_s || 0).toFixed(1)} s`],
  ];
  if (u.est_llm_cost_usd != null) items.push(["Inference cost", `$${Number(u.est_llm_cost_usd).toFixed(4)}`]);
  meter.innerHTML = items.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("");
}

// ------------------------------------------------------------------- sheet
function renderWorking() {
  const plan = live.plan
    ? `<ul class="subq">${live.plan.map((s) => `<li><span class="dot"></span><span>${esc(s.text)}</span></li>`).join("")}</ul>`
    : "";
  const progress = live.checked
    ? `<p class="muted">Checked ${live.checked} claim${live.checked === 1 ? "" : "s"} against their sources so far, ${live.verified} verified.</p>`
    : `<p class="muted">Searching and reading sources. The log on the right shows every call.</p>`;
  sheet.innerHTML = `<div class="working"><h1>${esc(live.question)}</h1>${progress}${plan}</div>`;
}

function showError(message) {
  sheet.innerHTML = `<h1>${esc(live?.question || "Something went wrong")}</h1><div class="error-box"><p><strong>Stopped.</strong> ${esc(message)}</p></div>`;
  addLog(0, esc(message), "bad");
}

function numbering(r) {
  const order = {};
  const sids = new Set((r.sources || []).map((s) => s.id));
  for (const c of [...(r.claims || []), ...(r.dropped_claims || [])])
    for (const sid of c.source_ids || []) if (sids.has(sid) && !(sid in order)) order[sid] = Object.keys(order).length + 1;
  for (const s of r.sources || []) if (!(s.id in order)) order[s.id] = Object.keys(order).length + 1;
  return order;
}

function renderResult(r, replayName) {
  const num = numbering(r);
  const src = Object.fromEntries((r.sources || []).map((s) => [s.id, s]));
  const facts = (r.claims || []).filter((c) => c.kind === "fact");
  const okFacts = facts.filter((c) => ["supported", "partial", "conflicting"].includes(c.status));
  const totalFacts = facts.length + (r.dropped_claims || []).filter((c) => c.kind === "fact").length;
  const claimIds = new Set((r.claims || []).map((c) => c.id));
  const audit = r.mode === "audit";

  const cite = (c, sid) => {
    const s = src[sid];
    if (!s) return "";
    const ev = (c.evidence || []).find((e) => e.source_id === sid);
    const href = (ev && ev.highlight_url) || s.url;
    const label = ev && ev.highlight_url ? `Source ${num[sid]}, opens highlighted on the quote` : `Source ${num[sid]}`;
    const safe = safeUrl(href);
    return safe ? `<a class="cite" href="${esc(safe)}" target="_blank" rel="noopener noreferrer" aria-label="${esc(label)}" title="${esc(s.domain)}">${num[sid]}</a>` : "";
  };

  const tldr = esc(r.tldr || "").replace(/\[(C\d+)\]/g, (m, id) => (claimIds.has(id) ? `<a class="ref" href="#claim-${id}">${id}</a>` : m));

  let html = `<h1>${esc(audit ? "Checking the pasted text" : r.question)}</h1>`;
  html += `<div class="verdict-line"><span class="verdict ${esc(r.status)}">${esc(STATUS[r.status] || r.status)}</span>`;
  if (totalFacts) html += `<span>${okFacts.length} of ${totalFacts} factual claims verified against the source text</span>`;
  if ((r.conflicts || []).length) html += `<span>${r.conflicts.length} disagreement${r.conflicts.length === 1 ? "" : "s"}</span>`;
  if (replayName) html += `<span class="replay-note">Replay of ${esc(replayName)}</span>`;
  html += `</div>`;

  html += r.status === "insufficient_evidence" || r.status === "tool_failure"
    ? `<div class="error-box"><p>${tldr}</p></div>`
    : `<p class="answer">${tldr}</p>`;

  // Findings
  if ((r.claims || []).length) {
    html += `<h2>${audit ? "Claim by claim" : "What the sources say"}</h2><ol class="findings">`;
    let i = 0;
    for (const c of r.claims) {
      const cites = `<span class="cites">${(c.source_ids || []).map((sid) => cite(c, sid)).join("")}</span>`;
      let body;
      if (c.kind === "inference") {
        body = `<span class="tag-inference">Inference:</span>${esc(c.text)}`;
      } else if (c.status === "supported") {
        body = `<mark class="hl hl-verified" style="--i:${i++}">${esc(c.text)}</mark>`;
      } else if (c.status === "partial") {
        body = `<mark class="hl hl-partial" style="--i:${i++}">${esc(c.text)}</mark>`;
      } else if (c.status === "conflicting") {
        body = `<mark class="hl hl-conflict" style="--i:${i++}">${esc(c.text)}</mark>`;
      } else if (c.status === "contradicted") {
        body = `<s class="struck">${esc(c.text)}</s>`;
      } else {
        body = esc(c.text);
      }
      html += `<li class="finding" id="claim-${esc(c.id)}"><p class="claim"><span class="cid">${esc(c.id)}</span>${body} ${cites}</p>`;
      html += `<p class="claim-meta">${claimMeta(c, num, src)}</p>`;
      const quotes = (c.evidence || []).filter((e) => e.quote && e.verdict !== "does_not_support")
        .sort((a, b) => (a.verdict === "supports" ? -1 : 0) - (b.verdict === "supports" ? -1 : 0));
      if (quotes.length) {
        html += `<ul class="quotes">` + quotes.map((e) => {
          const s = src[e.source_id] || {};
          const verb = { supports: "says", partially_supports: "partly backs it", contradicts: "says otherwise" }[e.verdict] || "";
          const open = e.highlight_url ? link(e.highlight_url, "Open the page at this quote") : "";
          const near = e.quote_match === "near" ? `<span>closest matching sentence on the page</span>` : "";
          return `<li><blockquote class="quote ${esc(e.verdict)}"><q>${esc(e.quote.trim())}</q><span class="src"><span>[${num[e.source_id]}] ${esc(s.domain)} ${verb}</span>${open}${near}</span></blockquote></li>`;
        }).join("") + `</ul>`;
      }
      html += `</li>`;
    }
    html += `</ol>`;
  }

  // Disagreements
  if ((r.conflicts || []).length) {
    html += `<h2>Where sources disagree</h2>`;
    for (const cf of r.conflicts) {
      const side = (ids) => ids.filter((s) => s in num).map((s) => `[${num[s]}] ${esc(src[s]?.domain || "")}`).join(", ");
      html += `<div class="conflict"><p>${esc(cf.claim_text)}</p><p class="muted">Backed by ${side(cf.supporting) || "none"}` +
        (cf.contradicting.length ? `. Contradicted by ${side(cf.contradicting)}` : "") + `. ${esc(cf.note)}</p></div>`;
    }
  }

  // Removed
  if ((r.dropped_claims || []).length) {
    html += `<h2>Removed because no source backed them up</h2><ul class="plain-list removed">` +
      r.dropped_claims.map((c) => `<li><s class="struck">${esc(c.text)}</s><span class="why">${c.status === "contradicted" ? "a cited page contradicts it" : "no cited page contains it"}</span></li>`).join("") + `</ul>`;
  }

  // Gaps
  if ((r.gaps || []).length) {
    html += `<h2>Still unanswered</h2><ul class="plain-list">${r.gaps.map((g) => `<li>${esc(g)}</li>`).join("")}</ul>`;
  }

  // Sources
  if ((r.sources || []).length) {
    const cited = new Set([...(r.claims || []), ...(r.dropped_claims || [])].flatMap((c) => c.source_ids || []));
    html += `<h2>Sources</h2><ol class="sources">` +
      [...r.sources].sort((a, b) => num[a.id] - num[b.id]).map((s) => {
        const notes = [];
        if (s.published_date) notes.push(`published around ${esc(s.published_date)}`);
        if (!cited.has(s.id)) notes.push("read but not cited");
        if (s.fetch_error) notes.push("full page could not be read, checked against the search excerpt");
        const warn = (s.injection_flags || []).length ? `<span class="note warn">This page contains instructions aimed at AI agents. They were ignored.</span>` : "";
        return `<li value="${num[s.id]}">${link(s.url, s.title || s.url)} <span class="domain">${esc(s.domain)}</span>` +
          (notes.length ? `<span class="note">${notes.join(", ")}</span>` : "") + warn + `</li>`;
      }).join("") + `</ol>`;
  }

  // Method
  const u = r.usage || {};
  const models = Object.entries(u.by_model || {}).map(([m, v]) => `<code>${esc(m)}</code> ${v.calls}×`).join(", ");
  html += `<section class="method"><h2>How this was researched</h2>`;
  if (r.plan) html += `<p>Plan: ${r.plan.subquestions.map((s) => `${esc(s.text)}${s.covered ? " (answered)" : " (not answered)"}`).join("; ")}</p>`;
  html += `<p>${r.rounds} round${r.rounds === 1 ? "" : "s"} of search. Stopped because: ${esc(r.stop_reason || "finished")}</p>`;
  if ((r.queries_run || []).length) html += `<p>Searches: ${r.queries_run.map((q) => `“${esc(q)}”`).join(", ")}</p>`;
  if (models) html += `<p>Token Factory models: ${models}</p>`;
  if ((r.errors || []).length) html += `<p class="warn">Errors: ${r.errors.map(esc).join("; ")}</p>`;
  html += `</section>`;

  sheet.innerHTML = html;
  renderMeter(r);
  logList.scrollTop = logList.scrollHeight;
}

function claimMeta(c, num, src) {
  const ids = (sel) => (c.evidence || []).filter(sel).map((e) => `[${num[e.source_id]}]`).join(" ");
  const fully = new Set((c.evidence || []).filter((e) => e.verdict === "supports" && e.quote).map((e) => src[e.source_id]?.domain)).size;
  const partly = (c.evidence || []).some((e) => e.verdict === "partially_supports" && e.quote);
  switch (c.status) {
    case "supported":
      return `<strong>Verified</strong>, ${c.confidence} confidence: ` +
        (fully >= 2 ? `${fully} independent sites say this` : `one site says this`) +
        (partly ? `, another partly backs it` : "");
    case "partial":
      return `<strong>Partly verified</strong>: ${ids((e) => e.verdict === "partially_supports" && e.quote)} backs part of it`;
    case "conflicting":
      return `<strong>Sources disagree</strong>: ${ids((e) => e.verdict !== "contradicts" && e.verdict !== "does_not_support" && e.quote)} backs it, ` +
        `${ids((e) => e.verdict === "contradicts" && e.quote)} says otherwise`;
    case "contradicted":
      return `<strong>Contradicted</strong> by ${ids((e) => e.verdict === "contradicts" && e.quote)}`;
    case "inference":
      return `Conclusion drawn from the verified claims above, not a quoted fact`;
    default:
      return `<strong>Not verified</strong>: no cited page contains a sentence that backs this up`;
  }
}

// --------------------------------------------------------------------- init
setMode();
checkHealth();
logList.innerHTML = `<li class="log-empty">Every search, page read and model call appears here as it happens.</li>`;
