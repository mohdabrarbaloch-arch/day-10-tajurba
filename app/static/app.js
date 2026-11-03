/* Tajurba frontend — vanilla JS, no build step.
   State is deliberately tiny: a token, a user, and whatever the current
   view needs. Anything that survives a reload lives in localStorage. */

const API = "";
const TOKEN_KEY = "tajurba_token";
const USER_KEY = "tajurba_user";

const state = {
  token: localStorage.getItem(TOKEN_KEY) || null,
  user: JSON.parse(localStorage.getItem(USER_KEY) || "null"),
  datasets: [],
  runs: [],
  currentDataset: null,
  currentRun: null,
};

/* ── Helpers ─────────────────────────────────────────────────── */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

let toastTimer = null;
function toast(message, bad = false) {
  const el = $("#toast");
  el.textContent = message;
  el.classList.toggle("bad", bad);
  el.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), 3600);
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  if (options.body && !(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  const res = await fetch(API + path, { ...options, headers });

  if (res.status === 401 && state.token) {
    signOut();
    throw new Error("Session expired. Please sign in again.");
  }

  if (res.status === 204) return null;

  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }

  if (!res.ok) {
    let detail = "Request failed.";
    if (data && typeof data === "object") {
      if (typeof data.detail === "string") detail = data.detail;
      else if (Array.isArray(data.detail)) {
        detail = data.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
      }
    }
    throw new Error(detail);
  }
  return data;
}

function fmtDate(iso) {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) +
    " · " + d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

function fmtNum(n, digits = 4) {
  if (n === null || n === undefined) return "—";
  return Number(n).toFixed(digits);
}

/* ── Auth ────────────────────────────────────────────────────── */

function showAuth() {
  $("#auth-screen").classList.remove("hidden");
  $("#app").classList.add("hidden");
}

function showApp() {
  $("#auth-screen").classList.add("hidden");
  $("#app").classList.remove("hidden");
  $("#user-name").textContent = state.user ? state.user.name : "";
  loadSamples();
  loadDatasets();
}

function signOut() {
  state.token = null;
  state.user = null;
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
  showAuth();
}

function persist(token, user) {
  state.token = token;
  state.user = user;
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

$$(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    $$(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    const isLogin = tab.dataset.tab === "login";
    $("#login-form").classList.toggle("hidden", !isLogin);
    $("#register-form").classList.toggle("hidden", isLogin);
    $("#auth-error").classList.add("hidden");
  });
});

$("#login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = new FormData(e.target);
  const err = $("#auth-error");
  err.classList.add("hidden");
  try {
    const data = await api("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({
        email: form.get("email"),
        password: form.get("password"),
      }),
    });
    persist(data.access_token, data.user);
    showApp();
  } catch (ex) {
    err.textContent = ex.message;
    err.classList.remove("hidden");
  }
});

$("#register-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = new FormData(e.target);
  const err = $("#auth-error");
  err.classList.add("hidden");
  try {
    const data = await api("/api/auth/register", {
      method: "POST",
      body: JSON.stringify({
        name: form.get("name"),
        email: form.get("email"),
        password: form.get("password"),
      }),
    });
    persist(data.access_token, data.user);
    showApp();
  } catch (ex) {
    err.textContent = ex.message;
    err.classList.remove("hidden");
  }
});

$("#logout").addEventListener("click", signOut);

/* ── Navigation ──────────────────────────────────────────────── */

function showView(name) {
  ["datasets", "runs", "dataset", "run"].forEach((v) => {
    $(`#view-${v}`).classList.toggle("hidden", v !== name);
  });
  $$(".nav-btn").forEach((b) => {
    const target = b.dataset.view;
    b.classList.toggle("active", target === name || (name === "dataset" && target === "datasets") || (name === "run" && target === "runs"));
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

$$(".nav-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const view = btn.dataset.view;
    showView(view);
    if (view === "datasets") loadDatasets();
    if (view === "runs") loadRuns();
  });
});

$$("[data-back]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.back;
    showView(target);
    if (target === "datasets") loadDatasets();
    if (target === "runs") loadRuns();
  });
});

/* ── Samples ─────────────────────────────────────────────────── */

async function loadSamples() {
  try {
    const samples = await api("/api/samples");
    const wrap = $("#samples");
    wrap.innerHTML = samples
      .filter((s) => s.available)
      .map(
        (s) => `
      <button class="sample" data-sample="${esc(s.key)}">
        <div class="sample-title">
          ${esc(s.title)}
          <span class="pill ${s.task === "classification" ? "cls" : "reg"}">${esc(s.task)}</span>
        </div>
        <div class="sample-blurb">${esc(s.blurb)}</div>
      </button>`
      )
      .join("");

    wrap.querySelectorAll("[data-sample]").forEach((btn) => {
      btn.addEventListener("click", () => loadSample(btn.dataset.sample));
    });
  } catch (ex) {
    console.error(ex);
  }
}

async function loadSample(key) {
  try {
    toast("Loading sample…");
    const res = await fetch(`/api/samples/${key}`);
    if (!res.ok) throw new Error("Could not fetch the sample.");
    const text = await res.text();
    const blob = new Blob([text], { type: "text/csv" });
    const file = new File([blob], `${key}.csv`, { type: "text/csv" });
    await uploadFile(file);
  } catch (ex) {
    toast(ex.message, true);
  }
}

/* ── Datasets ────────────────────────────────────────────────── */

$("#file-input").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (file) await uploadFile(file);
  e.target.value = "";
});

async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  try {
    toast("Uploading…");
    const ds = await api("/api/datasets", { method: "POST", body: form });
    toast(`Uploaded ${ds.n_rows} rows.`);
    await loadDatasets();
    openDataset(ds.id);
  } catch (ex) {
    toast(ex.message, true);
  }
}

async function loadDatasets() {
  try {
    state.datasets = await api("/api/datasets");
    const wrap = $("#dataset-list");
    const empty = $("#datasets-empty");

    if (!state.datasets.length) {
      wrap.innerHTML = "";
      empty.classList.remove("hidden");
      return;
    }
    empty.classList.add("hidden");

    wrap.innerHTML = state.datasets
      .map(
        (d) => `
      <div class="card" data-dataset="${esc(d.id)}">
        <div class="card-title">${esc(d.name)}</div>
        <div class="card-meta">
          <span>${d.n_rows.toLocaleString()} rows</span>
          <span>${d.n_columns} cols</span>
          <span>${fmtDate(d.created_at)}</span>
        </div>
      </div>`
      )
      .join("");

    wrap.querySelectorAll("[data-dataset]").forEach((card) => {
      card.addEventListener("click", () => openDataset(card.dataset.dataset));
    });
  } catch (ex) {
    toast(ex.message, true);
  }
}

async function openDataset(id) {
  try {
    const ds = await api(`/api/datasets/${id}`);
    state.currentDataset = ds;

    $("#ds-title").textContent = ds.name;
    $("#ds-meta").textContent =
      `${ds.n_rows.toLocaleString()} rows · ${ds.n_columns} columns · ${ds.filename}`;

    const cols = ds.columns;
    const table = $("#ds-preview");
    table.innerHTML =
      `<thead><tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>` +
      `<tbody>${ds.preview
        .map((row) => `<tr>${cols.map((c) => `<td>${esc(row[c])}</td>`).join("")}</tr>`)
        .join("")}</tbody>`;

    const select = $("#target-select");
    select.innerHTML = cols
      .map((c) => `<option value="${esc(c)}">${esc(c)}</option>`)
      .join("");
    if (ds.suggested_target) select.value = ds.suggested_target;
    if (ds.suggested_task) $("#task-select").value = ds.suggested_task;

    showView("dataset");
  } catch (ex) {
    toast(ex.message, true);
  }
}

$("#train-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = new FormData(e.target);
  const btn = $("#train-btn");
  const original = btn.textContent;

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Training…';

  try {
    const run = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify({
        dataset_id: state.currentDataset.id,
        target: form.get("target"),
        task: form.get("task"),
        test_fraction: parseFloat(form.get("test_fraction")),
      }),
    });
    toast("Run complete.");
    openRun(run.id);
  } catch (ex) {
    toast(ex.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = original;
  }
});

/* ── Runs ────────────────────────────────────────────────────── */

async function loadRuns() {
  try {
    state.runs = await api("/api/runs");
    const wrap = $("#run-list");
    const empty = $("#runs-empty");

    if (!state.runs.length) {
      wrap.innerHTML = "";
      empty.classList.remove("hidden");
      return;
    }
    empty.classList.add("hidden");

    wrap.innerHTML = state.runs
      .map(
        (r) => `
      <div class="card" data-run="${esc(r.id)}">
        <div class="card-title">${esc(r.target)} <span class="pill ${r.task === "classification" ? "cls" : "reg"}">${esc(r.task)}</span></div>
        <div class="card-meta">
          <span>${esc(r.winner_name || "no winner")}</span>
          <span>${fmtDate(r.created_at)}</span>
        </div>
        <div class="card-score">
          <span class="score-big">${fmtNum(r.primary_score)}</span>
          <span class="score-label">${esc(r.primary_metric)} · baseline ${fmtNum(r.baseline_score)}</span>
        </div>
      </div>`
      )
      .join("");

    wrap.querySelectorAll("[data-run]").forEach((card) => {
      card.addEventListener("click", () => openRun(card.dataset.run));
    });
  } catch (ex) {
    toast(ex.message, true);
  }
}

async function openRun(id) {
  try {
    const run = await api(`/api/runs/${id}`);
    state.currentRun = run;

    $("#run-title").textContent = `Predicting ${run.target}`;
    $("#run-meta").textContent =
      `${run.task} · ${run.n_rows.toLocaleString()} rows · ${run.n_features} features · ` +
      `train/test ${run.train_rows}/${run.test_rows} · ${run.duration_seconds}s`;

    // Notes
    const notes = $("#run-notes");
    notes.innerHTML = (run.notes || [])
      .map((n) => `<div class="note">${esc(n)}</div>`)
      .join("");

    // Stats
    const beat = run.primary_score !== null && run.baseline_score !== null
      ? run.primary_score - run.baseline_score
      : null;
    $("#run-stats").innerHTML = `
      <div class="stat">
        <div class="stat-label">${esc(run.primary_metric)}</div>
        <div class="stat-value ${beat !== null && beat > 0 ? "good" : "bad"}">${fmtNum(run.primary_score)}</div>
        <div class="stat-note">best model</div>
      </div>
      <div class="stat">
        <div class="stat-label">baseline</div>
        <div class="stat-value">${fmtNum(run.baseline_score)}</div>
        <div class="stat-note">what you must beat</div>
      </div>
      <div class="stat">
        <div class="stat-label">lift</div>
        <div class="stat-value ${beat !== null && beat > 0 ? "good" : "bad"}">${beat === null ? "—" : (beat > 0 ? "+" : "") + fmtNum(beat)}</div>
        <div class="stat-note">over baseline</div>
      </div>
      <div class="stat">
        <div class="stat-label">winner</div>
        <div class="stat-value" style="font-size:15px;font-family:var(--mono);line-height:1.4">${esc(run.winner_name || "none")}</div>
        <div class="stat-note">${esc(run.winner_family || "")}</div>
      </div>`;

    // Leaderboard
    const metricKeys = run.task === "classification"
      ? ["accuracy", "macro_f1"]
      : ["r2", "rmse", "mae"];

    const rows = run.leaderboard
      .slice()
      .sort((a, b) => {
        const av = a.scores[run.primary_metric];
        const bv = b.scores[run.primary_metric];
        if (av === undefined) return 1;
        if (bv === undefined) return -1;
        return bv - av;
      });

    const head = `<thead><tr><th></th><th>Model</th>${metricKeys
      .map((k) => `<th>${esc(k)}</th>`)
      .join("")}<th>Time</th></tr></thead>`;

    const body = rows
      .map((c, i) => {
        const isWinner = c.name === run.winner_name;
        const cls = c.error ? "failed" : isWinner ? "winner" : "";
        const cells = metricKeys
          .map((k) => `<td class="num">${c.scores[k] !== undefined ? fmtNum(c.scores[k]) : "—"}</td>`)
          .join("");
        return `<tr class="${cls}">
          <td class="rank">${c.error ? "!" : i + 1}</td>
          <td>${esc(c.name)}${c.error ? ` — ${esc(c.error)}` : ""}</td>
          ${cells}
          <td class="num">${c.train_seconds}s</td>
        </tr>`;
      })
      .join("");

    const baselineCells = metricKeys
      .map((k) => `<td class="num">${run.baseline[k] !== undefined ? fmtNum(run.baseline[k]) : "—"}</td>`)
      .join("");

    $("#leaderboard").innerHTML =
      head +
      `<tbody>${body}<tr class="baseline">
        <td class="rank">—</td><td>baseline</td>${baselineCells}<td class="num">—</td>
      </tr></tbody>`;

    // Feature list
    $("#feature-list").innerHTML = (run.feature_names || [])
      .map((f) => `<span class="feature-chip">${esc(f)}</span>`)
      .join("");

    // Predict form — built from the dataset's source columns
    const ds = await api(`/api/datasets/${run.dataset_id}`);
    const inputs = ds.columns
      .filter((c) => c !== run.target)
      .map(
        (c) => `<label>${esc(c)}<input name="${esc(c)}" placeholder="value" /></label>`
      )
      .join("");
    $("#predict-form").innerHTML =
      `<div class="row">${inputs}</div><button type="submit" class="btn primary">Predict</button>`;
    $("#predict-result").classList.add("hidden");

    $("#predict-form").onsubmit = async (e) => {
      e.preventDefault();
      const form = new FormData(e.target);
      const row = {};
      form.forEach((v, k) => { row[k] = v; });
      try {
        const result = await api(`/api/runs/${run.id}/predict`, {
          method: "POST",
          body: JSON.stringify({ row }),
        });
        const probs = result.probabilities
          ? Object.entries(result.probabilities)
              .sort((a, b) => b[1] - a[1])
              .map(([k, v]) => `<div class="prob-row"><span>${esc(k)}</span><span>${(v * 100).toFixed(1)}%</span></div>`)
              .join("")
          : "";
        const box = $("#predict-result");
        box.innerHTML = `
          <div class="predict-label">prediction</div>
          <div class="predict-value">${esc(result.prediction)}</div>
          ${probs}`;
        box.classList.remove("hidden");
      } catch (ex) {
        toast(ex.message, true);
      }
    };

    showView("run");
  } catch (ex) {
    toast(ex.message, true);
  }
}

/* ── Boot ────────────────────────────────────────────────────── */

(async function boot() {
  if (!state.token) {
    showAuth();
    return;
  }
  try {
    const user = await api("/api/auth/me");
    state.user = user;
    localStorage.setItem(USER_KEY, JSON.stringify(user));
    showApp();
  } catch {
    signOut();
  }
})();
