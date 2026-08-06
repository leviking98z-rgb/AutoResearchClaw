const $ = (id) => document.getElementById(id);
const laneLabels = {
  reservoir: "Reservoir",
  design: "Design",
  build: "Build",
  pilot: "Pilot",
  scale: "Scale",
  report: "Report",
  completed: "Completed",
  rejected: "Rejected / Quarantine",
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    cache: "no-store",
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function metric(label, value) {
  const node = document.createElement("article");
  const key = document.createElement("small");
  const metric = document.createElement("strong");
  key.textContent = label;
  metric.textContent = value;
  node.append(key, metric);
  return node;
}

function formatNumber(value) {
  return Number(value || 0).toLocaleString();
}

function formatCompact(value) {
  const number = Number(value || 0);
  return new Intl.NumberFormat("en", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(number);
}

function formatMoney(value) {
  if (value === null || value === undefined) return "unpriced";
  return `$${Number(value || 0).toFixed(2)}`;
}

function ratioText(value) {
  return value === null || value === undefined
    ? "not configured"
    : `${(100 * Number(value)).toFixed(1)}%`;
}

function progress(label, value, detail = "") {
  const node = document.createElement("div");
  node.className = "progress-row";
  const head = document.createElement("div");
  head.className = "progress-head";
  const name = document.createElement("span");
  name.textContent = label;
  const amount = document.createElement("strong");
  amount.textContent = detail || ratioText(value);
  head.append(name, amount);
  const bar = document.createElement("div");
  bar.className = "bar";
  const fill = document.createElement("span");
  fill.style.width = `${Math.min(100, Math.max(0, 100 * Number(value || 0)))}%`;
  if (Number(value || 0) >= 0.8) fill.classList.add("critical-fill");
  else if (Number(value || 0) >= 0.5) fill.classList.add("warning-fill");
  bar.append(fill);
  node.append(head, bar);
  return node;
}

function renderBars(root, rows, valueKey, labelKey, formatter = formatCompact) {
  root.replaceChildren();
  const max = Math.max(1, ...rows.map((row) => Number(row[valueKey] || 0)));
  rows.slice(0, 8).forEach((row) => {
    const line = document.createElement("div");
    line.className = "usage-bar-row";
    const label = document.createElement("span");
    label.title = String(row[labelKey] || "");
    label.textContent = String(row[labelKey] || "unknown");
    const rail = document.createElement("div");
    rail.className = "usage-rail";
    const fill = document.createElement("span");
    fill.style.width = `${100 * Number(row[valueKey] || 0) / max}%`;
    rail.append(fill);
    const value = document.createElement("strong");
    value.textContent = formatter(row[valueKey]);
    line.append(label, rail, value);
    root.append(line);
  });
  if (!rows.length) root.textContent = "暂无用量";
}

function svgChart(points, value, { color = "#72e3ff", percent = false } = {}) {
  const namespace = "http://www.w3.org/2000/svg";
  const width = 720;
  const height = 190;
  const pad = 24;
  const values = points.map((point) => Number(value(point) || 0));
  const max = Math.max(percent ? 1 : 0, ...values, 1);
  const coords = values.map((item, index) => {
    const x = pad + (points.length <= 1 ? 0 : index * (width - 2 * pad) / (points.length - 1));
    const y = height - pad - (height - 2 * pad) * item / max;
    return [x, y];
  });
  const svg = document.createElementNS(namespace, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("role", "img");
  [[pad, pad, pad, height - pad], [pad, height - pad, width - pad, height - pad]].forEach((values) => {
    const axis = document.createElementNS(namespace, "line");
    ["x1", "y1", "x2", "y2"].forEach((name, index) => axis.setAttribute(name, values[index]));
    axis.setAttribute("class", "axis");
    svg.append(axis);
  });
  const polyline = document.createElementNS(namespace, "polyline");
  polyline.setAttribute("points", coords.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" "));
  polyline.setAttribute("fill", "none");
  polyline.setAttribute("stroke", color);
  polyline.setAttribute("stroke-width", "3");
  polyline.setAttribute("stroke-linejoin", "round");
  polyline.setAttribute("stroke-linecap", "round");
  svg.append(polyline);
  coords.forEach(([x, y], index) => {
    const circle = document.createElementNS(namespace, "circle");
    circle.setAttribute("cx", x.toFixed(1));
    circle.setAttribute("cy", y.toFixed(1));
    circle.setAttribute("r", "2.5");
    circle.setAttribute("fill", color);
    const title = document.createElementNS(namespace, "title");
    title.textContent = `${new Date(points[index].timestamp).toLocaleString()} · ${percent ? `${(100 * values[index]).toFixed(1)}%` : formatNumber(values[index])}`;
    circle.append(title);
    svg.append(circle);
  });
  if (points.length) {
    [
      [pad, "start", points[0]],
      [width - pad, "end", points.at(-1)],
    ].forEach(([x, anchor, point]) => {
      const label = document.createElementNS(namespace, "text");
      label.setAttribute("x", x);
      label.setAttribute("y", height - 5);
      label.setAttribute("text-anchor", anchor);
      label.textContent = new Date(point.timestamp).toLocaleString();
      svg.append(label);
    });
  }
  return svg;
}

function appendLabeledValue(root, label, value) {
  const wrapper = document.createElement("span");
  const strong = document.createElement("strong");
  const small = document.createElement("small");
  strong.textContent = label;
  small.textContent = value;
  wrapper.append(strong, small);
  root.append(wrapper);
}

function renderUsage(data) {
  if (!data.enabled) {
    $("usage-summary").textContent = "Usage monitoring disabled";
    return;
  }
  const totals = data.llm?.totals || {};
  const burn = data.llm?.burn_rate || {};
  const gpu = data.gpu || {};
  const costs = data.costs || {};
  const budgets = data.budgets || {};
  const accounted = Number(budgets.idea_accounted_token_total || 0);
  const auditTokens = Number(totals.total_tokens || 0);
  $("usage-summary").replaceChildren(
    metric("Audit Tokens", formatNumber(auditTokens)),
    metric("Idea-accounted", formatNumber(accounted)),
    metric("Audit Gap", formatNumber(auditTokens - accounted)),
    metric("Prompt / Output", `${formatCompact(totals.prompt_tokens)} / ${formatCompact(totals.completion_tokens)}`),
    metric("LLM Calls", `${formatNumber(totals.calls)} · ${formatNumber(totals.failed_calls)} failed`),
    metric("Last Hour", formatCompact(burn.tokens_last_hour)),
    metric("GPU Hours", Number(gpu.total_gpu_hours || 0).toFixed(2)),
    metric("Est. Cost", costs.estimated ? formatMoney(costs.total_estimated_usd) : "rates unset"),
  );
  $("burn-rate").textContent = `${formatCompact(burn.tokens_per_hour_24h)}/h · 30d projection ${formatCompact(burn.projected_30d_tokens_at_24h_rate)}`;
  $("token-trend").replaceChildren(svgChart(data.llm?.trend || [], (point) => point.total_tokens));
  const states = gpu.state_minutes || {};
  $("gpu-usage-meta").textContent = `${gpu.allocated_gpus || 0}/${gpu.capacity_gpus || 0} GPUs · ${gpu.pending_jobs || 0} pending · oldest ${Number(gpu.oldest_pending_job_age_minutes || 0).toFixed(0)}m`;
  $("gpu-trend").replaceChildren(svgChart(gpu.trend || [], (point) => point.utilization, { color: "#72f0d0", percent: true }));
  $("gpu-states").replaceChildren(
    metric("Running", `${Number(states.running || 0).toFixed(0)}m`),
    metric("Backlog wait", `${Number(states.backlog_idle || 0).toFixed(0)}m`),
    metric("Idle", `${Number(states.idle || 0).toFixed(0)}m`),
    metric("Unobserved", `${Number(states.unobserved || 0).toFixed(0)}m`),
  );
  $("pricing-coverage").textContent = costs.estimated ? `${(100 * Number(costs.coverage || 0)).toFixed(0)}% priced · estimated` : "configure rates for cost";
  const breakdown = $("usage-breakdown");
  const tier = document.createElement("div");
  const tierTitle = document.createElement("h4");
  tierTitle.textContent = "Tiers";
  const tierRows = document.createElement("div");
  renderBars(tierRows, data.llm?.by_tier || [], "total_tokens", "tier");
  tier.append(tierTitle, tierRows);
  const models = document.createElement("div");
  const modelTitle = document.createElement("h4");
  modelTitle.textContent = "Models";
  const modelRows = document.createElement("div");
  renderBars(modelRows, costs.by_model || [], "total_tokens", "model");
  models.append(modelTitle, modelRows);
  breakdown.replaceChildren(tier, models);
  const monthly = budgets.monthly || {};
  $("monthly-budget-meta").textContent = `${monthly.month || ""} · projected ${formatCompact(monthly.projected_tokens)} tokens`;
  const pressure = $("budget-pressure");
  pressure.replaceChildren(
    progress("Monthly tokens", monthly.tokens_ratio, `${formatCompact(monthly.tokens_used)} / ${monthly.tokens_budget ? formatCompact(monthly.tokens_budget) : "unset"}`),
    progress("Monthly GPU", monthly.gpu_hours_ratio, `${Number(monthly.gpu_hours_used || 0).toFixed(2)} / ${monthly.gpu_hours_budget || "unset"} GPU-h`),
    progress("Monthly estimated cost", monthly.cost_ratio, `${costs.estimated ? formatMoney(monthly.estimated_cost_usd) : "unpriced"} / ${monthly.cost_budget_usd ? formatMoney(monthly.cost_budget_usd) : "unset"}`),
  );
  (data.budgets?.ideas_near_limit || []).slice(0, 5).forEach((idea) => {
    pressure.append(progress(
      idea.title,
      Math.max(Number(idea.token_ratio || 0), Number(idea.gpu_ratio || 0)),
      `${(100 * Math.max(Number(idea.token_ratio || 0), Number(idea.gpu_ratio || 0))).toFixed(0)}%`,
    ));
  });
  const alerts = $("usage-alerts");
  alerts.replaceChildren();
  (data.alerts || []).forEach((alert) => {
    const row = document.createElement("button");
    row.className = `alert ${alert.severity}`;
    appendLabeledValue(
      row,
      String(alert.title || ""),
      String(alert.message || ""),
    );
    if (alert.idea_id) row.addEventListener("click", () => loadIdea(alert.idea_id));
    alerts.append(row);
  });
  if (!(data.alerts || []).length) alerts.textContent = "暂无告警";
  $("alert-count").textContent = `${(data.alerts || []).length} active`;
  const ideas = $("usage-ideas");
  ideas.replaceChildren();
  (data.ideas || []).slice(0, 12).forEach((idea) => {
    const row = document.createElement("button");
    row.className = "usage-idea";
    appendLabeledValue(
      row,
      String(idea.title || ""),
      `${idea.status || ""} · ${idea.family || ""}`,
    );
    const value = document.createElement("span");
    value.textContent = `${formatCompact(idea.llm_tokens)} tokens · ${Number(idea.gpu_hours || 0).toFixed(2)} GPU-h`;
    row.append(value);
    row.addEventListener("click", () => loadIdea(idea.idea_id));
    ideas.append(row);
  });
}

function card(idea) {
  const node = document.createElement("article");
  node.className = "card";
  const title = document.createElement("strong");
  title.textContent = idea.title;
  const meta = document.createElement("small");
  meta.textContent = `${idea.family} · score ${Number(idea.score || 0).toFixed(2)}`;
  const pill = document.createElement("span");
  pill.className = "pill";
  pill.textContent = idea.status;
  const usage = document.createElement("p");
  usage.textContent = `${Number(idea.gpu_seconds_spent || 0) / 3600.0} GPU-h · ${idea.llm_tokens_spent || 0} tokens · ${idea.jobs?.length || 0} jobs`;
  const button = document.createElement("button");
  button.textContent = "查看";
  button.addEventListener("click", () => loadIdea(idea.idea_id));
  node.append(title, meta, pill, usage, button);
  return node;
}

function render(data) {
  const controller = data.controller || {};
  $("connection").textContent = `${controller.status || "unknown"} · ${controller.timestamp || ""}`;
  const metrics = $("metrics");
  metrics.replaceChildren(
    metric("Reservoir", controller.ideas_by_status?.reservoir || 0),
    metric("Active", ["designing", "building", "piloting", "scaling", "reporting"].reduce((sum, key) => sum + Number(controller.ideas_by_status?.[key] || 0), 0)),
    metric("Completed", Number(controller.ideas_by_status?.completed || 0) + Number(controller.ideas_by_status?.completed_negative || 0)),
    metric("GPU Util", `${(100 * Number(data.gpu?.utilization || 0)).toFixed(1)}%`),
    metric("GPU Hours", Number(controller.gpu_hours_total || 0).toFixed(2)),
    metric("LLM Tokens", Number(controller.llm_tokens_total || 0).toLocaleString()),
  );
  const board = $("board");
  board.replaceChildren();
  Object.entries(laneLabels).forEach(([key, label]) => {
    const lane = document.createElement("section");
    lane.className = "lane";
    const values = data.lanes?.[key] || [];
    const heading = document.createElement("h2");
    heading.textContent = `${label} · ${values.length}`;
    lane.append(heading, ...values.map(card));
    board.append(lane);
  });
  const gpu = $("gpu");
  const utilization = 100 * Number(data.gpu?.utilization || 0);
  const gpuMeta = document.createElement("small");
  gpuMeta.textContent = `${data.gpu?.allocated || 0} / ${data.gpu?.total || 0} GPUs · target ${(100 * Number(data.gpu?.target_utilization || 0)).toFixed(0)}%`;
  const gpuBar = document.createElement("div");
  gpuBar.className = "bar";
  const gpuFill = document.createElement("span");
  gpuFill.style.width = `${Math.min(100, utilization)}%`;
  gpuBar.append(gpuFill);
  gpu.replaceChildren(gpuMeta, gpuBar);
  (data.gpu?.jobs || []).forEach((job) => {
    const row = document.createElement("div");
    row.className = "row";
    row.textContent = `${job.kind} · ${job.result?.allocated_gpus || 0} GPU · ${job.idea_id}`;
    gpu.append(row);
  });
  $("pause").disabled = !data.controls?.can_pause;
  $("resume").disabled = !data.controls?.can_resume;
  $("stop").disabled = !data.controls?.can_stop;
}

async function loadIdea(ideaId) {
  const payload = await api(`/api/ideas/${encodeURIComponent(ideaId)}`);
  const idea = payload.idea;
  $("detail-title").textContent = `${idea.title} · ${idea.idea_id}`;
  const root = $("detail");
  const left = document.createElement("pre");
  left.textContent = JSON.stringify({ idea, jobs: payload.jobs, attempts: payload.attempts }, null, 2);
  const right = document.createElement("pre");
  right.textContent = (payload.events || []).map((event) => `${event.timestamp} · ${event.event_type} · ${JSON.stringify(event)}`).join("\n");
  const grid = document.createElement("div");
  grid.className = "grid";
  grid.append(left, right);
  root.replaceChildren(grid);
  const actions = $("log-actions");
  actions.replaceChildren();
  (payload.attempts || []).slice(-8).reverse().forEach((attempt) => {
    ["stdout", "stderr", "attempt"].forEach((stream) => {
      const button = document.createElement("button");
      button.textContent = `${attempt.number} · ${stream}`;
      button.addEventListener("click", () => loadLog(ideaId, attempt.attempt_id, stream));
      actions.append(button);
    });
  });
}

async function loadLog(ideaId, attemptId, stream) {
  const response = await fetch(`/api/ideas/${encodeURIComponent(ideaId)}/logs?attempt_id=${encodeURIComponent(attemptId)}&stream=${stream}&limit=1200`, { cache: "no-store" });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  const pre = document.createElement("pre");
  pre.textContent = await response.text() || "暂无日志";
  $("detail").replaceChildren(pre);
}

async function refresh() {
  try {
    const usageHours = Number($("usage-window")?.value || 168);
    const [dashboard, usage] = await Promise.all([
      api("/api/dashboard"),
      api(`/api/usage?hours=${usageHours}`),
    ]);
    render(dashboard);
    renderUsage(usage);
    const payload = await api("/api/events?limit=80");
    const events = $("events");
    events.replaceChildren();
    (payload.events || []).slice().reverse().forEach((event) => {
      const row = document.createElement("div");
      row.className = "row";
      row.textContent = `${event.timestamp} · ${event.event_type} · ${event.idea_id || ""}`;
      events.append(row);
    });
  } catch (error) {
    $("connection").textContent = `连接失败: ${error.message}`;
  }
}

$("refresh").addEventListener("click", refresh);
$("usage-window").addEventListener("change", refresh);
$("pause").addEventListener("click", async () => {
  await api("/api/control/pause", { method: "POST", body: JSON.stringify({ reason: "dashboard" }) });
  refresh();
});
$("resume").addEventListener("click", async () => {
  await api("/api/control/resume", { method: "POST", body: JSON.stringify({ reason: "dashboard" }) });
  refresh();
});
$("stop").addEventListener("click", async () => {
  if (!window.confirm("确认停止 AutoResearch v2 controller？")) return;
  await api("/api/control/stop", { method: "POST", body: JSON.stringify({ reason: "dashboard" }) });
  refresh();
});
refresh();
window.setInterval(refresh, 4000);
