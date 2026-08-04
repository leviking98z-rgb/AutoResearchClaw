const $ = (id) => document.getElementById(id);
const laneLabels = {
  reservoir: "Reservoir",
  screening: "Screening",
  build: "Build",
  pilot: "Pilot / Repair",
  validation: "Validation",
  paper: "Paper",
  completed: "Completed",
  rejected: "Rejected / Parked",
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

function card(idea) {
  const node = document.createElement("article");
  node.className = "card";
  const title = document.createElement("strong");
  title.textContent = idea.title;
  const meta = document.createElement("small");
  meta.textContent = `${idea.idea_id} · ${idea.family || "unclassified"}`;
  const status = document.createElement("span");
  status.className = "pill";
  status.textContent = `${idea.status || "candidate"} · ${(100 * Number(idea.priority || 0)).toFixed(0)}%`;
  const work = document.createElement("p");
  work.textContent = Object.entries(idea.work_items || {})
    .map(([key, value]) => `${key}:${value}`)
    .join(" · ") || idea.primary_metric || "";
  const timeline = document.createElement("button");
  timeline.type = "button";
  timeline.className = "timeline-button";
  timeline.textContent = "查看日志";
  timeline.addEventListener("click", () => loadIdeaEvents(idea));
  node.append(title, meta, status, work, timeline);
  return node;
}

function eventSummary(event) {
  return Object.entries(event)
    .filter(([key]) => !["timestamp", "type", "factory_id", "idea_id"].includes(key))
    .map(([key, value]) => `${key}=${typeof value === "object" ? JSON.stringify(value) : value}`)
    .join(" · ");
}

async function loadIdeaEvents(idea) {
  const title = $("timeline-title");
  const root = $("timeline");
  const actions = $("log-actions");
  title.textContent = `${idea.title} · ${idea.idea_id}`;
  actions.replaceChildren();
  [
    ["Item Timeline", "events"],
    ["Pipeline Log", "pipeline"],
    ["Pipeline JSONL", "pipeline_events"],
    ["Operational JSONL", "operational_events"],
  ].forEach(([label, source]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.addEventListener("click", () => {
      if (source === "events") loadIdeaEvents(idea);
      else loadIdeaTextLog(idea, source);
    });
    actions.appendChild(button);
  });
  root.textContent = "加载中…";
  try {
    const payload = await api(`/api/ideas/${encodeURIComponent(idea.idea_id)}/events?limit=200`);
    root.replaceChildren();
    (payload.events || []).slice().reverse().forEach((event) => {
      const row = document.createElement("div");
      row.className = "row";
      row.textContent = `${event.timestamp || ""} · ${event.type || "event"} · ${eventSummary(event)}`;
      root.appendChild(row);
    });
    if (!(payload.events || []).length) root.textContent = "暂无 Idea 日志。";
  } catch (error) {
    root.textContent = `读取失败: ${error.message}`;
  }
}

async function loadIdeaTextLog(idea, source) {
  const root = $("timeline");
  root.textContent = "加载中…";
  try {
    const response = await fetch(
      `/api/ideas/${encodeURIComponent(idea.idea_id)}/logs?source=${encodeURIComponent(source)}&limit=600`,
      { cache: "no-store" },
    );
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const pre = document.createElement("pre");
    pre.textContent = await response.text() || "暂无日志。";
    root.replaceChildren(pre);
  } catch (error) {
    root.textContent = `读取失败: ${error.message}`;
  }
}

function render(data) {
  const factory = data.factory || {};
  $("connection").textContent = `${factory.status || "unknown"} · tick ${factory.tick || 0}`;
  const metrics = $("metrics");
  metrics.replaceChildren();
  const values = [
    ["Reservoir", factory.reservoir_size || 0],
    ["Active", Object.entries(factory.ideas_by_status || {}).filter(([key]) =>
      ["screening", "building", "smoke", "pilot", "validating", "paper", "repair"].includes(key)
    ).reduce((sum, [, value]) => sum + value, 0)],
    ["Completed", (factory.ideas_by_status?.completed || 0) + (factory.ideas_by_status?.completed_negative || 0)],
    ["GPU Allocated", data.gpu?.allocated || 0],
  ];
  values.forEach(([label, value]) => {
    const node = document.createElement("article");
    node.innerHTML = `<small>${label}</small><strong>${value}</strong>`;
    metrics.appendChild(node);
  });

  const observability = $("observability");
  observability.replaceChildren();
  const summary = data.observability || {};
  const throughput = summary.throughput || {};
  const latency = summary.latency || {};
  const reliability = summary.reliability || {};
  const outcomes = summary.outcomes || {};
  const budgets = summary.budgets || {};
  [
    ["Idea / h", Number(throughput.ideas_terminal_per_hour || 0).toFixed(2)],
    ["Yield", `${(100 * Number(outcomes.terminal_yield_rate || 0)).toFixed(1)}%`],
    ["Tick p95", `${Number(latency.factory_tick?.p95_sec || 0).toFixed(2)}s`],
    ["Queue p95", `${Number(latency.queue_wait?.p95_sec || 0).toFixed(1)}s`],
    ["Runtime p95", `${Number(latency.work_item_runtime?.p95_sec || 0).toFixed(1)}s`],
    ["Retries", reliability.retries || 0],
    ["GPU hours", Number(budgets.gpu_hours_total || 0).toFixed(2)],
    ["LLM calls", budgets.llm_calls_total || 0],
    ["Repairs", budgets.engineering_repairs_total || 0],
  ].forEach(([label, value]) => {
    const node = document.createElement("article");
    const key = document.createElement("small");
    key.textContent = label;
    const metric = document.createElement("strong");
    metric.textContent = value;
    node.append(key, metric);
    observability.appendChild(node);
  });

  const board = $("board");
  board.replaceChildren();
  Object.entries(laneLabels).forEach(([key, label]) => {
    const lane = document.createElement("section");
    lane.className = "lane";
    const ideas = data.lanes?.[key] || [];
    const heading = document.createElement("h2");
    heading.textContent = `${label} · ${ideas.length}`;
    lane.appendChild(heading);
    ideas.forEach((idea) => lane.appendChild(card(idea)));
    board.appendChild(lane);
  });

  const leases = $("leases");
  leases.replaceChildren();
  (data.gpu?.leases || []).forEach((lease) => {
    const row = document.createElement("div");
    row.className = "row";
    row.textContent = `${lease.status} · ${lease.allocated_gpus} GPU · ${lease.idea_id} / ${lease.item_id}`;
    leases.appendChild(row);
  });
  $("pause").disabled = !data.controls?.can_pause;
  $("resume").disabled = !data.controls?.can_resume;
}

async function refresh() {
  try {
    render(await api("/api/dashboard"));
    const payload = await api("/api/events?limit=80");
    const events = $("events");
    events.replaceChildren();
    (payload.events || []).slice().reverse().forEach((event) => {
      const row = document.createElement("div");
      row.className = "row";
      row.textContent = `${event.timestamp || ""} · ${event.type || "event"} · ${event.idea_id || ""}`;
      events.appendChild(row);
    });
  } catch (error) {
    $("connection").textContent = `连接失败: ${error.message}`;
  }
}

$("refresh").addEventListener("click", refresh);
$("pause").addEventListener("click", async () => {
  await api("/api/control/pause", { method: "POST", body: JSON.stringify({ reason: "dashboard" }) });
  refresh();
});
$("resume").addEventListener("click", async () => {
  await api("/api/control/resume", { method: "POST", body: JSON.stringify({ reason: "dashboard" }) });
  refresh();
});
refresh();
window.setInterval(refresh, 5000);
