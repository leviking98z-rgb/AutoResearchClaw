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
  gpu.innerHTML = `<small>${data.gpu?.allocated || 0} / ${data.gpu?.total || 0} GPUs · target ${(100 * Number(data.gpu?.target_utilization || 0)).toFixed(0)}%</small><div class="bar"><span style="width:${Math.min(100, utilization)}%"></span></div>`;
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
    render(await api("/api/dashboard"));
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
