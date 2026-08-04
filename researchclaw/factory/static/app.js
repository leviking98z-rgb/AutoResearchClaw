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
  node.append(title, meta, status, work);
  return node;
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
