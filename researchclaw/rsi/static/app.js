const state = {
  dashboard: null,
  activeTab: "logs",
  logSource: "pipeline",
  refreshTimer: null,
  logsTimer: null,
};

const $ = (id) => document.getElementById(id);

const STAGE_LABELS = {
  TOPIC_INIT: "题目初始化",
  PROBLEM_DECOMPOSE: "问题拆解",
  SEARCH_STRATEGY: "检索策略",
  LITERATURE_COLLECT: "文献收集",
  LITERATURE_SCREEN: "文献筛选",
  KNOWLEDGE_EXTRACT: "知识提取",
  SYNTHESIS: "知识综合",
  HYPOTHESIS_GEN: "假设生成",
  EXPERIMENT_DESIGN: "实验设计",
  CODE_GENERATION: "代码生成",
  RESOURCE_PLANNING: "资源规划",
  EXPERIMENT_RUN: "实验执行",
  ITERATIVE_REFINE: "迭代修复",
  RESULT_ANALYSIS: "结果分析",
  RESEARCH_DECISION: "研究决策",
  PAPER_OUTLINE: "论文提纲",
  PAPER_DRAFT: "论文初稿",
  PEER_REVIEW: "同行评审",
  PAPER_REVISION: "论文修订",
  QUALITY_GATE: "质量门禁",
  KNOWLEDGE_ARCHIVE: "知识归档",
  EXPORT_PUBLISH: "导出产物",
  CITATION_VERIFY: "引用核验",
};

const STATUS_LABELS = {
  running: "运行中",
  paused: "已暂停",
  pausing: "正在暂停",
  stopped: "已停止",
  crashed: "异常退出",
  completed: "已完成",
  degraded: "部分降级",
  ok: "正常",
  fail: "故障",
  failed: "失败",
  starting: "启动中",
  finished: "已完成",
  timed_out: "超时",
  unknown: "未知",
};

function text(id, value, fallback = "--") {
  const node = $(id);
  if (node) node.textContent = value ?? fallback;
}

function statusClass(value) {
  const normalized = String(value || "unknown").toLowerCase();
  if (["running", "ok", "finished", "completed"].includes(normalized)) return normalized;
  if (["degraded", "starting", "timed_out", "pausing"].includes(normalized)) return normalized;
  if (["fail", "failed", "crashed", "stopped", "offline"].includes(normalized)) return normalized;
  if (normalized.startsWith("paused")) return "paused";
  return "neutral";
}

function setStatusPill(id, value, label) {
  const node = $(id);
  if (!node) return;
  node.className = `status-pill ${statusClass(value)}`;
  node.textContent = label || STATUS_LABELS[value] || value || "未知";
}

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return "--";
  const value = Math.max(0, Number(seconds));
  const days = Math.floor(value / 86400);
  const hours = Math.floor((value % 86400) / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  if (days) return `${days}天 ${hours}小时`;
  if (hours) return `${hours}小时 ${minutes}分`;
  return `${minutes}分钟`;
}

function formatAge(seconds) {
  if (seconds === null || seconds === undefined) return "--";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

function formatTime(value) {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString("zh-CN", { hour12: false });
}

function truncate(value, limit = 160) {
  const raw = String(value || "").replace(/\s+/g, " ").trim();
  return raw.length > limit ? `${raw.slice(0, limit)}…` : raw;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    cache: "no-store",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}

function showToast(message, error = false) {
  const node = $("toast");
  node.textContent = message;
  node.className = `toast${error ? " error" : ""}`;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => node.classList.add("hidden"), 4500);
}

function renderDashboard(data) {
  state.dashboard = data;
  const campaign = data.campaign || {};
  const progress = data.progress || {};
  const current = progress.current_stage || {};
  const checkpoint = progress.checkpoint || {};
  const topic = data.topic || {};
  const experiment = data.experiment || {};
  const monitor = data.monitor || {};
  const infra = monitor.infrastructure || {};
  const controls = data.controls || {};

  setStatusPill("campaign-status", campaign.status);
  text("campaign-id", campaign.id);
  text("topic-title", topic.title || "尚未选定研究题目");
  text("research-question", topic.research_question || topic.hypothesis || "");
  text("cycle-value", campaign.cycle ? `Cycle ${campaign.cycle}` : "--");
  text(
    "current-stage",
    current.number
      ? `${String(current.number).padStart(2, "0")} · ${STAGE_LABELS[current.name] || current.name}`
      : "--",
  );
  text(
    "checkpoint-stage",
    checkpoint.number
      ? `${String(checkpoint.number).padStart(2, "0")} · ${STAGE_LABELS[checkpoint.name] || checkpoint.name}`
      : "--",
  );
  text("runtime-value", formatDuration(campaign.runtime_sec));
  text("autonomy-state", campaign.continuous ? "24×7 连续迭代" : "有限循环");
  text(
    "control-hint",
    campaign.status === "running"
      ? "系统正在自主推进；暂停会在安全控制点终止当前子任务。"
      : campaign.status === "pausing"
        ? "暂停请求已提交，正在等待当前子任务协作退出。"
        : "Supervisor 当前未运行，可从这里恢复。",
  );

  $("pause-button").disabled = !controls.can_pause;
  $("resume-button").disabled = !controls.can_resume;

  text("supervisor-metric", campaign.supervisor_alive ? "在线" : "离线");
  text(
    "supervisor-detail",
    `PID ${campaign.supervisor_pid || "--"} · heartbeat ${formatAge(campaign.heartbeat_age_sec)}前`,
  );
  text(
    "gpu-metric",
    infra.gpu_total ? `${infra.gpu_total} × H20` : "等待探测",
  );
  text(
    "gpu-detail",
    `${infra.alive_nodes || "--"} 节点 · Ray ${infra.ray_started ? "ready" : "unknown"} · 可用 ${infra.gpu_available ?? "--"} GPU`,
  );
  text("task-metric", experiment.task_id || "暂无任务");
  text(
    "task-detail",
    experiment.progress?.condition
      ? `${experiment.progress.condition} · Seed ${experiment.progress.seed || "--"}`
      : STATUS_LABELS[experiment.state] || experiment.state || "--",
  );
  text("cycle-metric", `${campaign.completed_cycles || 0} completed`);
  text(
    "cycle-detail",
    `${campaign.successful_cycles || 0} 成功 · ${campaign.failed_cycles || 0} 失败 · score ${campaign.last_score ?? "--"}`,
  );

  renderStages(progress.stages || [], current);
  renderExperiment(experiment, topic);
  renderHealth(monitor);
  $("last-refresh").textContent = `刷新于 ${new Date().toLocaleTimeString("zh-CN", { hour12: false })}`;
}

function renderStages(stages, current) {
  const root = $("stage-track");
  root.classList.remove("skeleton-block");
  root.replaceChildren();
  stages.forEach((stage) => {
    const node = document.createElement("div");
    node.className = `stage ${stage.status}${stage.checkpoint ? " checkpoint" : ""}`;
    node.title = `${stage.number}. ${stage.name} · ${stage.status}`;

    const number = document.createElement("span");
    number.className = "stage-number";
    number.textContent = String(stage.number).padStart(2, "0");

    const name = document.createElement("span");
    name.className = "stage-name";
    name.textContent = STAGE_LABELS[stage.name] || stage.name;

    node.append(number, name);
    root.appendChild(node);
  });

  const rollback = current.rollback;
  const banner = $("rollback-banner");
  const checkpointNumber = state.dashboard?.progress?.checkpoint?.number || 0;
  if (rollback && Number(current.number || 0) <= Number(checkpointNumber)) {
    banner.textContent = `研究决策为 ${rollback.decision}，流水线已回滚到 ${STAGE_LABELS[rollback.target] || rollback.target}。Checkpoint 保留在后续阶段，因此两者会同时显示。`;
    banner.classList.remove("hidden");
  } else {
    banner.classList.add("hidden");
  }
}

function renderExperiment(experiment, topic) {
  setStatusPill("task-state", experiment.state || "unknown");
  text("task-id", experiment.task_id);
  text("task-condition", experiment.progress?.condition);
  text("task-seed", experiment.progress?.seed);
  text("task-activity", truncate(experiment.progress?.activity, 180));
  text("topic-hypothesis", topic.hypothesis);
  text("topic-metric", topic.primary_metric);

  const download = experiment.progress?.download;
  const box = $("download-box");
  if (download) {
    box.classList.remove("hidden");
    text("download-percent", `${Number(download.percent).toFixed(1)}%`);
    text("download-detail", `${download.downloaded} / ${download.total}`);
    $("download-bar").style.width = `${Math.max(0, Math.min(100, download.percent))}%`;
  } else {
    box.classList.add("hidden");
  }
}

function renderHealth(monitor) {
  setStatusPill(
    "monitor-overall",
    monitor.overall,
    monitor.core_alive && monitor.overall === "degraded" ? "核心链路存活" : undefined,
  );
  const root = $("health-list");
  root.classList.remove("skeleton-block");
  root.replaceChildren();
  (monitor.checks || []).forEach((check) => {
    const row = document.createElement("div");
    row.className = "health-item";

    const light = document.createElement("span");
    light.className = `health-light ${check.status || "unknown"}`;

    const name = document.createElement("span");
    name.className = "health-name";
    name.textContent = check.name;

    const detail = document.createElement("span");
    detail.className = "health-detail";
    detail.textContent = check.detail || "--";
    detail.title = check.detail || "";

    const age = document.createElement("span");
    age.className = "health-age";
    age.textContent = check.observed_at ? formatTime(check.observed_at).split(" ").pop() : "--";

    row.append(light, name, detail, age);
    root.appendChild(row);
  });
  text(
    "health-explainer",
    monitor.core_alive && monitor.overall === "degraded"
      ? "Monitor 的 degraded 主要来自慢速外部探针或陈旧 checkpoint；Supervisor、Pipeline progress 与 Ray Pool 仍可独立证明任务存活。"
      : "",
  );
}

async function refreshDashboard() {
  const connection = $("connection");
  try {
    const data = await api("/api/dashboard");
    renderDashboard(data);
    connection.className = "connection online";
    connection.innerHTML = '<span class="pulse-dot"></span>已连接';
  } catch (error) {
    connection.className = "connection offline";
    connection.innerHTML = '<span class="pulse-dot"></span>连接失败';
    showToast(`读取状态失败：${error.message}`, true);
  }
}

async function refreshLogs() {
  if (state.activeTab !== "logs") return;
  try {
    const payload = await api(`/api/logs?source=${encodeURIComponent(state.logSource)}&tail=240`);
    const output = $("log-output");
    const nearBottom = output.scrollHeight - output.scrollTop - output.clientHeight < 80;
    output.textContent = payload.text || "暂无日志。";
    if (nearBottom) output.scrollTop = output.scrollHeight;
  } catch (error) {
    $("log-output").textContent = `日志读取失败：${error.message}`;
  }
}

function eventDetail(event) {
  const ignored = new Set(["timestamp", "type", "campaign_id"]);
  const parts = [];
  Object.entries(event).forEach(([key, value]) => {
    if (ignored.has(key)) return;
    if (typeof value === "object") return;
    parts.push(`${key}=${value}`);
  });
  return parts.join(" · ");
}

async function refreshEvents() {
  try {
    const payload = await api("/api/events?limit=100");
    const root = $("events-list");
    root.replaceChildren();
    (payload.events || []).forEach((event) => {
      const row = document.createElement("div");
      row.className = "event";
      const time = document.createElement("span");
      time.className = "event-time";
      time.textContent = formatTime(event.timestamp);
      const type = document.createElement("span");
      type.className = "event-type";
      type.textContent = event.type || "event";
      const detail = document.createElement("span");
      detail.className = "event-detail";
      detail.textContent = eventDetail(event) || "--";
      detail.title = detail.textContent;
      row.append(time, type, detail);
      root.appendChild(row);
    });
  } catch (error) {
    showToast(`事件读取失败：${error.message}`, true);
  }
}

async function refreshArtifacts() {
  try {
    const payload = await api("/api/artifacts?limit=160");
    const root = $("artifacts-list");
    root.replaceChildren();
    (payload.artifacts || []).forEach((artifact) => {
      const link = document.createElement("a");
      link.className = "artifact";
      link.href = artifact.url;
      link.target = "_blank";
      link.rel = "noopener";

      const kind = document.createElement("span");
      kind.className = "artifact-kind";
      kind.textContent = artifact.kind || "file";

      const info = document.createElement("span");
      const name = document.createElement("span");
      name.className = "artifact-name";
      name.textContent = artifact.name;
      const path = document.createElement("span");
      path.className = "artifact-path";
      path.textContent = artifact.path;
      info.append(name, path);
      link.append(kind, info);
      root.appendChild(link);
    });
    if (!(payload.artifacts || []).length) root.textContent = "尚未生成论文或可展示产物。";
  } catch (error) {
    showToast(`产物读取失败：${error.message}`, true);
  }
}

function switchTab(tab) {
  state.activeTab = tab;
  document.querySelectorAll(".tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tab);
  });
  document.querySelectorAll(".tab-view").forEach((view) => {
    view.classList.toggle("active", view.id === `${tab}-view`);
  });
  $("log-source").style.visibility = tab === "logs" ? "visible" : "hidden";
  if (tab === "logs") refreshLogs();
  if (tab === "events") refreshEvents();
  if (tab === "artifacts") refreshArtifacts();
}

async function control(action) {
  const isPause = action === "pause";
  const message = isPause
    ? "确认协作暂停？当前 Pipeline/远端任务会在安全控制点退出，GPU 租约与监控继续保留。"
    : "确认恢复 RSI Campaign？这会启动独立 Supervisor 服务并从持久状态继续。";
  if (!window.confirm(message)) return;
  const button = $(isPause ? "pause-button" : "resume-button");
  button.disabled = true;
  try {
    await api(`/api/control/${action}`, {
      method: "POST",
      body: JSON.stringify({ reason: `operator requested ${action} from dashboard` }),
    });
    showToast(isPause ? "暂停请求已提交。" : "恢复请求已提交。");
    window.setTimeout(refreshDashboard, 1200);
  } catch (error) {
    showToast(`${isPause ? "暂停" : "恢复"}失败：${error.message}`, true);
    button.disabled = false;
  }
}

function bindEvents() {
  $("refresh-button").addEventListener("click", () => {
    refreshDashboard();
    refreshLogs();
  });
  $("pause-button").addEventListener("click", () => control("pause"));
  $("resume-button").addEventListener("click", () => control("resume"));
  $("log-source").addEventListener("change", (event) => {
    state.logSource = event.target.value;
    refreshLogs();
  });
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
  });
}

async function start() {
  bindEvents();
  await Promise.all([refreshDashboard(), refreshLogs()]);
  state.refreshTimer = window.setInterval(refreshDashboard, 15000);
  state.logsTimer = window.setInterval(refreshLogs, 10000);
}

document.addEventListener("DOMContentLoaded", start);
