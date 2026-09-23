(() => {
  "use strict";

  if (window.location.protocol === "file:") {
    const showDirectFileNotice = () => document.body.classList.add("direct-file-mode");
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", showDirectFileNotice, { once: true });
    } else {
      showDirectFileNotice();
    }
    return;
  }

  const API_ROOT = "/api/v1";
  const ACTIVE_JOB_STATES = new Set(["pending", "queued", "running", "processing"]);
  const DONE_JOB_STATES = new Set(["completed", "complete", "succeeded", "success", "ready"]);
  const STORAGE_KEYS = {
    theme: "arf-studio-theme",
    activeKnowledgeBase: "arf-studio-active-kb",
    activeSection: "arf-studio-active-section",
  };
  const state = {
    bootstrap: null,
    csrfToken: "",
    knowledgeBases: [],
    activeKnowledgeBaseId: null,
    sources: [],
    jobs: [],
    modelSettings: {},
    currentTab: "search",
    editingKnowledgeBaseId: null,
    confirmAction: null,
    pollTimer: null,
    selectionVersion: 0,
    uploadCounter: 0,
    currentSection: "overview",
    rlOverview: null,
    rlRuns: [],
    rlDatasets: [],
    rlPollTimer: null,
  };

  const byId = (id) => document.getElementById(id);
  const elements = {};

  class ApiError extends Error {
    constructor(message, options = {}) {
      super(message);
      this.name = "ApiError";
      this.status = options.status || 0;
      this.code = options.code || "request_failed";
      this.details = options.details || null;
      this.requestId = options.requestId || "";
    }
  }

  function textValue(value, fallback = "") {
    if (value === null || value === undefined) return fallback;
    return String(value);
  }

  function createNode(tagName, options = {}, children = []) {
    const element = document.createElement(tagName);
    if (options.className) element.className = options.className;
    if (options.text !== undefined) element.textContent = textValue(options.text);
    if (options.type) element.type = options.type;
    if (options.title) element.title = options.title;
    if (options.hidden) element.hidden = true;
    if (options.attrs) {
      Object.entries(options.attrs).forEach(([name, value]) => {
        if (value !== null && value !== undefined) element.setAttribute(name, textValue(value));
      });
    }
    children.forEach((child) => {
      if (child instanceof Node) element.append(child);
      else if (child !== null && child !== undefined) element.append(document.createTextNode(textValue(child)));
    });
    return element;
  }

  function replaceChildren(element, ...children) {
    element.replaceChildren(...children.filter(Boolean));
  }

  function readStorage(key) {
    try {
      return window.localStorage.getItem(key);
    } catch (_error) {
      return null;
    }
  }

  function writeStorage(key, value) {
    try {
      window.localStorage.setItem(key, value);
    } catch (_error) {
      // The interface remains fully usable when private browsing blocks storage.
    }
  }

  function isWriteMethod(method) {
    return !["GET", "HEAD", "OPTIONS"].includes(method.toUpperCase());
  }

  async function parseResponse(response) {
    if (response.status === 204) return null;
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      try {
        return await response.json();
      } catch (_error) {
        throw new ApiError("服务返回了无法读取的数据。", {
          status: response.status,
          code: "invalid_response",
        });
      }
    }
    const plainText = await response.text();
    return plainText ? { message: plainText.slice(0, 500) } : null;
  }

  async function apiRequest(path, options = {}) {
    const method = (options.method || "GET").toUpperCase();
    const writeRequest = isWriteMethod(method);
    if (writeRequest && !state.csrfToken && !options.skipBootstrap) {
      await bootstrapSession();
    }

    const headers = new Headers(options.headers || {});
    headers.set("Accept", "application/json");
    let body = options.body;
    if (body && !(body instanceof FormData) && typeof body !== "string") {
      headers.set("Content-Type", "application/json");
      body = JSON.stringify(body);
    }
    if (writeRequest && state.csrfToken) headers.set("X-CSRF-Token", state.csrfToken);

    const controller = new AbortController();
    const timeoutMs = Number(options.timeoutMs || 30000);
    const timeout = window.setTimeout(() => controller.abort("timeout"), timeoutMs);
    if (options.signal) {
      if (options.signal.aborted) controller.abort(options.signal.reason);
      else options.signal.addEventListener("abort", () => controller.abort(options.signal.reason), { once: true });
    }

    let response;
    try {
      response = await window.fetch(`${API_ROOT}${path}`, {
        method,
        headers,
        body,
        credentials: "same-origin",
        cache: "no-store",
        signal: controller.signal,
      });
    } catch (error) {
      setConnectionState(false);
      if (error instanceof DOMException && error.name === "AbortError") {
        throw new ApiError("请求等待时间过长，请稍后重试。", { code: "timeout" });
      }
      throw new ApiError("无法连接本地服务，请确认服务仍在运行。", { code: "network_error" });
    } finally {
      window.clearTimeout(timeout);
    }

    const payload = await parseResponse(response);
    if (!response.ok) {
      const errorPayload = payload && typeof payload === "object" ? payload.error : null;
      const apiError = new ApiError(
        textValue(errorPayload?.message || payload?.message, `操作失败（${response.status}）`),
        {
          status: response.status,
          code: textValue(errorPayload?.code, "request_failed"),
          details: errorPayload?.details,
          requestId: textValue(errorPayload?.requestId),
        },
      );
      if (
        writeRequest &&
        !options.csrfRetried &&
        (response.status === 401 || response.status === 403) &&
        ["csrf_failed", "session_expired", "forbidden"].includes(apiError.code)
      ) {
        await bootstrapSession(true);
        return apiRequest(path, { ...options, csrfRetried: true, skipBootstrap: true });
      }
      throw apiError;
    }
    setConnectionState(true);
    return payload;
  }

  let bootstrapPromise = null;
  async function bootstrapSession(force = false) {
    if (bootstrapPromise && !force) return bootstrapPromise;
    bootstrapPromise = apiRequest("/bootstrap", { skipBootstrap: true, timeoutMs: 12000 })
      .then((payload) => {
        state.bootstrap = payload || {};
        state.csrfToken = textValue(payload?.security?.csrfToken || payload?.csrf_token);
        state.modelSettings = payload?.modelSettings || payload?.model_settings || {};
        applyCapabilities(payload?.capabilities || {});
        renderModelConnection();
        const version = payload?.app?.version;
        elements.appVersion.textContent = version ? `v${version}` : "本地运行";
        elements.fatalState.hidden = true;
        return payload;
      })
      .catch((error) => {
        showFatalError(error);
        throw error;
      })
      .finally(() => {
        bootstrapPromise = null;
      });
    return bootstrapPromise;
  }

  function applyCapabilities(capabilities) {
    const extensions = Array.isArray(capabilities.supportedExtensions)
      ? capabilities.supportedExtensions.map((item) => textValue(item).toLowerCase())
      : [".txt", ".md", ".markdown", ".rst", ".pdf", ".docx", ".html", ".htm"];
    const normalized = extensions.map((extension) => (extension.startsWith(".") ? extension : `.${extension}`));
    state.supportedExtensions = new Set(normalized);
    state.maxUploadBytes = Number(capabilities.maxUploadBytes || 0);
    state.searchModes = Array.isArray(capabilities.searchModes) ? capabilities.searchModes : ["bm25"];
    elements.fileInput.accept = normalized.join(",");
    const formatSummary = normalized.map((item) => item.slice(1).toUpperCase()).join("、");
    const sizeSummary = state.maxUploadBytes ? `，单个不超过 ${formatBytes(state.maxUploadBytes)}` : "";
    elements.uploadHelp.textContent = `支持 ${formatSummary || "常见文档格式"}${sizeSummary}`;
  }

  function setConnectionState(online) {
    elements.connectionStatus.classList.toggle("is-online", online);
    elements.connectionStatus.classList.toggle("is-offline", !online);
    elements.connectionLabel.textContent = online ? "本地服务正常" : "连接已断开";
  }

  function showFatalError(error) {
    setConnectionState(false);
    elements.fatalMessage.textContent = friendlyError(error);
    elements.fatalState.hidden = false;
  }

  function friendlyError(error) {
    if (error instanceof ApiError) return error.message;
    return "操作没有完成，请稍后重试。";
  }

  function showToast(title, message = "", type = "success", duration = 4600) {
    const toast = createNode("div", {
      className: `toast${type === "error" ? " is-error" : type === "warning" ? " is-warning" : ""}`,
      attrs: { role: type === "error" ? "alert" : "status" },
    });
    const icon = createNode("span", { className: "toast-icon", text: type === "error" ? "!" : type === "warning" ? "i" : "✓", attrs: { "aria-hidden": "true" } });
    const copy = createNode("div", { className: "toast-copy" }, [
      createNode("strong", { text: title }),
      message ? createNode("span", { text: message }) : null,
    ]);
    const close = createNode("button", { type: "button", text: "×", title: "关闭通知", attrs: { "aria-label": "关闭通知" } });
    const remove = () => toast.remove();
    close.addEventListener("click", remove);
    toast.append(icon, copy, close);
    elements.toastRegion.append(toast);
    if (duration > 0) window.setTimeout(remove, duration);
  }

  function setButtonBusy(button, busy, label) {
    button.disabled = busy;
    button.setAttribute("aria-busy", busy ? "true" : "false");
    const labelNode = button.querySelector("span");
    if (labelNode && label) {
      if (!button.dataset.normalLabel) button.dataset.normalLabel = labelNode.textContent || "";
      labelNode.textContent = busy ? label : button.dataset.normalLabel;
    }
  }

  function formatBytes(bytes) {
    const value = Number(bytes || 0);
    if (!Number.isFinite(value) || value <= 0) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
    const amount = value / 1024 ** index;
    return `${amount >= 10 || index === 0 ? amount.toFixed(0) : amount.toFixed(1)} ${units[index]}`;
  }

  function formatDate(value) {
    if (!value) return "未知时间";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return textValue(value);
    return new Intl.DateTimeFormat("zh-CN", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(date);
  }

  function relativeDate(value) {
    if (!value) return "尚未更新";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "最近";
    const seconds = Math.round((date.getTime() - Date.now()) / 1000);
    const formatter = new Intl.RelativeTimeFormat("zh-CN", { numeric: "auto" });
    const intervals = [
      [31536000, "year"],
      [2592000, "month"],
      [86400, "day"],
      [3600, "hour"],
      [60, "minute"],
    ];
    for (const [amount, unit] of intervals) {
      if (Math.abs(seconds) >= amount) return formatter.format(Math.round(seconds / amount), unit);
    }
    return "刚刚";
  }

  function normalizeItems(payload) {
    if (Array.isArray(payload)) return payload;
    if (Array.isArray(payload?.items)) return payload.items;
    if (Array.isArray(payload?.knowledgeBases)) return payload.knowledgeBases;
    if (Array.isArray(payload?.sources)) return payload.sources;
    if (Array.isArray(payload?.jobs)) return payload.jobs;
    return [];
  }

  const STUDIO_SECTIONS = new Set([
    "overview",
    "training",
    "evaluation",
    "assets",
    "models",
    "comparison",
    "knowledge",
  ]);

  function switchSection(section, options = {}) {
    const target = STUDIO_SECTIONS.has(section) ? section : "overview";
    state.currentSection = target;
    writeStorage(STORAGE_KEYS.activeSection, target);
    document.querySelectorAll("[data-view]").forEach((view) => {
      view.hidden = view.dataset.view !== target;
    });
    document.querySelectorAll("[data-section]").forEach((button) => {
      const active = button.dataset.section === target;
      button.classList.toggle("is-active", active);
      if (active) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
    });
    closeSidebar();
    if (options.focus !== false) elements.mainContent?.focus({ preventScroll: true });
    if (options.scroll !== false) window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function renderModelConnection() {
    if (!elements.assistantModelBadge) return;
    const settings = state.modelSettings || {};
    const endpoint = textValue(settings.baseUrl || settings.base_url).trim();
    const model = textValue(settings.model).trim();
    const configured = Boolean(endpoint && model);
    elements.assistantModelBadge.textContent = configured ? "已配置" : "未配置";
    elements.assistantModelBadge.className = `capability-badge ${configured ? "available" : "configured"}`;
    elements.assistantModelName.textContent = model || "尚未设置";
    elements.assistantModelEndpoint.textContent = endpoint || "尚未设置";
    elements.assistantModelEndpoint.title = endpoint;
  }

  function diagnosticPresentation(status) {
    const normalized = textValue(status).toLowerCase();
    if (["pass", "passed", "ready", "ok"].includes(normalized)) {
      return { label: "通过", className: "" };
    }
    if (["warn", "warning", "setup_required"].includes(normalized)) {
      return { label: "需配置", className: "warning" };
    }
    if (["fail", "failed", "error"].includes(normalized)) {
      return { label: "未通过", className: "failed" };
    }
    return { label: "检查中", className: "checking" };
  }

  function renderRLOverview() {
    const overview = state.rlOverview || {};
    const summary = overview.summary || {};
    const core = overview.diagnostics?.core || {};
    const training = overview.diagnostics?.training || {};
    const coreState = diagnosticPresentation(core.status);
    const trainingReady = Boolean(summary.trainingRuntimeReady);

    elements.rlCoreBadge.textContent = coreState.className ? coreState.label : "核心已就绪";
    elements.rlEngineStatus.textContent = coreState.label;
    elements.rlEngineStatus.className = `capability-badge ${coreState.className ? "configured" : "available"}`;
    elements.rlEngineDetail.textContent = coreState.className
      ? "部分核心检查需要处理"
      : `Python ${textValue(core.pythonVersion, "环境正常")} · 本地可运行`;
    elements.rlTrainingStatus.textContent = trainingReady ? "已就绪" : "需配置";
    elements.rlTrainingStatus.className = `capability-badge ${trainingReady ? "available" : "configured"}`;
    elements.rlTrainingDetail.textContent = trainingReady
      ? "PyTorch 与 verl 训练运行时检查通过"
      : "需要另行安装 PyTorch、verl 与 GPU 运行时";
    elements.metricTrainingRuntime.textContent = trainingReady ? "已就绪" : "未连接";
    elements.metricTrainingRuntime.classList.toggle("runtime-ready", trainingReady);
    elements.trainerModelBadge.textContent = trainingReady ? "运行时就绪" : "需配置";
    elements.trainerModelBadge.className = `capability-badge ${trainingReady ? "available" : "configured"}`;
    elements.trainerRuntimeDetail.textContent = trainingReady ? "检查通过" : "未检测到完整训练依赖";
    elements.environmentPageStatus.textContent = coreState.className ? "核心环境需处理" : "核心环境正常";

    const diagnostics = [
      { title: "Agent RL 核心", report: core },
      { title: "GPU / verl 训练环境", report: training },
    ];
    const rows = diagnostics.map(({ title, report }) => {
      const presentation = diagnosticPresentation(report.status);
      const checks = Array.isArray(report.checks) ? report.checks : [];
      const requiredIssues = checks.filter((check) => check.required && diagnosticPresentation(check.status).className).length;
      const optionalIssues = checks.filter((check) => !check.required && diagnosticPresentation(check.status).className).length;
      const detail = checks.length
        ? requiredIssues
          ? `${checks.length} 项检查，${requiredIssues} 项必需依赖未就绪`
          : optionalIssues
            ? `${checks.length} 项检查，核心通过，${optionalIssues} 个可选运行时未安装`
            : `${checks.length} 项检查，全部通过`
        : "尚无诊断详情";
      return createNode("div", { className: "diagnostic-row" }, [
        createNode("span", { className: `diagnostic-dot ${presentation.className}`.trim() }),
        createNode("div", {}, [
          createNode("strong", { text: title }),
          createNode("small", { text: detail }),
        ]),
        createNode("span", { text: presentation.label }),
      ]);
    });
    replaceChildren(elements.diagnosticsList, ...rows);
  }

  function runStatusPresentation(status) {
    const normalized = textValue(status).toLowerCase();
    if (["queued", "pending"].includes(normalized)) return { label: "等待运行", className: "running" };
    if (["running", "processing"].includes(normalized)) return { label: "运行中", className: "running" };
    if (["succeeded", "success", "completed"].includes(normalized)) return { label: "已完成", className: "succeeded" };
    if (["failed", "error"].includes(normalized)) return { label: "失败", className: "failed" };
    return { label: normalized || "未知", className: "" };
  }

  function createRLRunRow(run) {
    const status = runStatusPresentation(run.status);
    const result = run.result || {};
    const rolloutCount = Number(run.config?.rolloutsPerTask || run.config?.rollouts_per_task || 0);
    let detail = `${formatDate(run.updatedAt || run.updated_at)}${rolloutCount ? ` · 每题 ${rolloutCount} 次 rollout` : ""}`;
    if (status.className === "succeeded") {
      const trajectories = Number(result.trajectory_count || 0);
      const accepted = Number(result.accepted_trajectory_count || 0);
      detail = `${trajectories} 条轨迹 · ${accepted} 条通过信号筛选`;
    } else if (status.className === "failed" && run.error) {
      detail = textValue(run.error).slice(0, 150);
    }
    return createNode("div", { className: "run-item", attrs: { "data-run-id": textValue(run.id) } }, [
      createNode("span", { className: "run-item-icon", text: "RL", attrs: { "aria-hidden": "true" } }),
      createNode("span", { className: "run-item-copy" }, [
        createNode("strong", { text: textValue(run.title, "本地离线 RL 数据管线") }),
        createNode("small", { text: detail }),
      ]),
      createNode("span", { className: `run-status ${status.className}`.trim(), text: status.label }),
    ]);
  }

  function renderRLRuns() {
    const runs = Array.isArray(state.rlRuns) ? state.rlRuns : [];
    const running = runs.filter((run) => ["queued", "running"].includes(textValue(run.status).toLowerCase())).length;
    const completed = runs.filter((run) => textValue(run.status).toLowerCase() === "succeeded").length;
    const failed = runs.filter((run) => textValue(run.status).toLowerCase() === "failed").length;
    elements.metricRunningRuns.textContent = textValue(running);
    elements.metricCompletedRuns.textContent = textValue(completed);
    elements.metricFailedRuns.textContent = textValue(failed);
    elements.trainingNavStatus.classList.toggle("is-active", running > 0);
    elements.trainingNavStatus.setAttribute("aria-label", running ? `${running} 个运行中的任务` : "没有运行中的任务");

    const empty = createNode("div", { className: "empty-run-state" }, [
      createNode("span", { text: "◎", attrs: { "aria-hidden": "true" } }),
      createNode("strong", { text: "暂无离线运行" }),
      createNode("small", { text: "新建一次运行，验证完整的数据路径。" }),
    ]);
    replaceChildren(elements.trainingRunList, ...(runs.length ? runs.map(createRLRunRow) : [empty]));
    const recent = runs.slice(0, 3);
    const compactEmpty = createNode("div", { className: "empty-run-state" }, [
      createNode("span", { text: "◎", attrs: { "aria-hidden": "true" } }),
      createNode("strong", { text: "还没有运行记录" }),
      createNode("small", { text: "从轨迹演示或离线数据管线开始。" }),
    ]);
    replaceChildren(elements.overviewRunList, ...(recent.length ? recent.map(createRLRunRow) : [compactEmpty]));
    scheduleRLPolling();
  }

  function renderRLDatasets() {
    const datasets = Array.isArray(state.rlDatasets) ? state.rlDatasets : [];
    const rows = datasets.map((dataset) => createNode("div", { className: "dataset-row" }, [
      createNode("span", { className: "dataset-row-icon", text: "JSONL", attrs: { "aria-hidden": "true" } }),
      createNode("span", { className: "dataset-row-copy" }, [
        createNode("strong", { text: textValue(dataset.name, "未命名数据集") }),
        createNode("small", { text: dataset.builtIn ? "项目内置 · 可直接运行" : `本机导入 · ${formatDate(dataset.createdAt)}` }),
      ]),
      createNode("span", { className: "dataset-row-stat", text: `${Number(dataset.taskCount || 0)} 个任务` }),
      createNode("span", { className: "dataset-row-stat", text: `${Number(dataset.documentCount || 0)} 篇语料 · ${formatBytes(dataset.sizeBytes)}` }),
    ]));
    const empty = createNode("div", { className: "empty-run-state" }, [
      createNode("span", { text: "◎" }),
      createNode("strong", { text: "没有可用数据集" }),
      createNode("small", { text: "请导入 QA 与检索语料 JSONL。" }),
    ]);
    replaceChildren(elements.rlDatasetList, ...(rows.length ? rows : [empty]));

    const selected = elements.offlineDatasetSelect.value;
    const options = datasets.map((dataset) => createNode("option", {
      text: `${textValue(dataset.name)} · ${Number(dataset.taskCount || 0)} 个任务`,
      attrs: { value: textValue(dataset.id) },
    }));
    replaceChildren(elements.offlineDatasetSelect, ...options);
    if (datasets.some((dataset) => textValue(dataset.id) === selected)) {
      elements.offlineDatasetSelect.value = selected;
    } else if (datasets.length) {
      elements.offlineDatasetSelect.value = textValue(datasets[0].id);
    }
  }

  async function loadRLOverview(options = {}) {
    try {
      state.rlOverview = await apiRequest("/rl/overview");
      renderRLOverview();
    } catch (error) {
      if (!options.silent) showToast("无法读取 RL 环境", friendlyError(error), "error");
      elements.rlCoreBadge.textContent = "暂不可用";
    }
  }

  async function loadRLRuns(options = {}) {
    try {
      const payload = await apiRequest("/rl/offline-runs");
      state.rlRuns = normalizeItems(payload);
      renderRLRuns();
    } catch (error) {
      if (!options.silent) showToast("无法读取运行记录", friendlyError(error), "error");
    }
  }

  async function loadRLDatasets(options = {}) {
    try {
      const payload = await apiRequest("/rl/datasets");
      state.rlDatasets = normalizeItems(payload);
      renderRLDatasets();
    } catch (error) {
      if (!options.silent) showToast("无法读取数据集", friendlyError(error), "error");
    }
  }

  async function loadRLWorkspace(options = {}) {
    await Promise.all([
      loadRLOverview(options),
      loadRLRuns(options),
      loadRLDatasets(options),
    ]);
  }

  function scheduleRLPolling() {
    if (state.rlPollTimer) window.clearTimeout(state.rlPollTimer);
    state.rlPollTimer = null;
    const active = state.rlRuns.some((run) => ["queued", "running"].includes(textValue(run.status).toLowerCase()));
    if (active && !document.hidden) {
      state.rlPollTimer = window.setTimeout(() => loadRLRuns({ silent: true }), 1500);
    }
  }

  async function runTrajectoryDemo() {
    const buttons = [elements.runDemoButton, elements.quickDemoButton];
    buttons.forEach((button) => {
      button.disabled = true;
      button.setAttribute("aria-busy", "true");
    });
    elements.demoResult.hidden = false;
    replaceChildren(elements.demoResult, createNode("div", { className: "skeleton-result", attrs: { "aria-label": "正在运行真实轨迹" } }));
    try {
      const result = await apiRequest("/rl/trajectory-demo", { method: "POST", timeoutMs: 60000 });
      const summary = result?.summary || {};
      const metrics = [
        ["步骤", Number(summary.stepCount || 0)],
        ["总奖励", Number(summary.totalReward || 0).toFixed(3)],
        ["生成 Token", Number(summary.generatedTokens || 0)],
        ["观察 Token", Number(summary.observationTokens || 0)],
      ].map(([label, value]) => createNode("div", {}, [
        createNode("span", { text: label }),
        createNode("strong", { text: value }),
      ]));
      replaceChildren(elements.demoResult,
        createNode("div", { className: "demo-result-head" }, [
          createNode("strong", { text: "真实轨迹已完成：搜索工具 → 环境反馈 → 最终答案 → 奖励" }),
          createNode("span", { text: textValue(result.status, "completed") }),
        ]),
        createNode("div", { className: "demo-metrics" }, metrics),
      );
      showToast("轨迹演示完成", `轨迹 ${textValue(result.trajectoryId).slice(0, 16)}… 已在本机真实执行。`);
    } catch (error) {
      replaceChildren(elements.demoResult, createNode("div", { className: "result-error" }, [
        createNode("strong", { text: "轨迹演示没有完成" }),
        createNode("span", { text: friendlyError(error) }),
      ]));
    } finally {
      buttons.forEach((button) => {
        button.disabled = false;
        button.setAttribute("aria-busy", "false");
      });
    }
  }

  function openOfflineRunDialog() {
    showDialog(elements.offlineRunDialog, elements.offlineDatasetSelect);
  }

  async function createOfflineRun(event) {
    event.preventDefault();
    if (!elements.offlineRunForm.reportValidity()) return;
    setButtonBusy(elements.offlineRunSubmit, true, "正在创建");
    try {
      const run = await apiRequest("/rl/offline-runs", {
        method: "POST",
        body: {
          datasetId: elements.offlineDatasetSelect.value || "sample",
          rolloutsPerTask: Number(elements.offlineRolloutCount.value || 3),
          seed: 0,
          maxConcurrency: 4,
        },
        timeoutMs: 30000,
      });
      closeDialog(elements.offlineRunDialog);
      if (run) state.rlRuns.unshift(run);
      renderRLRuns();
      switchSection("training");
      showToast("离线运行已开始", "正在本机生成真实轨迹、评测和训练批次。");
    } catch (error) {
      showToast("无法开始离线运行", friendlyError(error), "error", 0);
    } finally {
      setButtonBusy(elements.offlineRunSubmit, false, "正在创建");
    }
  }

  function openDatasetDialog() {
    elements.datasetForm.reset();
    showDialog(elements.datasetDialog, elements.datasetName);
  }

  async function createRLDataset(event) {
    event.preventDefault();
    if (!elements.datasetForm.reportValidity()) return;
    const qaFile = elements.datasetQaFile.files?.[0];
    const corpusFile = elements.datasetCorpusFile.files?.[0];
    if (!qaFile || !corpusFile) return;
    const body = new FormData();
    body.append("name", elements.datasetName.value.trim());
    body.append("qaFile", qaFile, qaFile.name);
    body.append("corpusFile", corpusFile, corpusFile.name);
    setButtonBusy(elements.datasetSubmit, true, "正在校验");
    try {
      const dataset = await apiRequest("/rl/datasets", { method: "POST", body, timeoutMs: 120000 });
      closeDialog(elements.datasetDialog);
      if (dataset) state.rlDatasets.unshift(dataset);
      renderRLDatasets();
      showToast("数据集已导入", `${textValue(dataset?.name)} 可以用于离线运行。`);
    } catch (error) {
      showToast("数据集导入失败", friendlyError(error), "error", 0);
    } finally {
      setButtonBusy(elements.datasetSubmit, false, "正在校验");
    }
  }

  async function copyCommand(button) {
    const command = textValue(button.dataset.copy);
    if (!command) return;
    try {
      await navigator.clipboard.writeText(command);
      const original = button.textContent;
      button.textContent = "已复制";
      window.setTimeout(() => { button.textContent = original; }, 1300);
    } catch (_error) {
      showToast("无法自动复制", "请手动选择命令文本。", "warning");
    }
  }

  function activeKnowledgeBase() {
    return state.knowledgeBases.find((item) => textValue(item.id) === state.activeKnowledgeBaseId) || null;
  }

  async function loadKnowledgeBases(options = {}) {
    elements.libraryListLoading.hidden = false;
    try {
      const payload = await apiRequest("/knowledge-bases");
      state.knowledgeBases = normalizeItems(payload);
      renderKnowledgeBases();
      if (!state.knowledgeBases.length) {
        showWelcome();
        return;
      }
      const preferred = options.preferredId || state.activeKnowledgeBaseId || readStorage(STORAGE_KEYS.activeKnowledgeBase);
      const target = state.knowledgeBases.find((item) => textValue(item.id) === preferred) || state.knowledgeBases[0];
      await selectKnowledgeBase(textValue(target.id));
    } catch (error) {
      showToast("无法读取知识库", friendlyError(error), "error", 0);
    } finally {
      elements.libraryListLoading.hidden = true;
    }
  }

  function renderKnowledgeBases() {
    const items = state.knowledgeBases.map((knowledgeBase) => {
      const id = textValue(knowledgeBase.id);
      const name = textValue(knowledgeBase.name, "未命名知识库");
      const button = createNode("button", {
        type: "button",
        attrs: {
          "aria-current": id === state.activeKnowledgeBaseId ? "page" : null,
          "aria-label": `打开知识库：${name}`,
        },
      }, [
        createNode("span", { className: "library-mini-avatar", text: name.trim().slice(0, 1) || "知", attrs: { "aria-hidden": "true" } }),
        createNode("span", { className: "library-list-copy" }, [
          createNode("strong", { text: name }),
          createNode("small", { text: textValue(knowledgeBase.description, "本地知识库") || "本地知识库" }),
        ]),
      ]);
      button.addEventListener("click", () => selectKnowledgeBase(id));
      return createNode("li", { className: "library-list-item" }, [button]);
    });
    replaceChildren(elements.libraryList, ...items);
    elements.libraryListEmpty.hidden = items.length > 0;
    elements.knowledgeNavCount.textContent = textValue(items.length);
    elements.overviewKbCount.textContent = textValue(items.length);
  }

  function showWelcome() {
    state.activeKnowledgeBaseId = null;
    state.sources = [];
    state.jobs = [];
    stopJobPolling();
    elements.welcomeView.hidden = false;
    elements.workspace.hidden = true;
    renderKnowledgeBases();
    renderActivity();
  }

  async function selectKnowledgeBase(id) {
    if (!id) return;
    const version = ++state.selectionVersion;
    state.activeKnowledgeBaseId = id;
    writeStorage(STORAGE_KEYS.activeKnowledgeBase, id);
    renderKnowledgeBases();
    elements.welcomeView.hidden = true;
    elements.workspace.hidden = false;
    closeSidebar();
    renderActiveKnowledgeBase();
    renderDocumentLoading();
    stopJobPolling();
    try {
      const encoded = encodeURIComponent(id);
      const [knowledgeBase, sourcePayload, jobPayload] = await Promise.all([
        apiRequest(`/knowledge-bases/${encoded}`),
        apiRequest(`/knowledge-bases/${encoded}/sources`),
        apiRequest(`/jobs?knowledgeBaseId=${encoded}&limit=100`),
      ]);
      if (version !== state.selectionVersion) return;
      const listIndex = state.knowledgeBases.findIndex((item) => textValue(item.id) === id);
      if (listIndex >= 0 && knowledgeBase) state.knowledgeBases[listIndex] = knowledgeBase;
      state.sources = normalizeItems(sourcePayload);
      state.jobs = normalizeItems(jobPayload);
      renderKnowledgeBases();
      renderActiveKnowledgeBase();
      renderDocuments();
      renderActivity();
      scheduleJobPolling();
    } catch (error) {
      if (version !== state.selectionVersion) return;
      showToast("知识库载入失败", friendlyError(error), "error", 0);
      renderDocumentsError(error);
    }
  }

  function renderActiveKnowledgeBase() {
    const knowledgeBase = activeKnowledgeBase();
    if (!knowledgeBase) return;
    const name = textValue(knowledgeBase.name, "知识库");
    elements.libraryTitle.textContent = name;
    elements.libraryAvatar.textContent = name.trim().slice(0, 1) || "知";
    elements.libraryDescription.textContent = textValue(knowledgeBase.description) || "从资料中查找信息并获得有引用的回答。";
    const ready = state.sources.filter((source) => normalizedStatus(source.status) === "ready").length;
    const processing = state.jobs.filter((job) => ACTIVE_JOB_STATES.has(normalizedStatus(job.status))).length;
    const failed = state.sources.filter((source) => normalizedStatus(source.status) === "failed").length;
    elements.statSources.textContent = textValue(state.sources.length);
    elements.statReady.textContent = textValue(ready);
    elements.statProcessing.textContent = textValue(processing);
    elements.statUpdated.textContent = relativeDate(knowledgeBase.updatedAt || knowledgeBase.updated_at);
    elements.documentsTabCount.textContent = textValue(state.sources.length);
    if (processing > 0) setIndexBadge("正在处理", "warning");
    else if (failed > 0) setIndexBadge("部分失败", "danger");
    else if (ready > 0) setIndexBadge("可以使用", "success");
    else setIndexBadge("等待资料", "neutral");
  }

  function setIndexBadge(label, variant) {
    elements.libraryIndexStatus.textContent = label;
    elements.libraryIndexStatus.className = `badge badge-${variant}`;
  }

  function normalizedStatus(status) {
    const value = textValue(status).toLowerCase();
    if (["complete", "completed", "succeeded", "success", "indexed"].includes(value)) return "ready";
    if (["pending", "queued", "running", "processing", "indexing"].includes(value)) return "processing";
    if (["failed", "error"].includes(value)) return "failed";
    return value || "unknown";
  }

  function statusPresentation(status) {
    const normalized = normalizedStatus(status);
    if (normalized === "ready") return { label: "已就绪", variant: "success" };
    if (normalized === "processing") return { label: "处理中", variant: "warning" };
    if (normalized === "failed") return { label: "处理失败", variant: "danger" };
    return { label: "等待处理", variant: "neutral" };
  }

  function renderDocumentLoading() {
    const skeletons = Array.from({ length: 3 }, () => createNode("div", { className: "skeleton-result" }));
    replaceChildren(elements.documentList, createNode("div", { className: "skeleton-stack", attrs: { "aria-label": "正在载入资料" } }, skeletons));
    elements.documentsEmpty.hidden = true;
  }

  function renderDocumentsError(error) {
    replaceChildren(elements.documentList, createNode("div", { className: "result-error" }, [
      createNode("strong", { text: "资料列表载入失败" }),
      createNode("span", { text: friendlyError(error) }),
    ]));
    elements.documentsEmpty.hidden = true;
  }

  function renderDocuments() {
    const filter = elements.documentFilter.value.trim().toLocaleLowerCase("zh-CN");
    const sources = state.sources.filter((source) => textValue(source.originalName || source.name).toLocaleLowerCase("zh-CN").includes(filter));
    const rows = sources.map((source) => createDocumentRow(source));
    replaceChildren(elements.documentList, ...rows);
    elements.documentsEmpty.hidden = state.sources.length > 0;
    elements.documentSummary.textContent = filter
      ? `显示 ${sources.length} / ${state.sources.length} 个文件`
      : `共 ${state.sources.length} 个文件`;
    renderActiveKnowledgeBase();
  }

  function createDocumentRow(source) {
    const sourceId = textValue(source.id);
    const filename = textValue(source.originalName || source.name, "未命名文件");
    const extension = filename.includes(".") ? filename.split(".").pop().slice(0, 5).toUpperCase() : "DOC";
    const status = statusPresentation(source.status);
    const fileIcon = createNode("span", { className: "file-icon", text: extension, attrs: { "aria-hidden": "true" } });
    const copy = createNode("span", { className: "document-name" }, [
      createNode("strong", { text: filename, title: filename }),
      createNode("small", { text: source.error ? textValue(source.error) : `${Number(source.documentCount || 0)} 个内容片段` }),
    ]);
    const badge = createNode("span", { className: `badge badge-${status.variant} document-status`, text: status.label });
    const date = createNode("span", { className: "document-date", text: formatDate(source.updatedAt || source.createdAt) });
    const size = createNode("span", { className: "document-size", text: formatBytes(source.sizeBytes) });
    const retry = createNode("button", { className: "icon-button reindex-source", type: "button", text: "↻", title: `重新处理 ${filename}`, attrs: { "aria-label": `重新处理 ${filename}` } });
    retry.addEventListener("click", () => reindexSource(sourceId));
    const remove = createNode("button", { className: "icon-button delete-source", type: "button", text: "×", title: `删除 ${filename}`, attrs: { "aria-label": `删除 ${filename}` } });
    remove.addEventListener("click", () => requestSourceDeletion(source));
    const actions = createNode("span", { className: "document-actions" }, [retry, remove]);
    return createNode("div", { className: "document-row" }, [fileIcon, copy, badge, date, size, actions]);
  }

  function openLibraryDialog(knowledgeBase = null) {
    state.editingKnowledgeBaseId = knowledgeBase ? textValue(knowledgeBase.id) : null;
    elements.libraryDialogTitle.textContent = knowledgeBase ? "编辑知识库" : "新建知识库";
    elements.libraryName.value = knowledgeBase ? textValue(knowledgeBase.name) : "";
    elements.libraryDescriptionInput.value = knowledgeBase ? textValue(knowledgeBase.description) : "";
    elements.librarySave.textContent = knowledgeBase ? "保存修改" : "创建知识库";
    showDialog(elements.libraryDialog, elements.libraryName);
  }

  async function saveKnowledgeBase(event) {
    event.preventDefault();
    if (!elements.libraryForm.reportValidity()) return;
    const payload = {
      name: elements.libraryName.value.trim(),
      description: elements.libraryDescriptionInput.value.trim(),
    };
    const editingId = state.editingKnowledgeBaseId;
    elements.librarySave.disabled = true;
    elements.librarySave.setAttribute("aria-busy", "true");
    try {
      const result = editingId
        ? await apiRequest(`/knowledge-bases/${encodeURIComponent(editingId)}`, { method: "PATCH", body: payload })
        : await apiRequest("/knowledge-bases", { method: "POST", body: payload });
      closeDialog(elements.libraryDialog);
      showToast(editingId ? "知识库已更新" : "知识库已创建", editingId ? "修改已保存。" : "现在可以添加资料了。");
      await loadKnowledgeBases({ preferredId: textValue(result?.id || editingId) });
      if (!editingId) {
        switchTab("documents");
        window.setTimeout(() => elements.fileInput.click(), 180);
      }
    } catch (error) {
      showToast(editingId ? "保存失败" : "创建失败", friendlyError(error), "error", 0);
    } finally {
      elements.librarySave.disabled = false;
      elements.librarySave.setAttribute("aria-busy", "false");
    }
  }

  function requestKnowledgeBaseDeletion() {
    const knowledgeBase = activeKnowledgeBase();
    if (!knowledgeBase) return;
    openConfirmation({
      title: "删除这个知识库？",
      message: `“${textValue(knowledgeBase.name)}”及其中的全部资料会从本机删除，此操作无法撤销。`,
      actionLabel: "确认删除",
      action: async () => {
        await apiRequest(`/knowledge-bases/${encodeURIComponent(textValue(knowledgeBase.id))}`, { method: "DELETE" });
        state.activeKnowledgeBaseId = null;
        showToast("知识库已删除");
        await loadKnowledgeBases();
      },
    });
  }

  function requestSourceDeletion(source) {
    const filename = textValue(source.originalName || source.name, "这个文件");
    openConfirmation({
      title: "删除这份资料？",
      message: `“${filename}”及其索引内容会被删除，此操作无法撤销。`,
      actionLabel: "删除资料",
      action: async () => {
        const kbId = encodeURIComponent(state.activeKnowledgeBaseId);
        await apiRequest(`/knowledge-bases/${kbId}/sources/${encodeURIComponent(textValue(source.id))}`, { method: "DELETE" });
        state.sources = state.sources.filter((item) => textValue(item.id) !== textValue(source.id));
        renderDocuments();
        showToast("资料已删除", filename);
      },
    });
  }

  function openConfirmation({ title, message, actionLabel, action }) {
    elements.confirmTitle.textContent = title;
    elements.confirmMessage.textContent = message;
    elements.confirmAction.textContent = actionLabel;
    state.confirmAction = action;
    showDialog(elements.confirmDialog, elements.confirmAction);
  }

  async function runConfirmedAction(event) {
    event.preventDefault();
    if (typeof state.confirmAction !== "function") return;
    elements.confirmAction.disabled = true;
    try {
      await state.confirmAction();
      closeDialog(elements.confirmDialog);
    } catch (error) {
      showToast("操作失败", friendlyError(error), "error", 0);
    } finally {
      elements.confirmAction.disabled = false;
      state.confirmAction = null;
    }
  }

  function validateFiles(fileList) {
    const valid = [];
    Array.from(fileList || []).forEach((file) => {
      const dot = file.name.lastIndexOf(".");
      const extension = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
      if (state.supportedExtensions?.size && !state.supportedExtensions.has(extension)) {
        showToast("不支持这个文件", `${file.name}：请选择 ${Array.from(state.supportedExtensions).join("、")} 格式。`, "warning");
      } else if (state.maxUploadBytes && file.size > state.maxUploadBytes) {
        showToast("文件太大", `${file.name}：单个文件不能超过 ${formatBytes(state.maxUploadBytes)}。`, "warning");
      } else if (file.size === 0) {
        showToast("文件是空的", file.name, "warning");
      } else {
        valid.push(file);
      }
    });
    return valid;
  }

  async function uploadFiles(fileList) {
    if (!state.activeKnowledgeBaseId) return;
    const files = validateFiles(fileList);
    if (!files.length) return;
    switchTab("documents");
    elements.uploadQueue.hidden = false;
    const queue = files.map((file) => ({ file, view: createUploadView(file) }));
    const workers = Array.from({ length: Math.min(2, queue.length) }, async () => {
      while (queue.length) {
        const item = queue.shift();
        if (item) await uploadOneFile(item.file, item.view);
      }
    });
    await Promise.all(workers);
    elements.fileInput.value = "";
    await refreshSourcesAndJobs();
    window.setTimeout(() => {
      if (!elements.uploadQueue.querySelector(".is-uploading")) elements.uploadQueue.hidden = true;
    }, 5000);
  }

  function createUploadView(file) {
    state.uploadCounter += 1;
    const copy = createNode("span", { className: "upload-item-copy" }, [
      createNode("strong", { text: file.name }),
      createNode("small", { text: `${formatBytes(file.size)} · 等待上传` }),
    ]);
    const progress = createNode("span", { className: "progress-track", attrs: { role: "progressbar", "aria-label": `上传 ${file.name}`, "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": "0" } }, [
      createNode("span", { className: "progress-bar" }),
    ]);
    const status = createNode("span", { className: "upload-state", text: "…", attrs: { "aria-hidden": "true" } });
    const view = createNode("div", { className: "upload-item", attrs: { "data-upload-id": textValue(state.uploadCounter) } }, [copy, progress, status]);
    elements.uploadQueue.append(view);
    return view;
  }

  async function uploadOneFile(file, view) {
    const copyStatus = view.querySelector("small");
    const progress = view.querySelector("[role='progressbar']");
    const stateIcon = view.querySelector(".upload-state");
    view.classList.add("is-uploading");
    progress.setAttribute("aria-valuenow", "65");
    copyStatus.textContent = `${formatBytes(file.size)} · 正在上传`;
    const body = new FormData();
    body.append("file", file, file.name);
    try {
      const kbId = encodeURIComponent(state.activeKnowledgeBaseId);
      const result = await apiRequest(`/knowledge-bases/${kbId}/sources`, {
        method: "POST",
        body,
        timeoutMs: 120000,
      });
      view.classList.remove("is-uploading");
      view.classList.add("is-complete");
      progress.setAttribute("aria-valuenow", "100");
      copyStatus.textContent = `${formatBytes(file.size)} · 已上传，正在处理`;
      stateIcon.textContent = "✓";
      if (result?.source) upsertById(state.sources, result.source);
      if (result?.job) upsertById(state.jobs, result.job);
    } catch (error) {
      view.classList.remove("is-uploading");
      view.classList.add("is-failed");
      progress.setAttribute("aria-valuenow", "0");
      copyStatus.textContent = friendlyError(error);
      stateIcon.textContent = "!";
      stateIcon.classList.add("is-error");
      showToast("上传失败", `${file.name}：${friendlyError(error)}`, "error", 0);
    }
    renderActivity();
    scheduleJobPolling();
  }

  function upsertById(items, incoming) {
    const id = textValue(incoming?.id);
    const index = items.findIndex((item) => textValue(item.id) === id);
    if (index >= 0) items[index] = incoming;
    else items.unshift(incoming);
  }

  async function refreshSourcesAndJobs() {
    if (!state.activeKnowledgeBaseId) return;
    const selectedId = state.activeKnowledgeBaseId;
    const encoded = encodeURIComponent(selectedId);
    try {
      const [sourcePayload, jobPayload] = await Promise.all([
        apiRequest(`/knowledge-bases/${encoded}/sources`),
        apiRequest(`/jobs?knowledgeBaseId=${encoded}&limit=100`),
      ]);
      if (selectedId !== state.activeKnowledgeBaseId) return;
      state.sources = normalizeItems(sourcePayload);
      state.jobs = normalizeItems(jobPayload);
      renderDocuments();
      renderActivity();
      scheduleJobPolling();
    } catch (error) {
      showToast("状态刷新失败", friendlyError(error), "error");
    }
  }

  async function reindexSource(sourceId) {
    if (!state.activeKnowledgeBaseId) return;
    try {
      const result = await apiRequest(`/knowledge-bases/${encodeURIComponent(state.activeKnowledgeBaseId)}/sources/${encodeURIComponent(sourceId)}/reindex`, { method: "POST" });
      if (result?.job) upsertById(state.jobs, result.job);
      renderActivity();
      renderActiveKnowledgeBase();
      scheduleJobPolling();
      showToast("已开始重新处理", "可以继续使用其他功能。");
    } catch (error) {
      showToast("无法重新处理", friendlyError(error), "error", 0);
    }
  }

  async function reindexKnowledgeBase() {
    if (!state.activeKnowledgeBaseId) return;
    elements.reindexButton.disabled = true;
    try {
      const result = await apiRequest(`/knowledge-bases/${encodeURIComponent(state.activeKnowledgeBaseId)}/reindex`, { method: "POST" });
      normalizeItems(result).forEach((job) => upsertById(state.jobs, job));
      renderActivity();
      renderActiveKnowledgeBase();
      scheduleJobPolling();
      showToast("已开始重建索引", `${state.sources.length} 份资料将在后台重新处理。`);
    } catch (error) {
      showToast("无法重建索引", friendlyError(error), "error", 0);
    } finally {
      elements.reindexButton.disabled = false;
    }
  }

  function scheduleJobPolling() {
    stopJobPolling();
    if (!state.jobs.some((job) => ACTIVE_JOB_STATES.has(normalizedStatus(job.status)))) return;
    state.pollTimer = window.setTimeout(pollJobs, 1300);
  }

  function stopJobPolling() {
    if (state.pollTimer) window.clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }

  async function pollJobs() {
    state.pollTimer = null;
    if (!state.activeKnowledgeBaseId || document.hidden) {
      scheduleJobPolling();
      return;
    }
    const before = new Map(state.jobs.map((job) => [textValue(job.id), normalizedStatus(job.status)]));
    try {
      const kbId = encodeURIComponent(state.activeKnowledgeBaseId);
      const payload = await apiRequest(`/jobs?knowledgeBaseId=${kbId}&limit=100`);
      state.jobs = normalizeItems(payload);
      const completed = state.jobs.filter((job) => {
        const oldStatus = before.get(textValue(job.id));
        return oldStatus === "processing" && DONE_JOB_STATES.has(textValue(job.status).toLowerCase());
      });
      const failed = state.jobs.filter((job) => before.get(textValue(job.id)) === "processing" && normalizedStatus(job.status) === "failed");
      if (completed.length || failed.length) {
        const sourcePayload = await apiRequest(`/knowledge-bases/${kbId}/sources`);
        state.sources = normalizeItems(sourcePayload);
        renderDocuments();
        if (completed.length) showToast("资料处理完成", `${completed.length} 个任务已经就绪。`);
        if (failed.length) showToast("有资料处理失败", textValue(failed[0].error, "请在资料列表中查看并重试。"), "error", 0);
      }
      renderActivity();
      renderActiveKnowledgeBase();
    } catch (_error) {
      // A transient polling failure is retried quietly; direct actions still surface errors.
    }
    scheduleJobPolling();
  }

  function renderActivity() {
    const visibleJobs = state.jobs.filter((job) => ACTIVE_JOB_STATES.has(normalizedStatus(job.status)) || normalizedStatus(job.status) === "failed").slice(0, 8);
    if (!visibleJobs.length) {
      elements.activityDrawer.hidden = true;
      return;
    }
    elements.activityDrawer.hidden = false;
    const activeCount = visibleJobs.filter((job) => ACTIVE_JOB_STATES.has(normalizedStatus(job.status))).length;
    elements.activityTitle.textContent = activeCount ? `正在处理 ${activeCount} 个任务` : "任务需要注意";
    elements.activitySubtitle.textContent = activeCount ? "处理在本机后台进行" : "有任务未能完成";
    const rows = visibleJobs.map((job) => {
      const source = state.sources.find((item) => textValue(item.id) === textValue(job.sourceId));
      const name = source ? textValue(source.originalName) : job.kind === "reindex" ? "重建索引" : "处理资料";
      const status = statusPresentation(job.status);
      return createNode("div", { className: "activity-item" }, [
        createNode("div", {}, [createNode("strong", { text: name }), createNode("small", { text: textValue(job.error, formatDate(job.updatedAt)) })]),
        createNode("span", { text: status.label }),
      ]);
    });
    replaceChildren(elements.activityList, ...rows);
  }

  function switchTab(tabName, focus = false) {
    const valid = ["answer", "search", "documents"];
    if (!valid.includes(tabName)) return;
    state.currentTab = tabName;
    valid.forEach((name) => {
      const tab = byId(`tab-${name}`);
      const panel = byId(`panel-${name}`);
      const selected = name === tabName;
      tab.classList.toggle("is-active", selected);
      tab.setAttribute("aria-selected", selected ? "true" : "false");
      tab.tabIndex = selected ? 0 : -1;
      panel.hidden = !selected;
    });
    if (focus) byId(`tab-${tabName}`).focus();
  }

  function currentSearchMode() {
    return state.searchModes?.includes("hybrid_character") ? "hybrid_character" : "bm25";
  }

  async function submitSearch(event) {
    event.preventDefault();
    if (!elements.searchForm.reportValidity() || !state.activeKnowledgeBaseId) return;
    const query = elements.searchInput.value.trim();
    if (!query) return;
    setButtonBusy(elements.searchSubmit, true);
    renderResultLoading(elements.searchResults, false);
    try {
      const result = await apiRequest(`/knowledge-bases/${encodeURIComponent(state.activeKnowledgeBaseId)}/search`, {
        method: "POST",
        body: {
          query,
          topK: Number(elements.searchTopK.value || 5),
          mode: currentSearchMode(),
        },
        timeoutMs: 45000,
      });
      renderSearchResults(normalizeItems(result), query);
    } catch (error) {
      renderResultError(elements.searchResults, "搜索没有完成", error, () => elements.searchForm.requestSubmit());
    } finally {
      setButtonBusy(elements.searchSubmit, false);
    }
  }

  function renderSearchResults(items, query) {
    if (!items.length) {
      replaceChildren(elements.searchResults, createNode("div", { className: "result-empty" }, [
        createNode("strong", { text: "没有找到相关内容" }),
        createNode("span", { text: "换一个更短或更具体的关键词试试。" }),
      ]));
      return;
    }
    const header = createNode("div", { className: "result-header" }, [
      createNode("h3", { text: `“${query}”的搜索结果` }),
      createNode("span", { text: `找到 ${items.length} 条相关内容` }),
    ]);
    const cards = items.map((item, index) => {
      const rawScore = Number(item.score || 0);
      const score = Number.isFinite(rawScore) ? rawScore.toFixed(rawScore >= 10 ? 1 : 3) : "—";
      return createNode("article", { className: "search-result-card" }, [
        createNode("div", { className: "search-result-top" }, [
          createNode("span", { className: "search-result-source", text: `${index + 1}. ${textValue(item.sourceName, "未知来源")}` }),
          createNode("span", { className: "score-label", text: `相关度 ${score}` }),
        ]),
        createNode("p", { text: textValue(item.contents || item.content || item.text, "没有可显示的内容") }),
      ]);
    });
    replaceChildren(elements.searchResults, header, createNode("div", { className: "search-result-list" }, cards));
  }

  async function submitAnswer(event) {
    event.preventDefault();
    if (!elements.answerForm.reportValidity() || !state.activeKnowledgeBaseId) return;
    const question = elements.answerInput.value.trim();
    if (!question) return;
    if (!state.sources.some((source) => normalizedStatus(source.status) === "ready")) {
      showToast("还没有可用资料", "请先添加资料，并等待处理完成。", "warning");
      switchTab("documents", true);
      return;
    }
    setButtonBusy(elements.answerSubmit, true, "正在思考");
    renderResultLoading(elements.answerResults, true);
    try {
      const result = await apiRequest(`/knowledge-bases/${encodeURIComponent(state.activeKnowledgeBaseId)}/answer`, {
        method: "POST",
        body: { question, topK: 5, mode: currentSearchMode() },
        timeoutMs: 120000,
      });
      renderAnswer(result || {});
    } catch (error) {
      renderResultError(elements.answerResults, "暂时无法回答", error, () => elements.answerForm.requestSubmit());
      if (error instanceof ApiError && error.code === "model_not_configured") window.setTimeout(openSettings, 250);
    } finally {
      setButtonBusy(elements.answerSubmit, false, "正在思考");
    }
  }

  function renderAnswer(result) {
    const citations = Array.isArray(result.citations) ? result.citations : [];
    const identity = createNode("div", { className: "answer-identity" }, [
      createNode("span", { text: "AI", attrs: { "aria-hidden": "true" } }),
      createNode("span", { text: "基于资料的回答" }),
    ]);
    const header = createNode("div", { className: "answer-card-header" }, [
      identity,
      createNode("span", { className: "answer-meta", text: textValue(result.model, "本地模型") }),
    ]);
    const cardChildren = [header, createNode("p", { className: "answer-content", text: textValue(result.content, "模型没有返回内容。") })];
    if (citations.length) {
      cardChildren.push(createNode("div", { className: "citations-title", text: `引用来源 · ${citations.length}` }));
      cardChildren.push(createNode("div", { className: "citation-list" }, citations.map((citation, index) => createCitationCard(citation, index))));
    }
    replaceChildren(elements.answerResults, createNode("article", { className: "answer-card" }, cardChildren));
  }

  function createCitationCard(citation, fallbackIndex) {
    const index = Number(citation.index || fallbackIndex + 1);
    return createNode("div", { className: "citation-card" }, [
      createNode("div", { className: "citation-source" }, [
        createNode("span", { className: "citation-index", text: Number.isFinite(index) ? index : fallbackIndex + 1 }),
        createNode("span", { className: "citation-name", text: textValue(citation.sourceName, "未知来源"), title: textValue(citation.sourceName, "未知来源") }),
      ]),
      createNode("p", { text: textValue(citation.excerpt, "没有可显示的摘录") }),
    ]);
  }

  function renderResultLoading(container, answer) {
    const count = answer ? 1 : 3;
    const blocks = Array.from({ length: count }, () => createNode("div", { className: `skeleton-result${answer ? " answer-skeleton" : ""}` }));
    replaceChildren(container, createNode("div", { className: "skeleton-stack", attrs: { "aria-label": answer ? "正在生成回答" : "正在搜索" } }, blocks));
  }

  function renderResultError(container, title, error, retry) {
    const button = createNode("button", { className: "button button-secondary", type: "button", text: "重试" });
    button.addEventListener("click", retry);
    replaceChildren(container, createNode("div", { className: "result-error" }, [
      createNode("strong", { text: title }),
      createNode("span", { text: friendlyError(error) }),
      button,
    ]));
  }

  async function openSettings() {
    showDialog(elements.settingsDialog, elements.modelBaseUrl);
    populateModelSettings(state.modelSettings);
    elements.modelTestResult.hidden = true;
    try {
      const settings = await apiRequest("/model-settings");
      state.modelSettings = settings || {};
      populateModelSettings(state.modelSettings);
    } catch (error) {
      showToast("无法读取模型设置", friendlyError(error), "error");
    }
  }

  function populateModelSettings(settings) {
    elements.modelBaseUrl.value = textValue(settings.baseUrl || settings.base_url);
    elements.modelName.value = textValue(settings.model);
    elements.modelApiKey.value = "";
    elements.clearApiKey.checked = false;
    const configured = Boolean(settings.apiKeyConfigured ?? settings.api_key_configured);
    elements.apiKeyState.textContent = configured ? "已保存在本机" : "可选";
    elements.clearKeyWrap.hidden = !configured;
    elements.modelApiKey.placeholder = configured ? "留空则保留现有密钥" : "无密钥的本地模型可留空";
    renderModelConnection();
  }

  function collectModelSettings() {
    const payload = {
      baseUrl: elements.modelBaseUrl.value.trim(),
      model: elements.modelName.value.trim(),
      clearApiKey: elements.clearApiKey.checked,
    };
    const apiKey = elements.modelApiKey.value.trim();
    if (apiKey) payload.apiKey = apiKey;
    return payload;
  }

  async function saveModelSettings(event, options = {}) {
    if (event) event.preventDefault();
    if (!elements.settingsForm.reportValidity()) return null;
    if (elements.clearApiKey.checked && elements.modelApiKey.value.trim()) {
      showToast("请检查密钥设置", "不能同时填写新密钥并选择清除密钥。", "warning");
      return null;
    }
    elements.settingsSave.disabled = true;
    try {
      const result = await apiRequest("/model-settings", { method: "PUT", body: collectModelSettings() });
      state.modelSettings = result || {};
      populateModelSettings(state.modelSettings);
      if (!options.silent) {
        closeDialog(elements.settingsDialog);
        showToast("模型设置已保存", "现在可以在知识库中使用 AI 问答。");
      }
      return result;
    } catch (error) {
      showToast("模型设置保存失败", friendlyError(error), "error", 0);
      return null;
    } finally {
      elements.settingsSave.disabled = false;
    }
  }

  async function testModelConnection() {
    if (!elements.settingsForm.reportValidity()) return;
    setButtonBusy(elements.modelTest, true, "正在连接");
    elements.modelTestResult.hidden = false;
    elements.modelTestResult.className = "connection-test field-full";
    elements.modelTestResult.textContent = "正在保存设置并测试连接…";
    const saved = await saveModelSettings(null, { silent: true });
    if (!saved) {
      setButtonBusy(elements.modelTest, false, "正在连接");
      return;
    }
    try {
      const result = await apiRequest("/model-settings/test", { method: "POST", timeoutMs: 60000 });
      elements.modelTestResult.className = "connection-test field-full is-success";
      elements.modelTestResult.textContent = `${textValue(result?.message, "连接成功。")} ${result?.model ? `模型：${textValue(result.model)}` : ""}`.trim();
    } catch (error) {
      elements.modelTestResult.className = "connection-test field-full is-error";
      elements.modelTestResult.textContent = friendlyError(error);
    } finally {
      setButtonBusy(elements.modelTest, false, "正在连接");
    }
  }

  function showDialog(dialog, initialFocus) {
    if (dialog.open) return;
    dialog.showModal();
    document.body.classList.add("modal-open");
    window.setTimeout(() => initialFocus?.focus(), 0);
  }

  function closeDialog(dialog) {
    if (dialog.open) dialog.close();
    if (!document.querySelector("dialog[open]")) document.body.classList.remove("modal-open");
  }

  function toggleSidebar() {
    const open = !elements.sidebar.classList.contains("is-open");
    elements.sidebar.classList.toggle("is-open", open);
    elements.sidebarBackdrop.hidden = !open;
    elements.sidebarToggle.setAttribute("aria-expanded", open ? "true" : "false");
  }

  function closeSidebar() {
    elements.sidebar.classList.remove("is-open");
    elements.sidebarBackdrop.hidden = true;
    elements.sidebarToggle.setAttribute("aria-expanded", "false");
  }

  function applyTheme(theme) {
    const selected = theme || readStorage(STORAGE_KEYS.theme) || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    document.documentElement.dataset.theme = selected;
    document.querySelector("meta[name='theme-color']")?.setAttribute("content", selected === "dark" ? "#0e1016" : "#f6f7fb");
    elements.themeToggle.setAttribute("aria-label", selected === "dark" ? "切换浅色主题" : "切换深色主题");
    elements.themeToggle.title = selected === "dark" ? "使用浅色主题" : "使用深色主题";
  }

  function toggleTheme() {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    writeStorage(STORAGE_KEYS.theme, next);
    applyTheme(next);
  }

  function handleTabKeydown(event) {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    const tabs = ["answer", "search", "documents"];
    let index = tabs.indexOf(state.currentTab);
    if (event.key === "ArrowRight") index = (index + 1) % tabs.length;
    if (event.key === "ArrowLeft") index = (index - 1 + tabs.length) % tabs.length;
    if (event.key === "Home") index = 0;
    if (event.key === "End") index = tabs.length - 1;
    event.preventDefault();
    switchTab(tabs[index], true);
  }

  function bindEvents() {
    document.querySelectorAll("[data-section]").forEach((button) => {
      button.addEventListener("click", () => switchSection(button.dataset.section));
    });
    document.querySelectorAll("[data-section-target]").forEach((button) => {
      button.addEventListener("click", () => switchSection(button.dataset.sectionTarget));
    });
    document.querySelectorAll("[data-copy]").forEach((button) => {
      button.addEventListener("click", () => copyCommand(button));
    });
    elements.runDemoButton.addEventListener("click", runTrajectoryDemo);
    elements.quickDemoButton.addEventListener("click", runTrajectoryDemo);
    elements.newOfflineRunButton.addEventListener("click", openOfflineRunDialog);
    elements.offlineRunForm.addEventListener("submit", createOfflineRun);
    elements.newDatasetButton.addEventListener("click", openDatasetDialog);
    elements.datasetForm.addEventListener("submit", createRLDataset);
    elements.refreshRunsButton.addEventListener("click", () => loadRLRuns());
    elements.trainingRefreshButton.addEventListener("click", () => loadRLRuns());
    elements.diagnosticsRefreshButton.addEventListener("click", () => loadRLOverview());
    elements.modelsSettingsButton.addEventListener("click", openSettings);
    elements.modelCardSettingsButton.addEventListener("click", openSettings);
    elements.themeToggle.addEventListener("click", toggleTheme);
    elements.sidebarToggle.addEventListener("click", toggleSidebar);
    elements.sidebarBackdrop.addEventListener("click", closeSidebar);
    elements.createLibraryIcon.addEventListener("click", () => openLibraryDialog());
    elements.createLibraryButton.addEventListener("click", () => openLibraryDialog());
    elements.welcomeCreate.addEventListener("click", () => openLibraryDialog());
    elements.settingsOpen.addEventListener("click", openSettings);
    elements.welcomeSettings.addEventListener("click", openSettings);
    elements.libraryForm.addEventListener("submit", saveKnowledgeBase);
    elements.confirmForm.addEventListener("submit", runConfirmedAction);
    elements.settingsForm.addEventListener("submit", saveModelSettings);
    elements.modelTest.addEventListener("click", testModelConnection);
    elements.retryBootstrap.addEventListener("click", initializeData);
    elements.reindexButton.addEventListener("click", reindexKnowledgeBase);
    elements.renameLibrary.addEventListener("click", () => {
      elements.libraryMenu.hidden = true;
      openLibraryDialog(activeKnowledgeBase());
    });
    elements.deleteLibrary.addEventListener("click", () => {
      elements.libraryMenu.hidden = true;
      requestKnowledgeBaseDeletion();
    });
    elements.libraryMenuButton.addEventListener("click", () => {
      const open = elements.libraryMenu.hidden;
      elements.libraryMenu.hidden = !open;
      elements.libraryMenuButton.setAttribute("aria-expanded", open ? "true" : "false");
    });
    document.addEventListener("click", (event) => {
      if (!elements.libraryMenu.hidden && !elements.libraryMenu.contains(event.target) && event.target !== elements.libraryMenuButton) {
        elements.libraryMenu.hidden = true;
        elements.libraryMenuButton.setAttribute("aria-expanded", "false");
      }
    });

    document.querySelectorAll("[data-close-dialog]").forEach((button) => {
      button.addEventListener("click", () => closeDialog(byId(button.dataset.closeDialog)));
    });
    document.querySelectorAll("dialog").forEach((dialog) => {
      dialog.addEventListener("close", () => {
        if (!document.querySelector("dialog[open]")) document.body.classList.remove("modal-open");
      });
      dialog.addEventListener("click", (event) => {
        if (event.target === dialog) closeDialog(dialog);
      });
    });

    document.querySelectorAll(".workspace-tab").forEach((tab) => {
      tab.addEventListener("click", () => switchTab(tab.dataset.tab));
      tab.addEventListener("keydown", handleTabKeydown);
    });
    elements.answerForm.addEventListener("submit", submitAnswer);
    elements.searchForm.addEventListener("submit", submitSearch);
    elements.answerInput.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        elements.answerForm.requestSubmit();
      }
    });
    elements.answerSuggestions.querySelectorAll("button").forEach((button) => {
      button.addEventListener("click", () => {
        elements.answerInput.value = button.textContent || "";
        elements.answerInput.focus();
      });
    });

    const openFilePicker = () => {
      switchTab("documents");
      elements.fileInput.click();
    };
    elements.addDocumentsButton.addEventListener("click", openFilePicker);
    elements.documentsAddButton.addEventListener("click", openFilePicker);
    elements.emptyAddButton.addEventListener("click", openFilePicker);
    elements.dropZone.addEventListener("click", () => elements.fileInput.click());
    elements.dropZone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        elements.fileInput.click();
      }
    });
    elements.fileInput.addEventListener("change", () => uploadFiles(elements.fileInput.files));
    ["dragenter", "dragover"].forEach((name) => elements.dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      elements.dropZone.classList.add("is-dragging");
    }));
    ["dragleave", "drop"].forEach((name) => elements.dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      elements.dropZone.classList.remove("is-dragging");
    }));
    elements.dropZone.addEventListener("drop", (event) => uploadFiles(event.dataTransfer?.files));
    elements.documentFilter.addEventListener("input", renderDocuments);
    elements.activityToggle.addEventListener("click", () => {
      const expanded = elements.activityToggle.getAttribute("aria-expanded") === "true";
      elements.activityToggle.setAttribute("aria-expanded", expanded ? "false" : "true");
      elements.activityList.hidden = expanded;
    });
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) {
        scheduleJobPolling();
        scheduleRLPolling();
      }
    });
    document.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k" && !document.querySelector("dialog[open]")) {
        event.preventDefault();
        if (state.activeKnowledgeBaseId) {
          switchSection("knowledge", { focus: false });
          switchTab("search");
          elements.searchInput.focus();
        }
      }
      if (event.key === "Escape") closeSidebar();
    });
  }

  function cacheElements() {
    const ids = [
      "main-content", "training-nav-status", "knowledge-nav-count", "overview-kb-count", "rl-core-badge", "rl-engine-status",
      "rl-engine-detail", "rl-eval-status", "rl-training-status", "rl-training-detail", "run-demo-button", "quick-demo-button",
      "demo-result", "refresh-runs-button", "overview-run-list", "new-offline-run-button", "metric-running-runs",
      "metric-completed-runs", "metric-failed-runs", "metric-training-runtime", "training-refresh-button", "training-run-list",
      "environment-page-status", "diagnostics-refresh-button", "diagnostics-list", "new-dataset-button", "rl-dataset-list",
      "models-settings-button", "model-card-settings-button", "assistant-model-badge", "assistant-model-name",
      "assistant-model-endpoint", "trainer-model-badge", "trainer-runtime-detail", "offline-run-dialog", "offline-run-form",
      "offline-dataset-select", "offline-rollout-count", "offline-run-submit", "dataset-dialog", "dataset-form", "dataset-name",
      "dataset-qa-file", "dataset-corpus-file", "dataset-submit",
      "connection-status", "connection-label", "theme-toggle", "settings-open", "sidebar-toggle", "sidebar-backdrop", "sidebar",
      "create-library-icon", "library-list", "library-list-empty", "library-list-loading", "create-library-button", "app-version",
      "welcome-view", "welcome-create", "welcome-settings", "workspace", "library-avatar", "library-title", "library-description",
      "library-index-status", "reindex-button", "add-documents-button", "library-menu-button", "library-menu", "rename-library", "delete-library",
      "stat-sources", "stat-ready", "stat-processing", "stat-updated", "documents-tab-count", "answer-form", "answer-input", "answer-submit",
      "answer-suggestions", "answer-results", "search-form", "search-input", "search-submit", "search-top-k", "search-results",
      "documents-add-button", "drop-zone", "file-input", "upload-help", "upload-queue", "document-filter", "document-summary", "document-list",
      "documents-empty", "empty-add-button", "activity-drawer", "activity-toggle", "activity-title", "activity-subtitle", "activity-list", "toast-region",
      "library-dialog", "library-form", "library-dialog-title", "library-name", "library-description-input", "library-save", "settings-dialog",
      "settings-form", "model-base-url", "model-name", "model-api-key", "api-key-state", "clear-key-wrap", "clear-api-key", "model-test-result",
      "model-test", "settings-save", "confirm-dialog", "confirm-form", "confirm-title", "confirm-message", "confirm-action", "fatal-state",
      "fatal-message", "retry-bootstrap",
    ];
    ids.forEach((id) => {
      const key = id.replace(/-([a-z])/g, (_match, letter) => letter.toUpperCase());
      elements[key] = byId(id);
    });
  }

  async function initializeData() {
    elements.fatalState.hidden = true;
    try {
      await bootstrapSession(true);
      await Promise.all([loadKnowledgeBases(), loadRLWorkspace()]);
    } catch (_error) {
      // bootstrapSession already renders a persistent, actionable connection error.
    }
  }

  function initialize() {
    document.body.classList.add("studio-runtime");
    cacheElements();
    applyTheme();
    bindEvents();
    switchSection(readStorage(STORAGE_KEYS.activeSection) || "overview", { focus: false, scroll: false });
    initializeData();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize, { once: true });
  else initialize();
})();
