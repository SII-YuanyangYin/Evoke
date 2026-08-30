const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

async function readResponsePayload(response) {
  const body = await response.text();
  if (!body) return {};
  try {
    return JSON.parse(body);
  } catch {
    return {detail: `${response.status} ${response.statusText}: ${body}`};
  }
}

const state = {
  chunks: 4,
  image: null,
  pathPoints: [],
  lookPoints: [{x: .5, y: .5}, {x: .5, y: .5}],
  drawing: false,
  lookDrawing: false,
  eventSource: null,
  pollTimer: null,
  jobId: null,
  segmentCount: 0,
  selectedSegment: null,
  autoFollowSegments: true,
  jobStatus: "idle",
  projectId: null,
  revisionId: null,
  lastProgressAt: 0,
  lastProgressDetail: null,
  continuationBaseChunks: 0,
  segmentUrls: [],
  fullPlaylist: null,
  fullPlaylistIndex: -1,
  playlistActiveSlot: 0,
  playlistGeneration: 0,
  playlistTransitioning: false,
  fullDownloadUrl: null,
};

let promptExamples = [
  "A quiet moss-covered forest opens ahead, soft morning light passing through tall ancient trees, cinematic realism, natural colors.",
  "The path continues deeper into the forest as tiny luminous particles begin to drift between the trees, subtle and believable.",
  "A shallow stream appears across the trail, reflecting the canopy while the light gradually turns warmer and more golden.",
  "The forest opens onto a vast cliff edge above a sea of clouds, distant mountains emerging through the mist, epic wide landscape.",
];

const presets = {
  push: [{x: .5, y: .84}, {x: .5, y: .64}, {x: .5, y: .4}, {x: .5, y: .16}],
  orbit: Array.from({length: 30}, (_, i) => {
    const angle = Math.PI * (1.1 + i / 29 * 1.35);
    return {x: .5 + Math.cos(angle) * .3, y: .5 + Math.sin(angle) * .3};
  }),
  truck: [{x: .16, y: .52}, {x: .38, y: .48}, {x: .62, y: .48}, {x: .84, y: .52}],
  sweep: Array.from({length: 32}, (_, i) => {
    const t = i / 31;
    return {x: .18 + t * .64, y: .78 - t * .58 + Math.sin(t * Math.PI * 2) * .12};
  }),
  static: [{x: .5, y: .5}, {x: .5, y: .5}],
};

const lookPresets = {
  center: [{x: .5, y: .5}, {x: .5, y: .5}],
  left: [{x: .5, y: .5}, {x: .2, y: .5}],
  right: [{x: .5, y: .5}, {x: .8, y: .5}],
  up: [{x: .5, y: .5}, {x: .5, y: .18}],
  down: [{x: .5, y: .5}, {x: .5, y: .82}],
};

const canvas = $("#trajectoryCanvas");
const context = canvas.getContext("2d");
const lookCanvas = $("#lookCanvas");
const lookContext = lookCanvas.getContext("2d");

function resizeCanvas() {
  const ratio = window.devicePixelRatio || 1;
  for (const [target, targetContext] of [[canvas, context], [lookCanvas, lookContext]]) {
    const rect = target.getBoundingClientRect();
    target.width = Math.round(rect.width * ratio);
    target.height = Math.round(rect.height * ratio);
    targetContext.setTransform(ratio, 0, 0, ratio, 0, 0);
  }
  drawPath();
  drawLookPath();
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
    y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
  };
}

function lookCanvasPoint(event) {
  const rect = lookCanvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
    y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
  };
}

function pointAtFraction(points, fraction) {
  if (points.length < 2) return points[0] || {x: .5, y: .5};
  const lengths = points.slice(1).map((point, index) => Math.hypot(point.x - points[index].x, point.y - points[index].y));
  const total = lengths.reduce((sum, length) => sum + length, 0);
  if (!total) return points[0];
  let target = total * fraction;
  for (let index = 0; index < lengths.length; index += 1) {
    if (target <= lengths[index]) {
      const mix = lengths[index] ? target / lengths[index] : 0;
      return {
        x: points[index].x + (points[index + 1].x - points[index].x) * mix,
        y: points[index].y + (points[index + 1].y - points[index].y) * mix,
      };
    }
    target -= lengths[index];
  }
  return points.at(-1);
}

function drawPath() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "rgba(219, 235, 222, .07)";
  context.lineWidth = 1;
  for (let x = 0; x <= width; x += 28) { context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke(); }
  for (let y = 0; y <= height; y += 28) { context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke(); }
  if (state.pathPoints.length < 1) return;

  const gradient = context.createLinearGradient(0, height, width, 0);
  gradient.addColorStop(0, "#73ded2");
  gradient.addColorStop(1, "#bdfc68");
  context.strokeStyle = gradient;
  context.lineWidth = 2.2;
  context.lineCap = "round";
  context.lineJoin = "round";
  context.shadowColor = "rgba(189,252,104,.35)";
  context.shadowBlur = 8;
  context.beginPath();
  state.pathPoints.forEach((point, index) => {
    const x = point.x * width, y = point.y * height;
    if (!index) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.stroke();
  context.shadowBlur = 0;

  const controlChunks = Math.max(1, state.chunks - state.continuationBaseChunks || state.chunks);
  for (let chunk = 0; chunk <= controlChunks; chunk += 1) {
    const point = pointAtFraction(state.pathPoints, chunk / controlChunks);
    const x = point.x * width, y = point.y * height;
    context.beginPath();
    context.fillStyle = chunk === 0 ? "#73ded2" : chunk === controlChunks ? "#ff8a4c" : "#0d100e";
    context.strokeStyle = chunk === controlChunks ? "#ff8a4c" : "#bdfc68";
    context.lineWidth = 1.4;
    context.arc(x, y, chunk === 0 || chunk === controlChunks ? 4.4 : 3.3, 0, Math.PI * 2);
    context.fill(); context.stroke();
    if (chunk > 0 && chunk < controlChunks) {
      context.fillStyle = "#8c978f";
      context.font = "7px ui-monospace, monospace";
      context.fillText(String(chunk + 1).padStart(2, "0"), x + 6, y - 5);
    }
  }
}

function usePreset(name) {
  state.pathPoints = presets[name].map(point => ({...point}));
  $$(".preset").forEach(button => button.classList.toggle("active", button.dataset.preset === name));
  drawPath();
}

function drawLookPath() {
  const width = lookCanvas.clientWidth;
  const height = lookCanvas.clientHeight;
  lookContext.clearRect(0, 0, width, height);
  lookContext.strokeStyle = "rgba(219, 235, 222, .07)";
  lookContext.lineWidth = 1;
  for (let x = 0; x <= width; x += 28) { lookContext.beginPath(); lookContext.moveTo(x, 0); lookContext.lineTo(x, height); lookContext.stroke(); }
  for (let y = 0; y <= height; y += 28) { lookContext.beginPath(); lookContext.moveTo(0, y); lookContext.lineTo(width, y); lookContext.stroke(); }
  lookContext.strokeStyle = "rgba(115, 222, 210, .28)";
  lookContext.setLineDash([4, 5]);
  lookContext.beginPath(); lookContext.moveTo(width / 2, 0); lookContext.lineTo(width / 2, height); lookContext.stroke();
  lookContext.beginPath(); lookContext.moveTo(0, height / 2); lookContext.lineTo(width, height / 2); lookContext.stroke();
  lookContext.setLineDash([]);
  if (!state.lookPoints.length) return;

  const gradient = lookContext.createLinearGradient(0, height, width, 0);
  gradient.addColorStop(0, "#73ded2"); gradient.addColorStop(1, "#bdfc68");
  lookContext.strokeStyle = gradient;
  lookContext.lineWidth = 2.2;
  lookContext.lineCap = "round";
  lookContext.lineJoin = "round";
  lookContext.beginPath();
  state.lookPoints.forEach((point, index) => {
    const x = point.x * width, y = point.y * height;
    if (!index) lookContext.moveTo(x, y); else lookContext.lineTo(x, y);
  });
  lookContext.stroke();

  const controlChunks = Math.max(1, state.chunks - state.continuationBaseChunks || state.chunks);
  for (let chunk = 0; chunk <= controlChunks; chunk += 1) {
    const point = pointAtFraction(state.lookPoints, chunk / controlChunks);
    const x = point.x * width, y = point.y * height;
    lookContext.beginPath();
    lookContext.fillStyle = chunk === 0 ? "#73ded2" : chunk === controlChunks ? "#ff8a4c" : "#0d100e";
    lookContext.strokeStyle = chunk === controlChunks ? "#ff8a4c" : "#bdfc68";
    lookContext.lineWidth = 1.4;
    lookContext.arc(x, y, chunk === 0 || chunk === controlChunks ? 4.4 : 3.3, 0, Math.PI * 2);
    lookContext.fill(); lookContext.stroke();
  }
}

function useLookPreset(name) {
  state.lookPoints = lookPresets[name].map(point => ({...point}));
  $$(".look-preset").forEach(button => button.classList.toggle("active", button.dataset.lookPreset === name));
  drawLookPath();
}

canvas.addEventListener("pointerdown", event => {
  state.drawing = true;
  state.pathPoints = [canvasPoint(event)];
  canvas.setPointerCapture(event.pointerId);
  $$(".preset").forEach(button => button.classList.remove("active"));
  drawPath();
});
canvas.addEventListener("pointermove", event => {
  if (!state.drawing) return;
  const point = canvasPoint(event);
  const last = state.pathPoints.at(-1);
  if (!last || Math.hypot(point.x - last.x, point.y - last.y) > .008) state.pathPoints.push(point);
  drawPath();
});
canvas.addEventListener("pointerup", event => {
  state.drawing = false;
  if (state.pathPoints.length === 1) state.pathPoints.push({...state.pathPoints[0]});
  canvas.releasePointerCapture(event.pointerId);
  drawPath();
});
canvas.addEventListener("pointercancel", () => { state.drawing = false; });

lookCanvas.addEventListener("pointerdown", event => {
  state.lookDrawing = true;
  const point = lookCanvasPoint(event);
  state.lookPoints = [{x: .5, y: .5}, point];
  lookCanvas.setPointerCapture(event.pointerId);
  $$(".look-preset").forEach(button => button.classList.remove("active"));
  drawLookPath();
});
lookCanvas.addEventListener("pointermove", event => {
  if (!state.lookDrawing) return;
  const point = lookCanvasPoint(event);
  const last = state.lookPoints.at(-1);
  if (!last || Math.hypot(point.x - last.x, point.y - last.y) > .008) state.lookPoints.push(point);
  drawLookPath();
});
lookCanvas.addEventListener("pointerup", event => {
  state.lookDrawing = false;
  lookCanvas.releasePointerCapture(event.pointerId);
  drawLookPath();
});
lookCanvas.addEventListener("pointercancel", () => { state.lookDrawing = false; });

function renderPrompts() {
  const list = $("#promptList");
  const existing = $$("textarea", list).map(input => input.value);
  list.innerHTML = "";
  for (let index = 0; index < state.chunks; index += 1) {
    const card = $("#promptTemplate").content.firstElementChild.cloneNode(true);
    $(".prompt-index strong", card).textContent = String(index + 1).padStart(2, "0");
    $(".prompt-index small", card).textContent = `${(index * 1.5).toFixed(1)}S`;
    const extending = state.continuationBaseChunks > 0 && state.chunks > state.continuationBaseChunks;
    const retained = extending && index < state.continuationBaseChunks;
    $(".prompt-copy label", card).textContent = retained
      ? `第 ${index + 1} 段 · 已保留`
      : extending ? `第 ${index + 1} 段 · 续写描述` : index ? `第 ${index + 1} 段画面描述` : "建立世界与主体";
    const textarea = $("textarea", card);
    textarea.value = existing[index] ?? promptExamples[index] ?? "";
    textarea.placeholder = index ? "描述这一段希望发生的变化…" : "描述起始场景、主体、光线与风格…";
    textarea.disabled = retained;
    card.classList.toggle("retained", retained);
    const count = $(".char-count", card);
    const updateCount = () => { count.textContent = `${textarea.value.length} / 4000`; };
    textarea.addEventListener("input", updateCount);
    updateCount();
    $(".copy-previous", card).addEventListener("click", () => {
      textarea.value = $$("textarea", list)[index - 1]?.value || "";
      updateCount();
      textarea.focus();
    });
    list.append(card);
  }
}

function updateTimeline() {
  $("#chunkCount").textContent = state.chunks;
  $("#totalDuration").textContent = `${(state.chunks * 1.5).toFixed(1)} 秒`;
  const appended = Math.max(0, state.chunks - state.continuationBaseChunks);
  $("#coverageDuration").textContent = state.continuationBaseChunks && appended
    ? `续写 ${(appended * 1.5).toFixed(1)}S`
    : `${(state.chunks * 1.5).toFixed(1)}S`;
  $("#moveDrawHelp").textContent = state.continuationBaseChunks && appended
    ? `此轨迹从 CH ${String(state.continuationBaseChunks).padStart(2, "0")} 片尾继续，仅控制新增段`
    : "拖动绘制移动轨迹";
  $("#lookDrawHelp").textContent = state.continuationBaseChunks && appended
    ? "从当前镜头方向继续，仅控制新增段"
    : "从中心拖动视角轨迹";
  $("#frameCount").textContent = String(36 * state.chunks - 3);
  $("#progressFraction").textContent = `${state.segmentCount} / ${state.chunks}`;
  const ruler = $("#timelineRuler");
  ruler.innerHTML = "";
  for (let i = 0; i < state.chunks; i += 1) {
    const cell = document.createElement("div");
    cell.className = "ruler-cell";
    cell.textContent = `${(i * 1.5).toFixed(1)}s`;
    ruler.append(cell);
  }
  renderPrompts();
  drawPath();
  drawLookPath();
  updateGenerateAction();
}

function setChunks(next) {
  const minimum = state.jobId && state.jobStatus === "complete" ? state.continuationBaseChunks : 1;
  state.chunks = Math.max(minimum || 1, Math.min(40, next));
  updateTimeline();
}

function setImage(file, resetSession = true) {
  if (!file || !file.type.startsWith("image/")) return;
  if (resetSession && state.jobId) {
    state.jobId = null;
    state.projectId = null;
    state.revisionId = null;
    state.jobStatus = "idle";
    state.continuationBaseChunks = 0;
    updateTimeline();
  }
  state.image = file;
  const preview = $("#referencePreview");
  preview.style.objectFit = "contain";
  preview.style.objectPosition = "center";
  preview.src = URL.createObjectURL(file);
  $("#dropZone").classList.add("has-image");
  $("#formError").textContent = "";
  updateGenerateAction();
}

function updateGenerateAction() {
  const button = $("#generateButton");
  if (!button) return;
  const extending = state.jobId && state.jobStatus === "complete" && state.chunks > state.continuationBaseChunks;
  const completeWithoutExtension = state.jobId && state.jobStatus === "complete" && !extending;
  if (!["queued", "running", "cancelling"].includes(state.jobStatus)) {
    button.disabled = Boolean(completeWithoutExtension);
  }
  $("span", button).textContent = extending
    ? `续写 CH ${String(state.continuationBaseChunks + 1).padStart(2, "0")}–${String(state.chunks).padStart(2, "0")}`
    : completeWithoutExtension ? "点击 + 增加 CHUNK 后续写" : "开始流式生成";
}

function selectOutput(url, key, pin = false) {
  state.playlistGeneration += 1;
  state.fullPlaylist = null;
  state.fullPlaylistIndex = -1;
  state.playlistActiveSlot = 0;
  state.playlistTransitioning = false;
  const video = $("#finalVideo");
  const buffer = $("#playlistBuffer");
  buffer.pause();
  buffer.controls = false;
  buffer.className = "";
  buffer.removeAttribute("src");
  buffer.load();
  const changed = video.dataset.outputUrl !== url;
  if (changed) {
    video.pause();
    video.dataset.outputUrl = url;
    video.preload = "auto";
    video.defaultMuted = true;
    video.muted = true;
    video.src = url;
    video.load();
  }
  video.controls = true;
  video.className = "visible playlist-active";
  const playPreview = () => video.play().catch(() => {});
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) playPreview();
  else video.addEventListener("canplay", playPreview, {once: true});
  $("#monitorEmpty").hidden = true;
  state.selectedSegment = key;
  if (pin) state.autoFollowSegments = false;
  $$(".segment-item", $("#segmentStream")).forEach(item => {
    item.classList.toggle("active", item.dataset.segment === String(key));
  });
  updateRetryButton();
  updateFullDownloadMenu();
}

function playlistVideoSlots() {
  return [$("#finalVideo"), $("#playlistBuffer")];
}

function clearPlaybackVideos() {
  state.playlistGeneration += 1;
  state.fullPlaylist = null;
  state.fullPlaylistIndex = -1;
  state.playlistActiveSlot = 0;
  state.playlistTransitioning = false;
  for (const video of playlistVideoSlots()) {
    video.pause();
    video.controls = false;
    video.className = "";
    video.removeAttribute("src");
    video.removeAttribute("data-output-url");
    video.removeAttribute("data-playlist-index");
    video.load();
  }
}

function prepareNextPlaylistItem(generation = state.playlistGeneration) {
  const urls = state.fullPlaylist;
  if (!urls?.length || generation !== state.playlistGeneration) return;
  const nextIndex = (state.fullPlaylistIndex + 1) % urls.length;
  const standbySlot = 1 - state.playlistActiveSlot;
  const standby = playlistVideoSlots()[standbySlot];
  standby.pause();
  standby.controls = false;
  standby.className = "visible playlist-standby";
  standby.dataset.playlistIndex = String(nextIndex);
  standby.dataset.outputUrl = `playlist:${nextIndex}:${urls[nextIndex]}`;
  standby.preload = "auto";
  standby.defaultMuted = true;
  standby.muted = true;
  standby.src = urls[nextIndex];
  standby.load();
}

function playPlaylistItem(index, autoplay = true) {
  const urls = state.fullPlaylist;
  if (!urls?.length || index < 0 || index >= urls.length) return;
  const generation = state.playlistGeneration;
  state.fullPlaylistIndex = index;
  state.playlistActiveSlot = 0;
  state.playlistTransitioning = false;
  const [video, buffer] = playlistVideoSlots();
  buffer.pause();
  buffer.controls = false;
  buffer.className = "visible playlist-standby";
  video.pause();
  video.controls = true;
  video.className = "visible playlist-active";
  video.dataset.playlistIndex = String(index);
  video.dataset.outputUrl = `playlist:${index}:${urls[index]}`;
  video.preload = "auto";
  video.defaultMuted = true;
  video.muted = true;
  video.src = urls[index];
  video.load();
  $("#monitorEmpty").hidden = true;
  prepareNextPlaylistItem(generation);
  if (autoplay) {
    const play = () => {
      if (generation === state.playlistGeneration) video.play().catch(() => {});
    };
    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) play();
    else video.addEventListener("canplay", play, {once: true});
  }
}

function advanceFullPlaylist(sourceVideo) {
  const urls = state.fullPlaylist;
  if (!urls?.length || state.playlistTransitioning) return;
  const slots = playlistVideoSlots();
  if (sourceVideo !== slots[state.playlistActiveSlot]) return;
  const generation = state.playlistGeneration;
  const currentIndex = state.fullPlaylistIndex;
  const nextIndex = (currentIndex + 1) % urls.length;
  const continuePlaying = currentIndex + 1 < urls.length;
  const nextSlot = 1 - state.playlistActiveSlot;
  const nextVideo = slots[nextSlot];
  state.playlistTransitioning = true;

  if (Number(nextVideo.dataset.playlistIndex) !== nextIndex) prepareNextPlaylistItem(generation);

  const reveal = () => {
    if (generation !== state.playlistGeneration || !state.fullPlaylist) return;
    const previous = slots[state.playlistActiveSlot];
    previous.controls = false;
    previous.className = "visible playlist-standby";
    nextVideo.controls = true;
    nextVideo.className = "visible playlist-active";
    state.playlistActiveSlot = nextSlot;
    state.fullPlaylistIndex = nextIndex;
    state.playlistTransitioning = false;
    prepareNextPlaylistItem(generation);
  };

  const revealAfterDecodedFrame = () => {
    const ready = () => {
      if (!continuePlaying) nextVideo.pause();
      reveal();
    };
    if (typeof nextVideo.requestVideoFrameCallback === "function") {
      nextVideo.requestVideoFrameCallback(ready);
    } else {
      requestAnimationFrame(ready);
    }
  };

  const start = () => {
    if (generation !== state.playlistGeneration) return;
    nextVideo.play().then(revealAfterDecodedFrame).catch(() => reveal());
  };
  if (nextVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) start();
  else nextVideo.addEventListener("canplay", start, {once: true});
}

function selectFullPlaylist(urls, pin = false) {
  if (!urls?.length) return;
  state.playlistGeneration += 1;
  state.fullPlaylist = [...urls];
  state.fullPlaylistIndex = 0;
  state.selectedSegment = "full";
  if (pin) state.autoFollowSegments = false;
  $$(".segment-item", $("#segmentStream")).forEach(item => {
    item.classList.toggle("active", item.dataset.segment === "full");
  });
  updateRetryButton();
  playPlaylistItem(0);
  updateFullDownloadMenu();
}

function updateRetryButton() {
  const button = $("#retryChunkButton");
  const selected = Number(state.selectedSegment);
  const isChunk = state.selectedSegment !== null && state.selectedSegment !== "full" && Number.isInteger(selected);
  button.disabled = state.jobStatus !== "complete" || !isChunk;
  button.textContent = isChunk ? `从 CH ${String(selected + 1).padStart(2, "0")} 起重做` : "从此段重做";
}

function updateFullDownloadMenu() {
  const menu = $("#fullDownloadMenu");
  const link = $("#downloadFullAction");
  const visible = state.selectedSegment === "full" && Boolean(state.fullDownloadUrl);
  menu.hidden = !visible;
  if (visible) {
    link.href = state.fullDownloadUrl;
  } else {
    menu.open = false;
    link.removeAttribute("href");
  }
}

function addOutputChoice(stream, url, key, label) {
  if (stream.querySelector(`[data-segment="${key}"]`)) return;
  const item = document.createElement("div");
  item.className = "segment-item";
  item.dataset.segment = String(key);
  item.tabIndex = 0;
  item.setAttribute("role", "button");
  item.setAttribute("aria-label", `查看 ${label}`);
  item.innerHTML = `<video src="${url}#t=0.01" muted loop playsinline preload="metadata"></video><span>${label}</span>`;
  item.addEventListener("mouseenter", () => $("video", item).play().catch(() => {}));
  item.addEventListener("mouseleave", () => $("video", item).pause());
  item.addEventListener("click", () => selectOutput(url, key, true));
  item.addEventListener("keydown", event => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectOutput(url, key, true);
    }
  });
  stream.append(item);
}

function addPlaylistChoice(stream, count) {
  let item = stream.querySelector('[data-segment="full"]');
  if (!item) {
    item = document.createElement("div");
    item.className = "segment-item playlist-item";
    item.dataset.segment = "full";
    item.tabIndex = 0;
    item.setAttribute("role", "button");
    item.setAttribute("aria-label", "连续播放全部 chunk");
    item.innerHTML = '<div class="playlist-thumb"><b>▶</b><small></small></div><span>FULL · 连播</span>';
    const activate = () => selectFullPlaylist(state.segmentUrls, true);
    item.addEventListener("click", activate);
    item.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); activate(); }
    });
    stream.append(item);
  }
  $("small", item).textContent = `${count} CHUNKS`;
}

function showSnapshot(snapshot) {
  const incomingStatus = snapshot.status || "idle";
  const terminalStatuses = ["complete", "failed", "cancelled"];
  const status = terminalStatuses.includes(state.jobStatus) && !terminalStatuses.includes(incomingStatus)
    ? state.jobStatus
    : incomingStatus;
  state.jobStatus = status;
  state.projectId = snapshot.projectId || state.projectId;
  state.revisionId = snapshot.revisionId || state.revisionId;
  state.fullDownloadUrl = snapshot.downloadVideo || null;
  const statusPill = $("#jobStatus");
  statusPill.className = `status-pill ${status}`;
  statusPill.textContent = status.toUpperCase();
  state.segmentCount = snapshot.completedChunks || 0;
  state.segmentUrls = [...(snapshot.segments || [])];
  const incomingDetail = snapshot.progressDetail;
  const incomingAt = Number(incomingDetail?.updatedAt || 0);
  if (incomingDetail && (!state.lastProgressAt || incomingAt >= state.lastProgressAt)) {
    state.lastProgressAt = incomingAt;
    state.lastProgressDetail = incomingDetail;
  }
  const complete = status === "complete";
  const detail = complete ? null : state.lastProgressDetail;
  $("#progressFraction").textContent = complete
    ? `${snapshot.chunks} / ${snapshot.chunks}`
    : detail
      ? `CH ${String(detail.chunkNumber).padStart(2, "0")} · ${detail.step}/${detail.totalSteps}`
      : `${state.segmentCount} / ${snapshot.chunks}`;
  const phaseElapsed = detail?.updatedAt && !["complete", "failed", "cancelled"].includes(status)
    ? ` · ${Math.max(0, Date.now() / 1000 - Number(detail.updatedAt)).toFixed(1)}s`
    : "";
  $("#progressMessage").textContent = complete
    ? "全部 chunk 已生成 · FULL 逐段连播 · 三点菜单按需合并下载"
    : `${snapshot.message || status}${phaseElapsed}`;
  $("#progressBar").style.width = `${complete ? 100 : Math.round((snapshot.progress || 0) * 100)}%`;
  const phaseOrder = ["geometry", "denoising", "decoding", "chunk_done"];
  const phaseIndex = complete ? phaseOrder.length : detail ? phaseOrder.indexOf(detail.phase) : -1;
  $$("#phaseStrip span").forEach((item, index) => {
    item.classList.toggle("active", index === phaseIndex);
    item.classList.toggle("done", phaseIndex > index || (detail?.phase === "chunk_done" && index <= phaseIndex));
  });
  $("#denoiseSteps").textContent = complete ? "3/3" : detail ? `${detail.step}/${detail.totalSteps}` : "0/3";
  if (detail && !snapshot.segments?.length) {
    $("#monitorEmpty strong").textContent = detail.message;
    $("#monitorEmpty span").textContent = detail.phase === "decoding"
      ? "解码完成后将立即播放当前 chunk"
      : "首个 chunk 内部进度正在实时更新";
  }
  $("#jobLog").textContent = (snapshot.log || []).join("\n") || "// waiting for worker";
  $("#jobLog").scrollTop = $("#jobLog").scrollHeight;

  const stream = $("#segmentStream");
  (snapshot.segments || []).forEach((url, index) => {
    addOutputChoice(stream, url, index, `CH ${String(index + 1).padStart(2, "0")}`);
  });
  if (snapshot.fullPlayback?.mode === "segments") addPlaylistChoice(stream, state.segmentUrls.length);
  if (state.autoFollowSegments && snapshot.segments?.length) {
    const latestIndex = snapshot.segments.length - 1;
    selectOutput(snapshot.segments[latestIndex], latestIndex);
    stream.scrollLeft = stream.scrollWidth;
  } else if (state.selectedSegment !== null) {
    const selected = stream.querySelector(`[data-segment="${state.selectedSegment}"]`);
    if (selected) selected.classList.add("active");
  }
  const done = ["complete", "failed", "cancelled"].includes(status);
  $("#generateButton").disabled = !done;
  $("#cancelButton").hidden = done;
  if (done && state.eventSource) { state.eventSource.close(); state.eventSource = null; }
  if (done && state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
  const ancestry = snapshot.parentRevisionId ? ` ← ${snapshot.parentRevisionId}` : "";
  $("#revisionLabel").textContent = snapshot.projectId && snapshot.revisionId
    ? `${snapshot.projectId} · ${snapshot.revisionId}${ancestry} · ${(snapshot.mode || "i2v").toUpperCase()}`
    : "旧版结果 · 首次重做时自动归档";
  if (complete && snapshot.chunks > state.continuationBaseChunks) {
    state.continuationBaseChunks = snapshot.chunks;
    renderPrompts();
    drawPath();
    drawLookPath();
  }
  updateGenerateAction();
  updateRetryButton();
}

function monitorJob(payload) {
  const changedJob = state.jobId !== payload.id;
  state.jobId = payload.id;
  if (changedJob) state.jobStatus = "idle";
  state.lastProgressAt = 0;
  state.lastProgressDetail = null;
  state.segmentUrls = [];
  state.fullPlaylist = null;
  state.fullPlaylistIndex = -1;
  state.fullDownloadUrl = null;
  showSnapshot(payload);
  startPolling(payload.id);
  if (state.eventSource) state.eventSource.close();
  state.eventSource = new EventSource(`api/jobs/${payload.id}/events`);
  state.eventSource.onmessage = event => showSnapshot(JSON.parse(event.data));
  state.eventSource.onerror = () => {
    if (state.eventSource) {
      state.eventSource.close(); state.eventSource = null;
      $("#formError").textContent = "实时状态流已切换为轮询更新。";
    }
  };
}

function startPolling(jobId) {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    try {
      const response = await fetch(`api/jobs/${jobId}`, {cache: "no-store"});
      if (response.ok) showSnapshot(await response.json());
    } catch (_) {
      // The SSE stream remains the primary channel; the next poll retries automatically.
    }
  }, 750);
}

function showHealth(health) {
  const topStatus = $("#weightStatus");
  const topParent = topStatus.parentElement;
  topParent.classList.toggle("ready", Boolean(health.ready));

  const gpu = health.gpu;
  const device = gpu?.devices?.[0];
  const gpuCheck = $("#gpuCheck");
  if (gpu?.available && device) {
    gpuCheck.className = `runtime-check ${gpu.capacityReady ? "ready" : "warning"}`;
    $("#gpuStatus").textContent = "GPU READY";
    const total = (device.totalMemoryMiB / 1024).toFixed(1);
    const free = (device.freeMemoryMiB / 1024).toFixed(1);
    $("#gpuDetail").textContent = `${device.name} · ${total}GB · 空闲 ${free}GB${gpu.capacityReady ? "" : " · 显存风险"}`;
  } else {
    gpuCheck.className = "runtime-check error";
    $("#gpuStatus").textContent = "GPU NOT FOUND";
    $("#gpuDetail").textContent = gpu?.error || "CUDA 设备不可见";
  }

  const weights = health.weights || {};
  const previewVae = health.previewVae || {};
  const geometry = health.geometry || {};
  const weightEntries = Object.entries(weights);
  const readyWeights = weightEntries.filter(([, ready]) => ready).length;
  const modelCheck = $("#modelCheck");
  const modelsReady = weightEntries.length === 3 && readyWeights === 3
    && (!previewVae.enabled || previewVae.ready)
    && (!geometry.enabled || geometry.ready);
  const modelTotal = 3 + (previewVae.enabled ? 1 : 0) + (geometry.enabled ? 1 : 0);
  const loadedModels = readyWeights + (previewVae.enabled && previewVae.ready ? 1 : 0)
    + (geometry.enabled && geometry.ready ? 1 : 0);
  modelCheck.className = `runtime-check ${modelsReady ? "ready" : "error"}`;
  $("#modelStatus").textContent = `${loadedModels} / ${modelTotal} READY`;
  $("#modelDetail").textContent = [
    `POST ${weights.postDistill ? "✓" : "×"}`,
    `BASE ${weights.base ? "✓" : "×"}`,
    `VIGEO 权重 ${weights.vigeo ? "✓" : "×"}`,
    ...(geometry.enabled ? [`VIGEO GPU ${geometry.ready ? "✓" : "…"}`] : []),
    ...(previewVae.enabled ? [`FAST VAE ${previewVae.ready ? "✓" : "×"}`] : []),
  ].join(" · ");

  const runtime = health.runtime || {phase: "loading", message: "正在启动常驻模型服务"};
  const runtimeCheck = $("#runtimeCheck");
  const busy = ["loading", "generating", "cancelling"].includes(runtime.phase);
  runtimeCheck.className = `runtime-check ${runtime.phase === "error" ? "error" : busy ? "busy" : runtime.phase === "queued" ? "warning" : "ready"}`;
  $("#runtimeStatus").textContent = ({
    idle: "IDLE",
    ready: "READY",
    queued: "QUEUED",
    loading: "LOADING",
    generating: "RUNNING",
    cancelling: "STOPPING",
    error: "ERROR",
  })[runtime.phase] || runtime.phase.toUpperCase();
  $("#runtimeDetail").textContent = runtime.phase === "ready" && !previewVae.enabled
    ? "Post-distill · ViGeo · 官方 Wan VAE 已加载并常驻 GPU"
    : runtime.message;

  topStatus.textContent = health.ready
    ? (gpu?.capacityReady ? "GPU 与模型就绪" : "GPU 与模型就绪 · 显存风险")
    : "运行环境未就绪";
}

async function refreshHealth() {
  try {
    const response = await fetch("api/health", {cache: "no-store"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    showHealth(await response.json());
  } catch (_) {
    $("#weightStatus").textContent = "后端离线";
  }
}

async function loadDefaultCase() {
  try {
    const response = await fetch("api/default-case", {cache: "no-store"});
    if (!response.ok) return;
    const defaultCase = await response.json();
    promptExamples = defaultCase.prompts;
    // Default restore is an explicit reset.  Discard the current editor values
    // before rendering, otherwise renderPrompts() intentionally preserves them.
    $("#promptList").replaceChildren();
    setChunks(defaultCase.chunks);
    const imageResponse = await fetch(defaultCase.referenceUrl, {cache: "no-store"});
    if (!imageResponse.ok) return;
    const blob = await imageResponse.blob();
    setImage(new File([blob], `${defaultCase.name || "default"}.jpg`, {type: blob.type || "image/jpeg"}));
  } catch (_) {
    // Manual upload remains available if the bundled example cannot be loaded.
  }
}

function resetSessionView() {
  if (state.eventSource) { state.eventSource.close(); state.eventSource = null; }
  if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
  state.jobId = null;
  state.projectId = null;
  state.revisionId = null;
  state.jobStatus = "idle";
  state.segmentCount = 0;
  state.selectedSegment = null;
  state.autoFollowSegments = true;
  state.continuationBaseChunks = 0;
  state.lastProgressAt = 0;
  state.lastProgressDetail = null;
  state.segmentUrls = [];
  state.fullDownloadUrl = null;
  $("#segmentStream").innerHTML = "";
  clearPlaybackVideos();
  updateFullDownloadMenu();
  $("#monitorEmpty").hidden = false;
  $("#monitorEmpty strong").textContent = "等待开始生成";
  $("#monitorEmpty span").textContent = "完成的 chunk 将在这里逐段出现";
  $("#jobStatus").className = "status-pill idle";
  $("#jobStatus").textContent = "READY";
  $("#progressMessage").textContent = "已恢复默认 case，可重新生成";
  $("#progressFraction").textContent = "0 / 6";
  $("#progressBar").style.width = "0%";
  $$("#phaseStrip span").forEach(item => item.classList.remove("active", "done"));
  $("#denoiseSteps").textContent = "0/3";
  $("#revisionLabel").textContent = "尚未创建本地版本";
  $("#jobLog").textContent = "// default case ready";
  $("#cancelButton").hidden = true;
  $("#formError").textContent = "";
  updateRetryButton();
  updateGenerateAction();
}

async function restoreDefaultCase() {
  resetSessionView();
  usePreset("push");
  useLookPreset("center");
  for (const [id, value] of Object.entries({motionScale: 24, verticalLift: 0, fov: 70, seed: 42})) {
    const input = $(`#${id}`);
    input.value = value;
    input.dispatchEvent(new Event("input"));
  }
  $("#followPath").checked = false;
  await loadDefaultCase();
  history.replaceState(null, "", `${location.pathname}?default=1`);
}

async function loadLatestSession() {
  try {
    const response = await fetch("api/session/latest", {cache: "no-store"});
    if (!response.ok) return;
    const snapshot = await response.json();
    const editor = snapshot.editor || {};
    state.continuationBaseChunks = Number(editor.chunks || snapshot.chunks || 0);
    state.pathPoints = (editor.pathPoints || presets.push).map(point => ({...point}));
    state.lookPoints = (editor.lookPoints?.length ? editor.lookPoints : lookPresets.center).map(point => ({...point}));
    promptExamples = editor.prompts || promptExamples;
    $("#promptList").replaceChildren();
    state.chunks = Number(editor.chunks || snapshot.chunks || 1);
    for (const [id, value] of Object.entries({
      motionScale: editor.motionScale,
      verticalLift: editor.verticalLift,
      fov: editor.fov,
      seed: editor.seed,
    })) {
      if (value !== undefined && $(`#${id}`)) {
        $(`#${id}`).value = value;
        $(`#${id}`).dispatchEvent(new Event("input"));
      }
    }
    $("#followPath").checked = editor.followPath !== false;
    updateTimeline();
    const imageResponse = await fetch(editor.referenceUrl, {cache: "no-store"});
    if (imageResponse.ok) {
      const blob = await imageResponse.blob();
      setImage(new File([blob], "project-reference.jpg", {type: blob.type || "image/jpeg"}), false);
    }
    monitorJob(snapshot);
  } catch (_) {
    // The bundled default case remains available when there is no local session.
  }
}

async function generate() {
  const error = $("#formError");
  error.textContent = "";
  const prompts = $$("#promptList textarea").map(input => input.value.trim());
  const extending = Boolean(state.jobId && state.jobStatus === "complete" && state.chunks > state.continuationBaseChunks);
  if (!extending && !state.image) { error.textContent = "请先上传一张参考图。"; return; }
  const emptyIndex = prompts.findIndex(prompt => !prompt);
  if (emptyIndex >= 0) { error.textContent = `请填写第 ${emptyIndex + 1} 个 chunk 的 prompt。`; $$("#promptList textarea")[emptyIndex].focus(); return; }
  if (state.pathPoints.length < 2) { error.textContent = "请选择预设或绘制一段运镜轨迹。"; return; }
  if (state.lookPoints.length < 2) { error.textContent = "请选择视角预设或从中心绘制视角轨迹。"; return; }

  $("#generateButton").disabled = true;
  $("#generateButton span").textContent = extending ? "正在建立续写版本…" : "正在创建任务…";
  $("#segmentStream").innerHTML = "";
  clearPlaybackVideos();
  $("#monitorEmpty").hidden = false;
  state.segmentCount = 0;
  state.selectedSegment = null;
  state.autoFollowSegments = true;
  state.fullDownloadUrl = null;
  updateFullDownloadMenu();
  const spec = {
    chunks: state.chunks,
    prompts,
    pathPoints: state.pathPoints,
    lookPoints: state.lookPoints,
    lookYawDegrees: 90,
    lookPitchDegrees: 45,
    motionScale: Number($("#motionScale").value),
    verticalLift: Number($("#verticalLift").value),
    fov: Number($("#fov").value),
    followPath: $("#followPath").checked,
    seed: Number($("#seed").value),
  };
  const form = new FormData();
  form.append("reference", state.image);
  form.append("spec", JSON.stringify(spec));
  try {
    const response = extending
      ? await fetch(`api/jobs/${state.jobId}/extend`, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            additionalChunks: state.chunks - state.continuationBaseChunks,
            prompts: prompts.slice(state.continuationBaseChunks),
            pathPoints: state.pathPoints,
            lookPoints: state.lookPoints,
            lookYawDegrees: spec.lookYawDegrees,
            lookPitchDegrees: spec.lookPitchDegrees,
            motionScale: spec.motionScale,
            verticalLift: spec.verticalLift,
            fov: spec.fov,
            followPath: spec.followPath,
            seed: spec.seed,
            contextChunks: 3,
          }),
        })
      : await fetch("api/jobs", {method: "POST", body: form});
    const payload = await readResponsePayload(response);
    if (!response.ok) throw new Error(payload.detail || "无法创建任务");
    monitorJob(payload);
  } catch (requestError) {
    error.textContent = requestError.message;
    $("#generateButton").disabled = false;
  } finally {
    updateGenerateAction();
  }
}

async function retrySelectedChunk() {
  const fromChunk = Number(state.selectedSegment);
  if (!state.jobId || state.jobStatus !== "complete" || !Number.isInteger(fromChunk)) return;
  const prompts = $$("#promptList textarea").map(input => input.value.trim());
  const emptyIndex = prompts.findIndex(prompt => !prompt);
  if (emptyIndex >= 0) {
    $("#formError").textContent = `请填写第 ${emptyIndex + 1} 个 chunk 的 prompt。`;
    return;
  }
  const retained = fromChunk > 0 ? `保留 CH 01–${String(fromChunk).padStart(2, "0")}，` : "不保留已有 chunk，";
  if (!window.confirm(`${retained}从 CH ${String(fromChunk + 1).padStart(2, "0")} 建立新版本？旧版本不会被覆盖。`)) return;
  const button = $("#retryChunkButton");
  button.disabled = true;
  button.textContent = "正在建立版本…";
  $("#formError").textContent = "";
  try {
    const response = await fetch(`api/jobs/${state.jobId}/retry`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({fromChunk, prompts, seed: Number($("#seed").value), contextChunks: 3}),
    });
    const payload = await readResponsePayload(response);
    if (!response.ok) throw new Error(payload.detail || "无法建立重生成版本");
    $("#segmentStream").innerHTML = "";
    clearPlaybackVideos();
    $("#monitorEmpty").hidden = false;
    state.segmentCount = 0;
    state.selectedSegment = null;
    state.autoFollowSegments = true;
    monitorJob(payload);
  } catch (error) {
    $("#formError").textContent = error.message;
    updateRetryButton();
  }
}

$("#referenceInput").addEventListener("change", event => setImage(event.target.files[0]));
$("#replaceImage").addEventListener("click", event => { event.preventDefault(); $("#referenceInput").click(); });
for (const eventName of ["dragenter", "dragover"]) $("#dropZone").addEventListener(eventName, event => { event.preventDefault(); $("#dropZone").classList.add("dragging"); });
for (const eventName of ["dragleave", "drop"]) $("#dropZone").addEventListener(eventName, event => { event.preventDefault(); $("#dropZone").classList.remove("dragging"); });
$("#dropZone").addEventListener("drop", event => setImage(event.dataTransfer.files[0]));
$$(".preset").forEach(button => button.addEventListener("click", () => usePreset(button.dataset.preset)));
$$(".look-preset").forEach(button => button.addEventListener("click", () => useLookPreset(button.dataset.lookPreset)));
$("#clearPath").addEventListener("click", () => { state.pathPoints = []; $$(".preset").forEach(button => button.classList.remove("active")); drawPath(); });
$("#clearLook").addEventListener("click", () => useLookPreset("center"));
$("#decreaseChunks").addEventListener("click", () => setChunks(state.chunks - 1));
$("#increaseChunks").addEventListener("click", () => setChunks(state.chunks + 1));
$("#fillPrompts").addEventListener("click", () => { const inputs = $$("#promptList textarea"); inputs.slice(1).forEach(input => { if (!input.value.trim()) { input.value = inputs[0].value; input.dispatchEvent(new Event("input")); } }); });
$("#generateButton").addEventListener("click", generate);
$("#defaultCaseButton").addEventListener("click", restoreDefaultCase);
for (const video of playlistVideoSlots()) video.addEventListener("ended", () => advanceFullPlaylist(video));
$("#retryChunkButton").addEventListener("click", retrySelectedChunk);
$("#cancelButton").addEventListener("click", async () => { if (state.jobId) await fetch(`api/jobs/${state.jobId}/cancel`, {method: "POST"}); });
for (const id of ["motionScale", "verticalLift", "fov"]) {
  const input = $(`#${id}`), output = $(`#${id}Value`);
  input.addEventListener("input", () => {
    output.textContent = id === "motionScale"
      ? `${Math.round(Number(input.value) / 24 * 100)}%`
      : `${input.value}${id === "fov" ? "°" : ""}`;
  });
}

window.addEventListener("resize", resizeCanvas);

usePreset("push");
useLookPreset("center");
updateTimeline();
const forceDefaultCase = new URLSearchParams(location.search).get("default") === "1";
loadDefaultCase().then(() => { if (!forceDefaultCase) return loadLatestSession(); });
refreshHealth();
setInterval(refreshHealth, 5000);
requestAnimationFrame(resizeCanvas);
