"use strict";
// ─── Types matching the sa3-server HTTP API ────────────────────────────────
let currentConfigFilename = "";
const CONFIG_EXT = ".json";
// ─── Dist-shift labels (param meanings per type) ───────────────────────────
const DIST_SHIFT_LABELS = {
    LogSNR: ["anchor_length", "anchor_logsnr", "rate", "logsnr_end"],
    Flux: ["min_length", "max_length", "alpha_min", "alpha_max"],
    Full: ["base_shift", "max_shift", "min_length", "max_length"],
    None: ["—", "—", "—", "—"],
};
// ─── State ──────────────────────────────────────────────────────────────────
const server = { host: "127.0.0.1", port: 8006 };
let configDefaults = null;
let loraList = [];
let activeLoras = [];
let currentSessionId = null;
let pastSongs = [];
// ─── DOM helpers ────────────────────────────────────────────────────────────
const $ = (s) => document.querySelector(s);
function val(s) {
    return $(s).value;
}
function num(s) {
    return parseFloat(val(s));
}
function int(s) {
    return parseInt(val(s), 10);
}
function isChecked(s) {
    return $(s).checked;
}
function setVal(s, v) {
    const el = $(s);
    if (el.type === "checkbox")
        el.checked = v;
    else
        el.value = String(v);
}
function apiBase() {
    return `http://${server.host}:${server.port}`;
}
// ─── API calls ──────────────────────────────────────────────────────────────
async function apiGet(path) {
    const r = await fetch(`${apiBase()}${path}`);
    if (!r.ok)
        throw new Error(`HTTP ${r.status}: ${r.statusText}`);
    return r.json();
}
async function apiPost(path, body) {
    const r = await fetch(`${apiBase()}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!r.ok)
        throw new Error(`HTTP ${r.status}: ${r.statusText}`);
    return r.json();
}
// ─── Slider-number sync ─────────────────────────────────────────────────────
function syncSliderToNum(sliderId, numId) {
    const slider = $(sliderId);
    const numInput = $(numId);
    if (!slider || !numInput)
        return;
    slider.addEventListener("input", () => { numInput.value = slider.value; });
    numInput.addEventListener("input", () => { slider.value = numInput.value; });
    numInput.addEventListener("change", () => { slider.value = numInput.value; });
}
// ─── Read form into request object ─────────────────────────────────────────
function readForm() {
    const distShift = val("#dist-shift");
    const dsParams = [
        num("#dsp1"), num("#dsp2"), num("#dsp3"), num("#dsp4"),
    ];
    const hasLatentTarget = val("#latent-target-std").trim().length > 0;
    const hasPeakNorm = val("#peak-normalize-db").trim().length > 0;
    const hasLimiter = val("#limiter-ceiling-db").trim().length > 0;
    const initPath = val("#init-path").trim();
    const negPrompt = val("#negative-prompt").trim();
    return {
        prompt: val("#prompt"),
        duration: num("#duration"),
        steps: int("#steps"),
        seed: int("#seed"),
        ...(initPath ? { init_path: initPath } : {}),
        init_noise_level: num("#init-noise-level"),
        inpaint_start: num("#inpaint-start"),
        inpaint_end: num("#inpaint-end"),
        ...(negPrompt ? { negative_prompt: negPrompt } : {}),
        cfg_scale: num("#cfg-scale"),
        cfg_rescale: num("#cfg-rescale"),
        apg_scale: num("#apg-scale"),
        cfg_norm_threshold: num("#cfg-norm-threshold"),
        cfg_interval_min: num("#cfg-interval-min"),
        cfg_interval_max: num("#cfg-interval-max"),
        dist_shift: distShift,
        dist_shift_params: dsParams,
        duration_padding_sec: num("#duration-padding"),
        keep_models: isChecked("#keep-models"),
        loras: activeLoras,
        encode_chunk_size: int("#encode-chunk-size"),
        encode_overlap: int("#encode-overlap"),
        decode_chunk_size: int("#decode-chunk-size"),
        decode_overlap: int("#decode-overlap"),
        latent_rescale: num("#latent-rescale"),
        latent_shift: num("#latent-shift"),
        ...(hasLatentTarget ? { latent_target_std: num("#latent-target-std") } : { latent_target_std: null }),
        latent_adapt_min: num("#latent-adapt-min"),
        latent_adapt_max: num("#latent-adapt-max"),
        ...(hasPeakNorm ? { peak_normalize_db: num("#peak-normalize-db") } : { peak_normalize_db: null }),
        ...(hasLimiter ? { limiter_ceiling_db: num("#limiter-ceiling-db") } : { limiter_ceiling_db: null }),
        limiter_knee: num("#limiter-knee"),
    };
}
// ─── Health ─────────────────────────────────────────────────────────────────
async function checkHealth() {
    const statusEl = $("#server-status");
    const modelInfo = $("#model-info");
    statusEl.textContent = "Connecting…";
    statusEl.className = "";
    statusEl.style.display = "";
    try {
        const h = await apiGet("/health");
        statusEl.textContent = "✓ Connected";
        statusEl.className = "ok";
        modelInfo.textContent = `${h.model} / ${h.encoding} ${h.loaded ? "(loaded)" : "(unloaded)"}`;
        modelInfo.style.display = "";
        loadLoras();
        fetchAudioFiles();
    }
    catch {
        statusEl.textContent = "✗ Server unreachable";
        statusEl.className = "err";
        modelInfo.style.display = "none";
    }
}
// ─── Config fetch (populate defaults from server) ───────────────────────────

async function fetchConfig() {
    try {
        configDefaults = await apiGet("/config");
        applyConfigDefaults();
    } catch {
        // server not connected yet
    }
}

function applyConfigDefaults() {
    if (!configDefaults) return;
    const c = configDefaults;
    setVal("#duration", c.duration);
    setVal("#duration-num", c.duration);
    setVal("#steps", c.steps);
    setVal("#steps-num", c.steps);
    setVal("#duration-padding", c.loop_pad_seconds);
    setVal("#duration-padding-num", c.loop_pad_seconds);
    setVal("#cfg-scale", c.cfg_scale);
    setVal("#cfg-rescale", c.cfg_rescale);
    setVal("#apg-scale", c.apg_scale);
    setVal("#cfg-norm-threshold", c.cfg_norm_threshold);
    setVal("#cfg-interval-min", c.cfg_interval_min);
    setVal("#cfg-interval-max", c.cfg_interval_max);
    setVal("#init-noise-level", c.init_noise_level);
    setVal("#loop-bpm", c.bpm);
    setVal("#encode-chunk-size", c.encode_chunk_size);
    setVal("#encode-overlap", c.encode_overlap);
    setVal("#decode-chunk-size", c.decode_chunk_size);
    setVal("#decode-overlap", c.decode_overlap);
    setVal("#inpaint-start", c.inpaint_start);
    setVal("#inpaint-end", c.inpaint_end);
    setVal("#seed", c.seed);
    if (c.loudness) {
        const l = c.loudness;
        if (l.latent_rescale != null) setVal("#latent-rescale", l.latent_rescale);
        if (l.latent_shift != null) setVal("#latent-shift", l.latent_shift);
        if (l.latent_target_std != null) setVal("#latent-target-std", l.latent_target_std);
        if (l.latent_adapt_min != null) setVal("#latent-adapt-min", l.latent_adapt_min);
        if (l.latent_adapt_max != null) setVal("#latent-adapt-max", l.latent_adapt_max);
        if (l.peak_normalize_db != null) setVal("#peak-normalize-db", l.peak_normalize_db);
        if (l.limiter_ceiling_db != null) setVal("#limiter-ceiling-db", l.limiter_ceiling_db);
        if (l.limiter_knee != null) setVal("#limiter-knee", l.limiter_knee);
    }
    onDistShiftChange();
}

// ─── Loras ──────────────────────────────────────────────────────────────────
async function loadLoras() {
    try {
        const r = await apiGet("/loras");
        loraList = r.loras;
        renderLoraDropdown();
    }
    catch {
        // server not connected yet
    }
}
function renderLoraDropdown() {
    const sel = $("#lora-select");
    sel.innerHTML = '<option value="">— select —</option>';
    for (const l of loraList) {
        const opt = document.createElement("option");
        opt.value = l.name;
        opt.textContent = l.name;
        sel.appendChild(opt);
    }
    renderActiveLoras();
}
function addLora() {
    const sel = $("#lora-select");
    const name = sel.value;
    if (!name)
        return;
    const strength = num("#lora-strength");
    if (activeLoras.some((l) => l.name === name))
        return;
    activeLoras.push({ name, strength });
    sel.value = "";
    renderActiveLoras();
}
function removeLora(name) {
    activeLoras = activeLoras.filter((l) => l.name !== name);
    renderActiveLoras();
}
function renderActiveLoras() {
    const container = $("#active-loras");
    container.innerHTML = "";
    for (const l of activeLoras) {
        const tag = document.createElement("span");
        tag.className = "lora-tag";
        tag.innerHTML = `${escapeHtml(l.name)} <span class="lora-str">(${l.strength.toFixed(2)})</span> <button class="small" data-name="${escapeHtml(l.name)}" title="Remove">&times;</button>`;
        tag.querySelector("button").addEventListener("click", () => removeLora(l.name));
        container.appendChild(tag);
    }
}
// ─── Audio-in file management ─────────────────────────────────────────────
async function fetchAudioFiles() {
    try {
        const r = await apiGet("/audio-in");
        const sel = $("#init-audio-select");
        const current = sel.value;
        sel.innerHTML = '<option value="">-- none (text-to-music) --</option>';
        for (const f of r.files) {
            const opt = document.createElement("option");
            opt.value = f;
            opt.textContent = f.replace(/\.[^.]+$/, "");
            sel.appendChild(opt);
        }
        // restore previous selection if still present
        if (current && r.files.includes(current)) {
            sel.value = current;
        }
        else {
            setVal("#init-path", "");
        }
    }
    catch {
        // server not connected yet
    }
}
function onInitAudioSelectChange() {
    const sel = $("#init-audio-select");
    const filename = sel.value;
    setVal("#init-path", filename ? `./audio-in/${filename}` : "");
}
async function uploadAudioFile() {
    const input = $("#init-audio-upload");
    const file = input.files?.[0];
    if (!file) {
        showError("Select a WAV file first");
        return;
    }
    const btn = $("#init-audio-upload-btn");
    btn.disabled = true;
    btn.textContent = "Uploading…";
    showError("");
    try {
        const formData = new FormData();
        formData.append("file", file);
        const r = await fetch(`${apiBase()}/upload`, { method: "POST", body: formData });
        if (!r.ok)
            throw new Error(`HTTP ${r.status}: ${r.statusText}`);
        const result = await r.json();
        if (!result.success)
            throw new Error(result.error || "Upload failed");
        // select the uploaded file in the dropdown
        await fetchAudioFiles();
        const sel = $("#init-audio-select");
        if (result.filename) {
            sel.value = result.filename;
            onInitAudioSelectChange();
        }
        input.value = "";
    }
    catch (e) {
        showError(e instanceof Error ? e.message : "Upload failed");
    }
    finally {
        btn.disabled = false;
        btn.textContent = "Upload";
    }
}
// ─── Dist-shift parameter defaults ──────────────────────────────────────────
function onDistShiftChange() {
    const type = val("#dist-shift");
    const labels = DIST_SHIFT_LABELS[type] || ["p1", "p2", "p3", "p4"];
    const defaults = (configDefaults && configDefaults.dist_shift_defaults && configDefaults.dist_shift_defaults[type]) || [0, 0, 0, 0];
    for (let i = 0; i < 4; i++) {
        const input = $(`#dsp${i + 1}`);
        const label = document.querySelector(`label[for="dsp${i + 1}"]`);
        if (label)
            label.textContent = labels[i];
        input.value = String(defaults[i]);
        input.disabled = type === "None";
    }
}
// ─── Generate ───────────────────────────────────────────────────────────────
let pollTimer = null;
let currentSongDataUrl = null;
let currentSongSeed = -1;
function pushCurrentToPastSongs() {
    if (currentSongDataUrl) {
        addSongEntry(currentSongDataUrl, currentSongSeed);
        currentSongDataUrl = null;
        deleteCurrentSongFromDB();
        const rs = $("#result-section");
        if (rs)
            rs.style.display = "none";
    }
}
async function generate() {
    if (currentSessionId) {
        cancelJob();
        return;
    }
    clearPolling();
    pushCurrentToPastSongs();
    const body = readForm();
    try {
        const r = await apiPost("/generate", body);
        startPolling(r.session_id, "gen");
    }
    catch (e) {
        showError(e instanceof Error ? e.message : "Request failed");
        enableButtons();
    }
}
async function generateLoop() {
    if (currentSessionId) {
        cancelJob();
        return;
    }
    clearPolling();
    pushCurrentToPastSongs();
    const body = {
        ...readForm(),
        bpm: num("#loop-bpm"),
        bars: int("#loop-bars"),
    };
    try {
        const r = await apiPost("/generate/loop", body);
        startPolling(r.session_id, "loop");
    }
    catch (e) {
        showError(e instanceof Error ? e.message : "Request failed");
        enableButtons();
    }
}
function startPolling(sessionId, which) {
    const progressBar = $("#progress-bar");
    const progressLabel = $("#progress-label");
    const resultAudio = $("#result-audio");
    const resultSection = $("#result-section");
    const seedInfo = $("#seed-info");
    currentSessionId = sessionId;
    progressBar.style.width = "0%";
    progressLabel.textContent = "queued";
    resultSection.style.display = "none";
    resultAudio.src = "";
    seedInfo.textContent = "";
    showError("");
    showCancelButton(which);
    pollTimer = setInterval(async () => {
        try {
            const r = await apiGet(`/poll_status/${sessionId}`);
            progressBar.style.width = `${r.progress}%`;
            if (r.status === "queued") {
                progressLabel.textContent = "queued…";
            }
            else if (r.status === "generating" || r.status === "encoding"
                || r.status === "decoding" || r.status === "finalizing") {
                progressLabel.textContent = `${r.status} step ${r.step}/${r.total_steps} (${r.progress}%)`;
            }
            else if (r.status === "completed") {
                progressLabel.textContent = `completed (${r.progress}%)`;
                if (r.audio_data) {
                    const dataUrl = `data:audio/wav;base64,${r.audio_data}`;
                    currentSongDataUrl = dataUrl;
                    currentSongSeed = r.meta?.seed ?? -1;
                    saveCurrentSongToDB(dataUrl, currentSongSeed);
                    resultAudio.src = dataUrl;
                    resultSection.style.display = "block";
                }
                const metaParts = [];
                if (r.meta?.seed != null)
                    metaParts.push(`Seed: ${r.meta.seed}`);
                if (r.meta?.loudness) {
                    const lm = r.meta.loudness;
                    if (lm.final_peak != null)
                        metaParts.push(`Peak: ${Number(lm.final_peak).toFixed(3)}`);
                    if (lm.decoded_peak != null)
                        metaParts.push(`Decoded: ${Number(lm.decoded_peak).toFixed(3)}`);
                }
                seedInfo.textContent = metaParts.join(" · ");
                clearPolling();
                enableButtons();
            }
            else if (r.status === "failed") {
                progressLabel.textContent = `failed: ${r.error || "unknown error"}`;
                clearPolling();
                enableButtons();
            }
            else if (r.status === "cancelled") {
                progressLabel.textContent = "cancelled by user";
                clearPolling();
                enableButtons();
            }
        }
        catch {
            progressLabel.textContent = "poll error";
            clearPolling();
            enableButtons();
        }
    }, 500);
}
function clearPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}
function enableButtons() {
    currentSessionId = null;
    const genBtn = $("#gen-btn");
    genBtn.disabled = false;
    genBtn.textContent = "🎵 Generate";
    genBtn.className = "primary";
    const loopBtn = $("#loop-btn");
    loopBtn.disabled = false;
    loopBtn.textContent = "🔄 Generate Loop";
    loopBtn.className = "loop";
}
function showCancelButton(which) {
    if (which === "gen") {
        const genBtn = $("#gen-btn");
        genBtn.disabled = false;
        genBtn.textContent = "✕ Cancel";
        genBtn.className = "danger";
    }
    else {
        const loopBtn = $("#loop-btn");
        loopBtn.disabled = false;
        loopBtn.textContent = "✕ Cancel";
        loopBtn.className = "danger";
    }
}
async function cancelJob() {
    if (!currentSessionId)
        return;
    try {
        await apiPost(`/cancel/${currentSessionId}`, {});
    }
    catch {
        // best effort — pipeline will pick up the flag on next check
    }
}
function showError(msg) {
    const el = $("#error-msg");
    if (el)
        el.textContent = msg;
}
// ─── IndexedDB persistence ───────────────────────────────────────────────────
const DB_NAME = "sa3web";
const DB_STORE = "songs";
const DB_VERSION = 1;
const CURRENT_SONG_ID = "_current";
function openDB() {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(DB_NAME, DB_VERSION);
        req.onupgradeneeded = () => {
            const db = req.result;
            if (!db.objectStoreNames.contains(DB_STORE)) {
                db.createObjectStore(DB_STORE, { keyPath: "id" });
            }
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}
async function loadSongsFromDB() {
    try {
        const db = await openDB();
        const tx = db.transaction(DB_STORE, "readonly");
        const store = tx.objectStore(DB_STORE);
        const all = store.getAll();
        all.onsuccess = () => {
            pastSongs = all.result
                .filter((e) => e.id !== CURRENT_SONG_ID)
                .sort((a, b) => b.timestamp - a.timestamp);
            renderPastSongs();
        };
    }
    catch { /* DB unavailable — start empty */ }
}
async function saveSongToDB(entry) {
    try {
        const db = await openDB();
        const tx = db.transaction(DB_STORE, "readwrite");
        tx.objectStore(DB_STORE).put(entry);
    }
    catch { /* silently fail */ }
}
async function deleteSongFromDB(id) {
    try {
        const db = await openDB();
        const tx = db.transaction(DB_STORE, "readwrite");
        tx.objectStore(DB_STORE).delete(id);
    }
    catch { /* silently fail */ }
}
async function saveCurrentSongToDB(dataUrl, seed) {
    try {
        const db = await openDB();
        const tx = db.transaction(DB_STORE, "readwrite");
        tx.objectStore(DB_STORE).put({ id: CURRENT_SONG_ID, dataUrl, seed, ts: Date.now() });
    }
    catch { /* silently fail */ }
}
async function deleteCurrentSongFromDB() {
    try {
        const db = await openDB();
        const tx = db.transaction(DB_STORE, "readwrite");
        tx.objectStore(DB_STORE).delete(CURRENT_SONG_ID);
    }
    catch { /* silently fail */ }
}
async function loadCurrentSongFromDB() {
    try {
        const db = await openDB();
        const tx = db.transaction(DB_STORE, "readonly");
        const store = tx.objectStore(DB_STORE);
        const req = store.get(CURRENT_SONG_ID);
        req.onsuccess = () => {
            const entry = req.result;
            if (entry?.dataUrl) {
                currentSongDataUrl = entry.dataUrl;
                currentSongSeed = entry.seed;
                const audio = $("#result-audio");
                const section = $("#result-section");
                if (audio && section) {
                    audio.src = entry.dataUrl;
                    section.style.display = "block";
                }
            }
        };
    }
    catch { /* silently fail */ }
}
// ─── Past Songs ──────────────────────────────────────────────────────────────
function addSongEntry(audioData, seed) {
    const now = new Date();
    const ts = now.getFullYear().toString() +
        String(now.getMonth() + 1).padStart(2, "0") +
        String(now.getDate()).padStart(2, "0") + "-" +
        String(now.getHours()).padStart(2, "0") +
        String(now.getMinutes()).padStart(2, "0") +
        String(now.getSeconds()).padStart(2, "0");
    const name = "Song-" + ts;
    const entry = {
        id: "song_" + now.getTime(),
        name,
        timestamp: now.getTime(),
        audioData,
        seed,
        config: readFormAsConfig(),
    };
    pastSongs.unshift(entry);
    renderPastSongs();
    saveSongToDB(entry);
}
function deleteSongEntry(id) {
    pastSongs = pastSongs.filter((s) => s.id !== id);
    renderPastSongs();
    deleteSongFromDB(id);
}
function deleteCurrentSong() {
    currentSongDataUrl = null;
    deleteCurrentSongFromDB();
    const rs = $("#result-section");
    if (rs)
        rs.style.display = "none";
    const audio = $("#result-audio");
    if (audio)
        audio.src = "";
}
function clearAllSongs() {
    if (!pastSongs.length)
        return;
    if (!confirm("Delete all past songs? This cannot be undone."))
        return;
    const ids = pastSongs.map((s) => s.id);
    pastSongs = [];
    renderPastSongs();
    for (const id of ids)
        deleteSongFromDB(id);
}
function renderPastSongs() {
    const container = $("#past-songs");
    const count = $("#past-count");
    if (!container)
        return;
    container.innerHTML = "";
    if (count)
        count.textContent = String(pastSongs.length);
    for (const song of pastSongs) {
        const div = document.createElement("div");
        div.className = "song-entry";
        const nameSpan = document.createElement("span");
        nameSpan.className = "song-name";
        nameSpan.textContent = song.name;
        const audio = document.createElement("audio");
        audio.src = song.audioData;
        audio.controls = true;
        const paramsSpan = document.createElement("span");
        paramsSpan.className = "song-params";
        const p = song.config;
        paramsSpan.textContent =
            `prompt: "${p.prompt.length > 40 ? p.prompt.slice(0, 40) + "…" : p.prompt}" · ` +
                `${p.dist_shift} · steps:${p.steps} · cfg:${p.cfg_scale} · seed:${song.seed}`;
        const actions = document.createElement("div");
        actions.className = "song-actions";
        const loadBtn = document.createElement("button");
        loadBtn.className = "small";
        loadBtn.textContent = "📋";
        loadBtn.title = "Load parameters";
        loadBtn.addEventListener("click", () => {
            song.config.version = 1;
            applyConfig(song.config);
        });
        const dlBtn = document.createElement("button");
        dlBtn.className = "small";
        dlBtn.textContent = "⬇";
        dlBtn.title = "Download WAV";
        dlBtn.addEventListener("click", () => {
            const a = document.createElement("a");
            a.href = song.audioData;
            a.download = song.name + ".wav";
            a.click();
        });
        const delBtn = document.createElement("button");
        delBtn.className = "small danger";
        delBtn.textContent = "✕";
        delBtn.title = "Delete";
        delBtn.addEventListener("click", () => deleteSongEntry(song.id));
        actions.appendChild(loadBtn);
        actions.appendChild(dlBtn);
        actions.appendChild(delBtn);
        div.appendChild(nameSpan);
        div.appendChild(audio);
        div.appendChild(paramsSpan);
        div.appendChild(actions);
        container.appendChild(div);
    }
}
function readFormAsConfig() {
    const dsParams = [
        num("#dsp1"), num("#dsp2"), num("#dsp3"), num("#dsp4"),
    ];
    const ltRaw = val("#latent-target-std").trim();
    const pnRaw = val("#peak-normalize-db").trim();
    const lcRaw = val("#limiter-ceiling-db").trim();
    return {
        version: 1,
        prompt: val("#prompt"),
        negative_prompt: val("#negative-prompt").trim(),
        duration: num("#duration"),
        steps: int("#steps"),
        seed: int("#seed"),
        duration_padding_sec: num("#duration-padding"),
        cfg_scale: num("#cfg-scale"),
        cfg_rescale: num("#cfg-rescale"),
        apg_scale: num("#apg-scale"),
        cfg_norm_threshold: num("#cfg-norm-threshold"),
        cfg_interval_min: num("#cfg-interval-min"),
        cfg_interval_max: num("#cfg-interval-max"),
        dist_shift: val("#dist-shift"),
        dist_shift_params: dsParams,
        keep_models: isChecked("#keep-models"),
        encode_chunk_size: int("#encode-chunk-size"),
        encode_overlap: int("#encode-overlap"),
        decode_chunk_size: int("#decode-chunk-size"),
        decode_overlap: int("#decode-overlap"),
        latent_rescale: num("#latent-rescale"),
        latent_shift: num("#latent-shift"),
        latent_target_std: ltRaw.length > 0 ? num("#latent-target-std") : null,
        latent_adapt_min: num("#latent-adapt-min"),
        latent_adapt_max: num("#latent-adapt-max"),
        peak_normalize_db: pnRaw.length > 0 ? num("#peak-normalize-db") : null,
        limiter_ceiling_db: lcRaw.length > 0 ? num("#limiter-ceiling-db") : null,
        limiter_knee: num("#limiter-knee"),
        init_path: val("#init-path").trim(),
        init_noise_level: num("#init-noise-level"),
        inpaint_start: num("#inpaint-start"),
        inpaint_end: num("#inpaint-end"),
        loop_bpm: num("#loop-bpm"),
        loop_bars: int("#loop-bars"),
        loras: activeLoras.map((l) => ({ ...l })),
    };
}
function applyConfig(cfg) {
    setVal("#prompt", cfg.prompt);
    setVal("#negative-prompt", cfg.negative_prompt || "");
    setVal("#duration", cfg.duration);
    setVal("#duration-num", cfg.duration);
    setVal("#steps", cfg.steps);
    setVal("#steps-num", cfg.steps);
    setVal("#seed", cfg.seed);
    setVal("#duration-padding", cfg.duration_padding_sec);
    setVal("#duration-padding-num", cfg.duration_padding_sec);
    setVal("#cfg-scale", cfg.cfg_scale);
    setVal("#cfg-rescale", cfg.cfg_rescale);
    setVal("#apg-scale", cfg.apg_scale);
    setVal("#cfg-norm-threshold", cfg.cfg_norm_threshold);
    setVal("#cfg-interval-min", cfg.cfg_interval_min);
    setVal("#cfg-interval-max", cfg.cfg_interval_max);
    setVal("#dist-shift", cfg.dist_shift);
    if (cfg.dist_shift_params) {
        for (let i = 0; i < 4; i++) {
            const inp = $(`#dsp${i + 1}`);
            inp.value = String(cfg.dist_shift_params[i]);
        }
    }
    onDistShiftChange();
    setVal("#keep-models", cfg.keep_models);
    setVal("#encode-chunk-size", cfg.encode_chunk_size);
    setVal("#encode-overlap", cfg.encode_overlap);
    setVal("#decode-chunk-size", cfg.decode_chunk_size);
    setVal("#decode-overlap", cfg.decode_overlap);
    setVal("#latent-rescale", cfg.latent_rescale);
    setVal("#latent-shift", cfg.latent_shift);
    setVal("#latent-target-std", cfg.latent_target_std != null ? String(cfg.latent_target_std) : "");
    setVal("#latent-adapt-min", cfg.latent_adapt_min);
    setVal("#latent-adapt-max", cfg.latent_adapt_max);
    setVal("#peak-normalize-db", cfg.peak_normalize_db != null ? String(cfg.peak_normalize_db) : "");
    setVal("#limiter-ceiling-db", cfg.limiter_ceiling_db != null ? String(cfg.limiter_ceiling_db) : "");
    setVal("#limiter-knee", cfg.limiter_knee);
    setVal("#init-path", cfg.init_path || "");
    // restore init audio dropdown from saved path
    {
        const sel = $("#init-audio-select");
        const initPath = cfg.init_path || "";
        const match = initPath.match(/audio-in\/(.+)$/);
        if (match && sel.querySelector(`option[value="${CSS.escape(match[1])}"]`)) {
            sel.value = match[1];
        }
        else {
            sel.value = "";
        }
    }
    setVal("#init-noise-level", cfg.init_noise_level);
    setVal("#inpaint-start", cfg.inpaint_start);
    setVal("#inpaint-end", cfg.inpaint_end);
    setVal("#loop-bpm", cfg.loop_bpm);
    setVal("#loop-bars", cfg.loop_bars);
    activeLoras = cfg.loras.map((l) => ({ ...l }));
    renderActiveLoras();
}
function saveConfig() {
    const suggested = currentConfigFilename || "sa3-config.json";
    const name = prompt("Save config as:", suggested);
    if (!name)
        return;
    const finalName = name.endsWith(CONFIG_EXT) ? name : name + CONFIG_EXT;
    currentConfigFilename = finalName;
    const cfg = readFormAsConfig();
    const blob = new Blob([JSON.stringify(cfg, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = finalName;
    a.click();
    URL.revokeObjectURL(url);
    const fnEl = $("#config-filename");
    if (fnEl)
        fnEl.textContent = currentConfigFilename;
}
function loadConfig() {
    const input = $("#load-config-input");
    input.value = "";
    input.click();
}
function onConfigFileSelected(e) {
    const file = e.target.files?.[0];
    if (!file)
        return;
    const reader = new FileReader();
    reader.onload = () => {
        try {
            const cfg = JSON.parse(reader.result);
            if (cfg.version !== 1) {
                showError("Unsupported config version");
                return;
            }
            applyConfig(cfg);
            currentConfigFilename = file.name;
            const fnEl = $("#config-filename");
            if (fnEl)
                fnEl.textContent = currentConfigFilename;
            showError("");
        }
        catch {
            showError("Invalid config file");
        }
    };
    reader.readAsText(file);
}
// ─── Helpers ────────────────────────────────────────────────────────────────
function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
}
// ─── Toggle collapsible sections ────────────────────────────────────────────
function setupCollapsibles() {
    for (const btn of document.querySelectorAll(".collapse-toggle")) {
        btn.addEventListener("click", () => {
            const target = document.getElementById(btn.getAttribute("data-target") || "");
            if (target) {
                target.classList.toggle("collapsed");
                btn.classList.toggle("collapsed");
            }
        });
    }
}
// ─── Theme toggle ────────────────────────────────────────────────────────────
function toggleTheme() {
    const html = document.documentElement;
    const isLight = html.getAttribute("data-theme") === "light";
    if (isLight) {
        html.removeAttribute("data-theme");
        localStorage.setItem("theme", "dark");
    }
    else {
        html.setAttribute("data-theme", "light");
        localStorage.setItem("theme", "light");
    }
    updateThemeBtn();
}
function updateThemeBtn() {
    const btn = $("#theme-btn");
    if (!btn)
        return;
    const isLight = document.documentElement.getAttribute("data-theme") === "light";
    btn.textContent = isLight ? "🌙" : "☀️";
}
function initTheme() {
    const saved = localStorage.getItem("theme");
    if (saved === "light") {
        document.documentElement.setAttribute("data-theme", "light");
    }
    updateThemeBtn();
}
// ─── Init ───────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    setupCollapsibles();
    initTheme();
    // Sync range sliders with their number companions
    syncSliderToNum("#duration", "#duration-num");
    syncSliderToNum("#steps", "#steps-num");
    syncSliderToNum("#duration-padding", "#duration-padding-num");
    // Dist-shift defaults
    onDistShiftChange();
    // Load past songs and current song from IndexedDB
    loadSongsFromDB();
    loadCurrentSongFromDB();
    // Event listeners
    $("#gen-btn").addEventListener("click", generate);
    $("#loop-btn").addEventListener("click", generateLoop);
    $("#lora-add-btn").addEventListener("click", addLora);
    $("#dist-shift").addEventListener("change", onDistShiftChange);
    $("#delete-current-btn").addEventListener("click", deleteCurrentSong);
    $("#clear-all-btn").addEventListener("click", clearAllSongs);
    $("#theme-btn").addEventListener("click", toggleTheme);
    $("#save-config-btn").addEventListener("click", saveConfig);
    $("#load-config-btn").addEventListener("click", loadConfig);
    $("#load-config-input").addEventListener("change", onConfigFileSelected);
    // Init audio controls
    $("#init-audio-select").addEventListener("change", onInitAudioSelectChange);
    $("#init-audio-refresh-btn").addEventListener("click", fetchAudioFiles);
    $("#init-audio-upload-btn").addEventListener("click", uploadAudioFile);
    // Ctrl+Enter (or Cmd+Enter) triggers generate
    document.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            generate();
        }
    });
    // Auto-connect to server on page load
    checkHealth();
    fetchConfig();
});
