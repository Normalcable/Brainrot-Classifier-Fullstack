/* 
  Academic Research Dashboard - State Management & Visualization 
  Refactored for strict execution flow, reactive UI, and Chart.js integration.
*/

const API_BASE = "http://localhost:8000";

// --- Global App State ---
const THRESHOLD_MAP = { default: 0.42, no_yt: 0.50 };

const AppState = {
    inputs: {
        file: null,
        modelVersion: 'no_yt',
        mode: 'ensemble'
    },
    isExecuting: false,
    results: null,
    compareResults: null,
    threshold: 0.50,
    apiReady: false,
    telemetryHistory: [],
    telemetrySummary: null,
};

// --- DOM References ---
const UI = {
    statusDot: document.getElementById('apiStatusDot'),
    statusText: document.getElementById('apiStatusText'),
    
    uploadZone: document.getElementById('uploadZone'),
    fileInput: document.getElementById('fileInput'),
    uploadLabel: document.getElementById('uploadLabel'),
    uploadSize: document.getElementById('uploadSize'),
    
    versionBtns: document.querySelectorAll('#modelVersionControl .seg-btn'),
    modeBtns: document.querySelectorAll('#analysisModeControl .seg-btn'),
    runBtn: document.getElementById('runBtn'),
    errorBox: document.getElementById('errorBox'),
    
    welcomeScreen: document.getElementById('welcomeScreen'),
    telemetryScreen: document.getElementById('telemetryScreen'),
    dashboardGrid: document.getElementById('dashboardGrid'),
    telCpuVal: document.getElementById('telCpuVal'),
    telRamVal: document.getElementById('telRamVal'),
    telGpuVal: document.getElementById('telGpuVal'),
    telStage: document.getElementById('telStage'),
    sysLog: document.getElementById('sysLog'),
    
    // Verdict / Dashboard UI
    singleVerdictCard: document.getElementById('singleVerdictCard'),
    verdictCard: document.querySelector('.verdict-card'),
    verdictLabel: document.getElementById('finalVerdictLabel'),
    confVal: document.getElementById('confidenceVal'),
    probBR: document.getElementById('probBR'),
    probClean: document.getElementById('probClean'),
    thresholdVal: document.getElementById('thresholdVal'),
    thresholdModelLabel: document.getElementById('thresholdModelLabel'),
    sidebarThresholdVal: document.getElementById('sidebarThresholdVal'),
    activeThresholdHint: document.getElementById('activeThresholdHint'),
    comparisonVerdictGrid: document.getElementById('comparisonVerdictGrid'),
    comparisonMetricsGrid: document.getElementById('comparisonMetricsGrid'),
    singleMetricsCard: document.getElementById('singleMetricsCard'),

    
    metricsList: document.getElementById('metricsList'),
    totalTimeLabel: document.getElementById('totalTimeLabel'),
    transcriptBox: document.getElementById('transcriptBox'),
    metaFile: document.getElementById('metaFile'),
    metaFolds: document.getElementById('metaFolds')
};

// --- Chart Instances ---
let modalityChartInstance = null;
let timelineChartInstance = null;
let hwTrajectoryChartInstance = null;
Chart.defaults.font.family = "'Fira Code', monospace";

function isDarkMode() {
    return document.documentElement.getAttribute('data-theme') === 'dark';
}

function getThemeColors() {
    const dark = isDarkMode();
    return {
        textMuted:   dark ? '#9B9894' : '#7A7571',
        textMain:    dark ? '#E8E6E3' : '#2D2B2A',
        gridLine:    dark ? 'rgba(255,255,255,0.1)' : '#E5E2DC',
        accent:      dark ? '#8A9BA3' : '#5C6B73',
        brainrot:    dark ? '#E08070' : '#C86454',
        clean:       dark ? '#7DA08D' : '#608070',
        warning:     dark ? '#D4A84B' : '#B8860B',
        tooltipBg:   dark ? 'rgba(30,30,30,0.95)' : 'rgba(249,248,246,0.95)',
        tooltipBorder: dark ? '#3A3836' : '#E5E2DC',
        sparkGrid:   dark ? 'rgba(255,255,255,0.08)' : 'rgba(229,226,220,0.6)',
        radarFill:   dark ? 'rgba(138,155,163,0.2)' : 'rgba(92,107,115,0.15)',
        brGradTop:   dark ? 'rgba(224,128,112,0.35)' : 'rgba(200,100,84,0.3)',
        brGradBot:   dark ? 'rgba(125,160,141,0.08)' : 'rgba(96,128,112,0.05)',
    };
}

function applyChartDefaults() {
    const tc = getThemeColors();
    Chart.defaults.color = tc.textMuted;
}
applyChartDefaults();

// --- Sparkline Class ---
class Sparkline {
    constructor(canvasId, wrapperId, color) {
        this.canvas = document.getElementById(canvasId);
        this.wrapper = document.getElementById(wrapperId);
        this.ctx = this.canvas.getContext('2d');
        this.color = color;
        this.data = new Array(50).fill(0);
        window.addEventListener('resize', () => this.resize());
    }
    resize() {
        const w = this.wrapper.clientWidth;
        const h = this.wrapper.clientHeight;
        if (w === 0 || h === 0) return; // still hidden, skip
        this.canvas.width = w;
        this.canvas.height = h;
        this.draw();
    }
    reset() {
        this.data = new Array(50).fill(0);
    }
    push(val) {
        this.data.shift();
        this.data.push(val);
        this.draw();
    }
    draw() {
        const { width, height } = this.canvas;
        if (width === 0 || height === 0) return;
        const ctx = this.ctx;
        ctx.clearRect(0, 0, width, height);

        // Subtle horizontal grid lines at 25%, 50%, 75%
        ctx.strokeStyle = getThemeColors().sparkGrid;
        ctx.lineWidth = 0.5;
        for (const pct of [0.25, 0.5, 0.75]) {
            const gy = height * (1 - pct);
            ctx.beginPath();
            ctx.moveTo(0, gy);
            ctx.lineTo(width, gy);
            ctx.stroke();
        }

        // Data path
        const step = width / (this.data.length - 1);
        ctx.beginPath();
        for (let i = 0; i < this.data.length; i++) {
            const x = i * step;
            const y = height - (this.data[i] / 100 * height);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }

        // Stroke the line
        ctx.strokeStyle = this.color;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Gradient fill underneath
        ctx.lineTo(width, height);
        ctx.lineTo(0, height);
        ctx.closePath();
        const gradient = ctx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, this.color + '55');
        gradient.addColorStop(0.7, this.color + '15');
        gradient.addColorStop(1, 'transparent');
        ctx.fillStyle = gradient;
        ctx.fill();
    }
}

let cpuSpark, ramSpark, gpuSpark;

// --- Initialization ---
window.addEventListener('DOMContentLoaded', () => {
    initTheme();
    cpuSpark = new Sparkline('canvasCpu', 'chartWrapCpu', '#5C6B73');
    ramSpark = new Sparkline('canvasRam', 'chartWrapRam', '#608070');
    gpuSpark = new Sparkline('canvasGpu', 'chartWrapGpu', '#C86454');
    checkHealth();
    bindEvents();
});

// --- Theme Toggle ---
function initTheme() {
    const saved = localStorage.getItem('theme');
    if (saved === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
    }
    syncThemeIcons();

    document.getElementById('themeToggle').addEventListener('click', () => {
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        if (isDark) {
            document.documentElement.removeAttribute('data-theme');
            localStorage.setItem('theme', 'light');
        } else {
            document.documentElement.setAttribute('data-theme', 'dark');
            localStorage.setItem('theme', 'dark');
        }
        syncThemeIcons();
    });
}

function syncThemeIcons() {
    const dark = isDarkMode();
    document.getElementById('iconMoon').style.display = dark ? 'none' : 'block';
    document.getElementById('iconSun').style.display = dark ? 'block' : 'none';
    applyChartDefaults();
    // Re-render charts if results are currently displayed
    if (AppState.results && !UI.dashboardGrid.classList.contains('hidden')) {
        renderCharts();
        renderHardwareSummary();
    }
}

// --- API Health Check ---
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
        const data = await res.json();
        if(data.models_ready) {
            AppState.apiReady = true;
            UI.statusDot.className = 'dot online';
            UI.statusText.textContent = `Online (Folds: ${data.folds_loaded.length}) - ${data.model_version_name}`;
        } else {
            setOffline('Loading models...');
        }
    } catch (e) {
        setOffline('Connection Failed');
    }
    validateExecutionState();
}

function setOffline(msg) {
    AppState.apiReady = false;
    UI.statusDot.className = 'dot offline';
    UI.statusText.textContent = msg;
}

// --- Event Binders ---
function bindEvents() {
    // 1. Upload Behavior
    UI.uploadZone.addEventListener('click', () => UI.fileInput.click());
    UI.uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); UI.uploadZone.classList.add('dragover'); });
    UI.uploadZone.addEventListener('dragleave', () => UI.uploadZone.classList.remove('dragover'));
    UI.uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        UI.uploadZone.classList.remove('dragover');
        if(e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
    });
    UI.fileInput.addEventListener('change', (e) => {
        if(e.target.files[0]) handleFile(e.target.files[0]);
    });

    // 2. Segmented Controls — Model Version (with threshold mapping)
    UI.versionBtns.forEach(btn => btn.addEventListener('click', (e) => {
        UI.versionBtns.forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        AppState.inputs.modelVersion = e.target.dataset.val;
        updateThresholdDisplay();
    }));

    UI.modeBtns.forEach(btn => btn.addEventListener('click', (e) => {
        UI.modeBtns.forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        AppState.inputs.mode = e.target.dataset.val;
    }));

    // 4. Execution
    UI.runBtn.addEventListener('click', executePipeline);

    // Initialize threshold display
    updateThresholdDisplay();
}

function updateThresholdDisplay() {
    const mv = AppState.inputs.modelVersion;
    if (mv === 'compare') {
        AppState.threshold = 0.42;
        UI.sidebarThresholdVal.textContent = '0.42 / 0.50';
    } else {
        AppState.threshold = THRESHOLD_MAP[mv];
        UI.sidebarThresholdVal.textContent = AppState.threshold.toFixed(2);
    }
    if (UI.thresholdVal) {
        UI.thresholdVal.textContent = AppState.threshold.toFixed(2);
    }
    if (UI.thresholdModelLabel) {
        const labels = { default: 'Model 1 — Default', no_yt: 'Model 2 — No-YT', compare: 'Both Models' };
        UI.thresholdModelLabel.textContent = labels[mv] || mv;
    }
}

function handleFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'].includes(ext)) {
        showError(`Invalid format: .${ext}. Use MP4/AVI/MKV.`);
        return;
    }
    AppState.inputs.file = file;
    UI.uploadLabel.textContent = file.name;
    UI.uploadSize.textContent = (file.size / (1024*1024)).toFixed(2) + " MB";
    UI.uploadZone.classList.add('has-file');
    hideError();
    validateExecutionState();
}

function validateExecutionState() {
    const canRun = AppState.apiReady && AppState.inputs.file && !AppState.isExecuting;
    UI.runBtn.disabled = !canRun;
}

function showError(msg) {
    UI.errorBox.textContent = msg;
    UI.errorBox.classList.remove('hidden');
}

function hideError() {
    UI.errorBox.classList.add('hidden');
}

// --- Execution Pipeline ---
async function executePipeline() {
    AppState.isExecuting = true;
    AppState.telemetryHistory = [];
    AppState.telemetrySummary = null;
    validateExecutionState();
    
    UI.welcomeScreen.classList.add('hidden');
    UI.dashboardGrid.classList.add('hidden');
    UI.telemetryScreen.classList.remove('hidden');
    UI.sysLog.innerHTML = '';
    hideError();

    // Reset sparklines and force resize now that the screen is visible
    cpuSpark.reset(); ramSpark.reset(); gpuSpark.reset();
    requestAnimationFrame(() => {
        cpuSpark.resize(); ramSpark.resize(); gpuSpark.resize();
    });

    const taskId = crypto.randomUUID();
    const evtSource = new EventSource(`${API_BASE}/telemetry/${taskId}`);
    
    let lastLog = '';
    evtSource.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.status === "done") return;
        
        // Update Sparklines
        cpuSpark.push(data.cpu_percent || 0);
        UI.telCpuVal.textContent = (data.cpu_percent || 0).toFixed(1) + "%";
        
        // RAM max roughly 32GB for scaling
        const ramPct = (data.ram_gb / 32) * 100;
        ramSpark.push(ramPct);
        UI.telRamVal.textContent = (data.ram_gb || 0).toFixed(1) + " GB";
        
        gpuSpark.push(data.gpu_percent || 0);
        UI.telGpuVal.textContent = (data.gpu_percent || 0).toFixed(1) + "%";
        
        // Update Colab-style usage bars
        document.getElementById('barCpu').style.width = (data.cpu_percent || 0) + '%';
        document.getElementById('barRam').style.width = ramPct + '%';
        document.getElementById('barGpu').style.width = (data.gpu_percent || 0) + '%';
        
        UI.telStage.textContent = data.stage;

        // Persist telemetry data point for post-run summary
        AppState.telemetryHistory.push({
            ts:    performance.now(),
            cpu:   data.cpu_percent || 0,
            ram:   data.ram_gb || 0,
            gpu:   data.gpu_percent || 0,
            vram:  data.vram_gb || 0,
            stage: data.stage,
        });
        
        if (data.log_message !== lastLog) {
            const p = document.createElement('p');
            p.textContent = data.log_message;
            p.className = 'new-log';
            UI.sysLog.appendChild(p);
            UI.sysLog.scrollTop = UI.sysLog.scrollHeight;
            
            // Remove highlight from previous
            Array.from(UI.sysLog.children).slice(0, -1).forEach(child => {
                child.classList.remove('new-log');
            });
            lastLog = data.log_message;
        }
    };

    const isCompare = AppState.inputs.modelVersion === 'compare';
    const endpoint = AppState.inputs.mode === 'ensemble' ? '/predict/ensemble' : '/predict';
    
    const formData = new FormData();
    formData.append('video', AppState.inputs.file);

    try {
        if (isCompare) {
            // Run both models
            const formData2 = new FormData();
            formData2.append('video', AppState.inputs.file);
            const [res1, res2] = await Promise.all([
                fetch(`${API_BASE}${endpoint}?model_version=default&task_id=${taskId}`, { method: 'POST', body: formData }),
                fetch(`${API_BASE}${endpoint}?model_version=no_yt&task_id=${taskId}_b`, { method: 'POST', body: formData2 })
            ]);
            if (!res1.ok) { const err = await res1.json(); throw new Error(err.detail || `Server Error ${res1.status}`); }
            if (!res2.ok) { const err = await res2.json(); throw new Error(err.detail || `Server Error ${res2.status}`); }
            AppState.results = await res1.json();
            AppState.compareResults = await res2.json();
        } else {
            const url = `${API_BASE}${endpoint}?model_version=${AppState.inputs.modelVersion}&task_id=${taskId}`;
            const res = await fetch(url, { method: 'POST', body: formData });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || `Server Error ${res.status}`);
            }
            AppState.results = await res.json();
            AppState.compareResults = null;
        }

        AppState.telemetrySummary = computeTelemetrySummary(AppState.telemetryHistory);
        
        UI.telemetryScreen.classList.add('hidden');
        UI.dashboardGrid.classList.remove('hidden');
        renderDashboard();

    } catch (e) {
        showError(`Pipeline Failed: ${e.message}`);
        UI.telemetryScreen.classList.add('hidden');
        UI.welcomeScreen.classList.remove('hidden');
    } finally {
        evtSource.close();
        AppState.isExecuting = false;
        validateExecutionState();
        checkHealth();
    }
}

// --- Render Logic ---
function renderDashboard() {
    const isCompare = AppState.inputs.modelVersion === 'compare';

    // Toggle single vs compare layouts
    if (isCompare) {
        UI.singleVerdictCard.classList.add('hidden');
        UI.singleMetricsCard.classList.add('hidden');
        UI.comparisonVerdictGrid.classList.remove('hidden');
        UI.comparisonMetricsGrid.classList.remove('hidden');
    } else {
        UI.singleVerdictCard.classList.remove('hidden');
        UI.singleMetricsCard.classList.remove('hidden');
        UI.comparisonVerdictGrid.classList.add('hidden');
        UI.comparisonMetricsGrid.classList.add('hidden');
    }

    renderVerdict();
    renderMetrics();
    renderCharts();
    renderHardwareSummary();

    
    UI.transcriptBox.textContent = AppState.results.transcript || "No speech detected.";
    UI.metaFile.textContent = `File: ${AppState.results.video_name}`;
    UI.metaFolds.textContent = `Folds Evaluated: ${AppState.results.folds_used} | Mode: ${AppState.inputs.mode.toUpperCase()}`;
}

function renderVerdict() {
    const isCompare = AppState.inputs.modelVersion === 'compare';
    if (isCompare && AppState.compareResults) {
        renderCompareVerdict();
        return;
    }
    const res = AppState.results;
    const isBrainrot = res.prob_brainrot >= AppState.threshold;
    
    UI.verdictCard.className = `card verdict-card ${isBrainrot ? 'state-brainrot' : 'state-clean'}`;
    UI.verdictLabel.textContent = isBrainrot ? "BRAINROT" : "CLEAN (NON-BRAINROT)";
    
    let dynamicConf = isBrainrot ? res.prob_brainrot : (1 - res.prob_brainrot);
        
    UI.confVal.textContent = `Conf: ${(dynamicConf * 100).toFixed(1)}%`;
    UI.probBR.textContent = (res.prob_brainrot * 100).toFixed(1) + "%";
    UI.probClean.textContent = (res.prob_non_brainrot * 100).toFixed(1) + "%";
}

function renderCompareVerdict() {
    const r1 = AppState.results;
    const r2 = AppState.compareResults;
    const t1 = THRESHOLD_MAP.default;
    const t2 = THRESHOLD_MAP.no_yt;

    // Default panel
    const isBR1 = r1.prob_brainrot >= t1;
    const panel1 = document.getElementById('comparePanel_default');
    panel1.className = `comparison-panel ${isBR1 ? 'state-brainrot' : 'state-clean'}`;
    document.getElementById('compareVerdict_default').textContent = isBR1 ? 'BRAINROT' : 'CLEAN';
    document.getElementById('compareProb_default_br').textContent = (r1.prob_brainrot * 100).toFixed(1) + '%';
    document.getElementById('compareProb_default_cl').textContent = (r1.prob_non_brainrot * 100).toFixed(1) + '%';
    const conf1 = isBR1 ? r1.prob_brainrot : (1 - r1.prob_brainrot);
    document.getElementById('compareConf_default').textContent = `Confidence: ${(conf1 * 100).toFixed(1)}%`;

    // No-YT panel
    const isBR2 = r2.prob_brainrot >= t2;
    const panel2 = document.getElementById('comparePanel_no_yt');
    panel2.className = `comparison-panel ${isBR2 ? 'state-brainrot' : 'state-clean'}`;
    document.getElementById('compareVerdict_no_yt').textContent = isBR2 ? 'BRAINROT' : 'CLEAN';
    document.getElementById('compareProb_no_yt_br').textContent = (r2.prob_brainrot * 100).toFixed(1) + '%';
    document.getElementById('compareProb_no_yt_cl').textContent = (r2.prob_non_brainrot * 100).toFixed(1) + '%';
    const conf2 = isBR2 ? r2.prob_brainrot : (1 - r2.prob_brainrot);
    document.getElementById('compareConf_no_yt').textContent = `Confidence: ${(conf2 * 100).toFixed(1)}%`;
}

function renderMetrics() {
    const isCompare = AppState.inputs.modelVersion === 'compare';
    if (isCompare && AppState.compareResults) {
        renderCompareMetricsPanel('default', AppState.results);
        renderCompareMetricsPanel('no_yt', AppState.compareResults);
        return;
    }
    const p = AppState.results.pipeline_metrics;
    if(!p) return;
    
    const maxTime = Math.max(p.visual_ext_s, p.audio_ext_s, p.text_ext_s, p.inference_s);
    
    const rows = [
        { label: 'Visual Extraction', val: p.visual_ext_s },
        { label: 'Audio Extraction', val: p.audio_ext_s },
        { label: 'Text Extraction (Whisper)', val: p.text_ext_s },
        { label: 'Fusion & Inference', val: p.inference_s }
    ];
    
    UI.metricsList.innerHTML = rows.map(r => {
        const pct = (r.val / maxTime) * 100;
        return `
        <div class="metric-row flex-between">
            <span>${r.label}</span>
            <span class="text-accent">${r.val.toFixed(2)}s</span>
        </div>
        <div class="metric-bar-wrap mb-2">
            <div class="metric-fill" style="width: ${pct}%"></div>
        </div>`;
    }).join("");
    
    UI.totalTimeLabel.textContent = AppState.results.processing_time_s.toFixed(2) + "s";
}

function renderCompareMetricsPanel(key, data) {
    const p = data.pipeline_metrics;
    const listEl = document.getElementById(`compareMetrics_${key}`);
    const totalEl = document.getElementById(`compareTotalTime_${key}`);
    if (!p || !listEl) return;
    const maxTime = Math.max(p.visual_ext_s, p.audio_ext_s, p.text_ext_s, p.inference_s);
    const rows = [
        { label: 'Visual Extraction', val: p.visual_ext_s },
        { label: 'Audio Extraction', val: p.audio_ext_s },
        { label: 'Text Extraction', val: p.text_ext_s },
        { label: 'Fusion & Inference', val: p.inference_s }
    ];
    listEl.innerHTML = rows.map(r => {
        const pct = (r.val / maxTime) * 100;
        return `<div class="metric-row flex-between"><span>${r.label}</span><span class="text-accent">${r.val.toFixed(2)}s</span></div>
        <div class="metric-bar-wrap mb-2"><div class="metric-fill" style="width: ${pct}%"></div></div>`;
    }).join('');
    totalEl.textContent = data.processing_time_s.toFixed(2) + 's';
}

function renderCharts() {
    const tc = getThemeColors();
    const ctxModality = document.getElementById('modalityChart').getContext('2d');
    const ctxTimeline = document.getElementById('timelineChart').getContext('2d');
    
    if(modalityChartInstance) modalityChartInstance.destroy();
    if(timelineChartInstance) timelineChartInstance.destroy();

    // 1. Modality Radar
    const weights = AppState.results.modality_weights || [0.33, 0.33, 0.34];
    modalityChartInstance = new Chart(ctxModality, {
        type: 'radar',
        data: {
            labels: ['Visual', 'Audio', 'Text'],
            datasets: [{
                label: 'Attention Weight',
                data: weights,
                backgroundColor: tc.radarFill,
                borderColor: tc.accent,
                pointBackgroundColor: tc.accent,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { r: { angleLines: { color: tc.gridLine }, grid: { color: tc.gridLine }, pointLabels: { color: tc.textMain }, ticks: { display: false } } },
            plugins: { legend: { display: false } }
        }
    });

    // 2. Timeline Graph
    let temporalData = AppState.results.temporal_probs || [];
    if(temporalData.length === 0) temporalData = [AppState.results.prob_brainrot];

    const labels = temporalData.map((_, i) => `${(i+1)}f`);
    
    const gradient = ctxTimeline.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, tc.brGradTop);
    gradient.addColorStop(1, tc.brGradBot);

    timelineChartInstance = new Chart(ctxTimeline, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Brainrot Prob Density',
                data: temporalData,
                borderColor: tc.brainrot,
                backgroundColor: gradient,
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { min: 0, max: 1, grid: { color: tc.gridLine } },
                x: { grid: { display: false } }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}



// --- Telemetry Aggregation & Post-Run Summary ---

function computeTelemetrySummary(history) {
    if (!history.length) return null;

    const cpuArr  = history.map(h => h.cpu);
    const gpuArr  = history.map(h => h.gpu);
    const ramArr  = history.map(h => h.ram);
    const vramArr = history.map(h => h.vram);
    const t0      = history[0].ts;

    return {
        peakCpu:  Math.max(...cpuArr),
        avgCpu:   cpuArr.reduce((a, b) => a + b, 0) / cpuArr.length,
        peakGpu:  Math.max(...gpuArr),
        avgGpu:   gpuArr.reduce((a, b) => a + b, 0) / gpuArr.length,
        maxRam:   Math.max(...ramArr),
        maxVram:  Math.max(...vramArr),
        avgRam:   ramArr.reduce((a, b) => a + b, 0) / ramArr.length,
        duration: (history[history.length - 1].ts - t0) / 1000,
        samples:  history.length,
        timeline: history.map(h => ({
            t:   parseFloat(((h.ts - t0) / 1000).toFixed(1)),
            cpu: h.cpu,
            gpu: h.gpu,
            ram: h.ram,
        })),
    };
}

function renderHardwareSummary() {
    const s = AppState.telemetrySummary;
    const card = document.getElementById('hardwareCard');
    if (!s) { card.classList.add('hidden'); return; }
    card.classList.remove('hidden');

    const grid = document.getElementById('hwMetricsGrid');
    const tc = getThemeColors();
    const tiles = [
        { label: 'Peak CPU',  peak: s.peakCpu.toFixed(1) + '%',  avg: `Avg: ${s.avgCpu.toFixed(1)}%`,  color: tc.accent },
        { label: 'Peak GPU',  peak: s.peakGpu.toFixed(1) + '%',  avg: `Avg: ${s.avgGpu.toFixed(1)}%`,  color: tc.brainrot },
        { label: 'Max RAM',   peak: s.maxRam.toFixed(1) + ' GB',  avg: `Avg: ${s.avgRam.toFixed(1)} GB`, color: tc.clean },
        { label: 'Max VRAM',  peak: s.maxVram.toFixed(2) + ' GB', avg: `${s.samples} samples · ${s.duration.toFixed(1)}s`, color: tc.warning },
    ];
    grid.innerHTML = tiles.map(m => `
        <div class="hw-metric-tile">
            <div class="hw-label">${m.label}</div>
            <div class="hw-peak" style="color:${m.color}">${m.peak}</div>
            <div class="hw-avg">${m.avg}</div>
        </div>
    `).join('');

    renderTrajectoryChart(s.timeline);
}

function renderTrajectoryChart(timeline) {
    const tc = getThemeColors();
    const ctx = document.getElementById('hwTrajectoryChart').getContext('2d');
    if (hwTrajectoryChartInstance) hwTrajectoryChartInstance.destroy();

    const labels = timeline.map(p => p.t + 's');

    const cpuGrad = ctx.createLinearGradient(0, 0, 0, 160);
    cpuGrad.addColorStop(0, isDarkMode() ? 'rgba(138,155,163,0.3)' : 'rgba(92,107,115,0.2)');
    cpuGrad.addColorStop(1, 'transparent');

    const gpuGrad = ctx.createLinearGradient(0, 0, 0, 160);
    gpuGrad.addColorStop(0, isDarkMode() ? 'rgba(224,128,112,0.3)' : 'rgba(200,100,84,0.2)');
    gpuGrad.addColorStop(1, 'transparent');

    const ramGrad = ctx.createLinearGradient(0, 0, 0, 160);
    ramGrad.addColorStop(0, isDarkMode() ? 'rgba(125,160,141,0.3)' : 'rgba(96,128,112,0.2)');
    ramGrad.addColorStop(1, 'transparent');

    hwTrajectoryChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'CPU %',
                    data: timeline.map(p => p.cpu),
                    borderColor: tc.accent,
                    backgroundColor: cpuGrad,
                    fill: true, tension: 0.3, pointRadius: 0, borderWidth: 1.5,
                },
                {
                    label: 'GPU %',
                    data: timeline.map(p => p.gpu),
                    borderColor: tc.brainrot,
                    backgroundColor: gpuGrad,
                    fill: true, tension: 0.3, pointRadius: 0, borderWidth: 1.5,
                },
                {
                    label: 'RAM (GB)',
                    data: timeline.map(p => p.ram),
                    borderColor: tc.clean,
                    backgroundColor: ramGrad,
                    fill: true, tension: 0.3, pointRadius: 0, borderWidth: 1.5,
                    yAxisID: 'yRam',
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                y: {
                    min: 0, max: 100,
                    grid: { color: tc.gridLine },
                    title: { display: true, text: '% Load', color: tc.textMuted, font: { size: 10 } },
                    ticks: { font: { size: 9 } },
                },
                yRam: {
                    position: 'right',
                    min: 0,
                    grid: { display: false },
                    title: { display: true, text: 'GB', color: tc.textMuted, font: { size: 10 } },
                    ticks: { font: { size: 9 } },
                },
                x: {
                    grid: { display: false },
                    ticks: { font: { size: 9 }, maxTicksLimit: 12 },
                },
            },
            plugins: {
                legend: {
                    labels: { boxWidth: 10, padding: 12, font: { size: 10 } },
                },
                tooltip: {
                    backgroundColor: tc.tooltipBg,
                    borderColor: tc.tooltipBorder,
                    titleColor: tc.textMain,
                    bodyColor: tc.textMuted,
                    borderWidth: 1,
                    titleFont: { size: 11 },
                    bodyFont: { size: 10 },
                    padding: 10,
                },
            },
        },
    });
}
