let currentFile = null;
let currentMode = 'single';
let history = [];

const $ = id => document.getElementById(id);

// ── Disclaimer Modal ─────────────────────────────────────
function showDisclaimer() {
  const modal = $('disclaimerModal');
  if (modal) {
    modal.classList.add('visible');
    document.body.style.overflow = 'hidden'; // Block background scroll
  }
}

function acceptDisclaimer() {
  const modal = $('disclaimerModal');
  if (modal) {
    modal.classList.remove('visible');
    document.body.style.overflow = ''; // Restore scroll
  }
}

// Show disclaimer on load
window.addEventListener('DOMContentLoaded', showDisclaimer);

// ── Drag & Drop ─────────────────────────────────────────
const zone = $('uploadZone');
zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
zone.addEventListener('drop', e => {
  e.preventDefault();
  zone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});

function handleFile(file) {
  if (!file) return;
  const allowed = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'];
  const ext = file.name.split('.').pop().toLowerCase();
  if (!allowed.includes(ext)) {
    showError(`Unsupported format: .${ext}`);
    return;
  }
  currentFile = file;
  $('fileName').textContent = file.name;
  $('fileSize').textContent = formatBytes(file.size);
  $('filePreview').classList.add('visible');
  zone.classList.add('has-file');
  $('predictBtn').disabled = false;
  hideError();
}

function removeFile() {
  currentFile = null;
  $('fileInput').value = '';
  $('filePreview').classList.remove('visible');
  zone.classList.remove('has-file');
  $('predictBtn').disabled = true;
  $('resultPanel').classList.remove('visible');
}

function setMode(mode) {
  currentMode = mode;
  $('modeSingle').classList.toggle('active', mode === 'single');
  $('modeEnsemble').classList.toggle('active', mode === 'ensemble');
}

// ── Health Check ──────────────────────────────────────────
async function checkHealth() {
  const base = $('apiUrl').value.replace(/\/$/, '');
  const pill = $('statusPill');
  pill.className = 'status-pill';
  $('statusText').textContent = 'CHECKING...';
  try {
    const r = await fetch(`${base}/health`, { signal: AbortSignal.timeout(5000) });
    const d = await r.json();
    if (d.models_ready) {
      pill.className = 'status-pill online';
      $('statusText').textContent = `ONLINE · ${d.folds_loaded.length} FOLDS`;
    } else {
      pill.className = 'status-pill offline';
      $('statusText').textContent = 'LOADING MODELS';
    }
  } catch {
    pill.className = 'status-pill offline';
    $('statusText').textContent = 'OFFLINE';
  }
}

// Auto-check on load
checkHealth();

// ── Prediction ────────────────────────────────────────────
async function predict() {
  if (!currentFile) return;
  const base = $('apiUrl').value.replace(/\/$/, '');
  const endpoint = currentMode === 'ensemble' ? '/predict/ensemble' : '/predict';

  showLoading(true);
  hideError();
  $('resultPanel').classList.remove('visible');
  $('predictBtn').disabled = true;

  // Animate steps
  const stepTimings = [500, 1500, 2800, 4200];
  stepTimings.forEach((t, i) => {
    setTimeout(() => {
      for (let j = 0; j < 4; j++) {
        $(`step${j + 1}`).className = 'step' + (j < i ? ' done' : j === i ? ' active' : '');
      }
      const msgs = ['UPLOADING VIDEO...', 'EXTRACTING VISUAL FEATURES...', 'EXTRACTING AUDIO FEATURES...', 'RUNNING TEXT ANALYSIS...'];
      $('loadingText').textContent = msgs[i];
    }, t);
  });

  try {
    const form = new FormData();
    form.append('video', currentFile);

    const r = await fetch(`${base}${endpoint}`, { method: 'POST', body: form });
    if (!r.ok) {
      const err = await r.json();
      throw new Error(err.detail || `HTTP ${r.status}`);
    }
    const data = await r.json();
    showResult(data);
    addHistory(data);
  } catch (e) {
    showError(`Request failed: ${e.message}`);
  } finally {
    showLoading(false);
    $('predictBtn').disabled = false;
    // reset steps
    for (let i = 1; i <= 4; i++) $(`step${i}`).className = 'step';
  }
}

function showLoading(on) {
  $('loadingPanel').classList.toggle('visible', on);
}

function showError(msg) {
  const p = $('errorPanel');
  p.textContent = `⚠ ${msg}`;
  p.classList.add('visible');
}

function hideError() {
  $('errorPanel').classList.remove('visible');
}

function showResult(d) {
  const isBR = d.prediction === 'BRAINROT';
  const card = $('verdictCard');
  card.className = `verdict-card ${isBR ? 'brainrot' : 'non-brainrot'}`;

  $('verdictEmoji').textContent = isBR ? '🤪' : '✅';
  $('verdictLabel').textContent = isBR ? 'BRAINROT' : 'NOT BRAINROT';
  $('confPct').textContent = `${(d.confidence * 100).toFixed(1)}%`;
  $('probBR').textContent = `${(d.prob_brainrot * 100).toFixed(1)}%`;
  $('probNBR').textContent = `${(d.prob_non_brainrot * 100).toFixed(1)}%`;

  $('detFile').textContent = d.video_name;
  $('detTime').textContent = `${d.processing_time_s}s`;
  $('detFolds').textContent = `${d.folds_used} fold${d.folds_used > 1 ? 's' : ''}`;
  $('detMode').textContent = currentMode.toUpperCase();
  $('transcriptText').textContent = d.transcript || 'no speech detected';

  // Fold bars
  const fb = $('foldBars');
  if (d.per_fold_probs && d.per_fold_probs.length > 1) {
    fb.innerHTML = `<div class="detail-label" style="margin-bottom:0.8rem;font-family:'Space Mono',monospace;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.12em;color:var(--muted)">PER-FOLD BRAINROT PROBABILITY</div>` +
      d.per_fold_probs.map((p, i) => `
      <div class="fold-bar-row">
        <div class="fold-label">FOLD ${i + 1}</div>
        <div class="fold-track"><div class="fold-fill" id="ff${i}" style="width:0%"></div></div>
        <div class="fold-pct">${(p * 100).toFixed(1)}%</div>
      </div>`).join('');
    fb.style.display = 'block';
    setTimeout(() => {
      d.per_fold_probs.forEach((p, i) => {
        const el = document.getElementById(`ff${i}`);
        if (el) el.style.width = `${p * 100}%`;
      });
    }, 100);
  } else {
    fb.innerHTML = '';
  }

  $('resultPanel').classList.add('visible');
  setTimeout(() => {
    $('confFill').style.width = `${d.confidence * 100}%`;
  }, 100);

  $('resultPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function addHistory(d) {
  const isBR = d.prediction === 'BRAINROT';
  history.unshift({ name: d.video_name, br: isBR, conf: d.confidence, time: new Date().toLocaleTimeString() });
  if (history.length > 10) history.pop();

  const list = $('historyList');
  list.innerHTML = history.map(h => `
  <div class="history-item">
    <span class="h-badge ${h.br ? 'br' : 'nbr'}">${h.br ? 'BRAINROT' : 'CLEAN'}</span>
    <span class="h-name">${h.name}</span>
    <span class="h-conf">${(h.conf * 100).toFixed(1)}%</span>
    <span class="h-time">${h.time}</span>
  </div>`).join('');
}

function formatBytes(b) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 / 1024).toFixed(1)} MB`;
}
