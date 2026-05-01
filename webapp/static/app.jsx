// Main app — orchestrates phases, streams real pipeline events from the FastAPI backend.

const { useState: useStateA, useEffect: useEffectA, useRef: useRefA } = React;

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "theme": "light",
  "palette": "navy"
}/*EDITMODE-END*/;

// Tweaks panel is only visible when ?dev=1 is set in the URL.
const DEV_MODE = new URLSearchParams(window.location.search).get('dev') === '1';

// ─── Palette system ────────────────────────────────────────────────────────
// Each palette overrides only the brand-color CSS variables; the theme
// (light/dark/hc) controls bg/fg. Palette + theme compose orthogonally.
const PALETTES = {
  navy:    { name: 'Navy & plum',    det: '#2d4a8a', llm: '#6b4e8c', good: '#2f6f4f', mid: '#8a6a1a', bad: '#9a3535' },
  teal:    { name: 'Teal & coral',   det: '#1f6b6e', llm: '#c46a4a', good: '#2f6f4f', mid: '#9a6a1a', bad: '#9a3535' },
  forest:  { name: 'Forest & ochre', det: '#2a5d3a', llm: '#a87830', good: '#2a5d3a', mid: '#a87830', bad: '#9a3535' },
  indigo:  { name: 'Indigo & rose',  det: '#3a3f8a', llm: '#a8456b', good: '#2f6f4f', mid: '#8a6a1a', bad: '#9a3535' },
  slate:   { name: 'Slate & amber',  det: '#3d4a5e', llm: '#b8721a', good: '#3a6b48', mid: '#b8721a', bad: '#9a3535' },
};

function softenColor(hex, amount = 0.92) {
  // Convert hex → rgb, blend with white (used for soft-tone variants in light theme)
  const r = parseInt(hex.slice(1,3), 16);
  const g = parseInt(hex.slice(3,5), 16);
  const b = parseInt(hex.slice(5,7), 16);
  const sr = Math.round(r + (255 - r) * amount);
  const sg = Math.round(g + (255 - g) * amount);
  const sb = Math.round(b + (255 - b) * amount);
  return `rgb(${sr}, ${sg}, ${sb})`;
}

function applyPalette(palette, theme) {
  const p = PALETTES[palette] || PALETTES.navy;
  const root = document.documentElement;
  // Brand colors — solid
  root.style.setProperty('--accent', p.det);
  root.style.setProperty('--det', p.det);
  root.style.setProperty('--llm', p.llm);
  root.style.setProperty('--good', p.good);
  root.style.setProperty('--mid', p.mid);
  root.style.setProperty('--bad', p.bad);
  // Soft variants depend on the active theme: dark uses alpha hex suffix,
  // light blends with white through softenColor().
  if (theme === 'dark') {
    root.style.setProperty('--accent-soft', `${p.det}1f`);
    root.style.setProperty('--det-soft',    `${p.det}1f`);
    root.style.setProperty('--llm-soft',    `${p.llm}26`);
    root.style.setProperty('--good-soft',   `${p.good}1f`);
    root.style.setProperty('--mid-soft',    `${p.mid}26`);
    root.style.setProperty('--bad-soft',    `${p.bad}26`);
  } else {
    root.style.setProperty('--accent-soft', softenColor(p.det, 0.88));
    root.style.setProperty('--det-soft',    softenColor(p.det, 0.88));
    root.style.setProperty('--llm-soft',    softenColor(p.llm, 0.86));
    root.style.setProperty('--good-soft',   softenColor(p.good, 0.88));
    root.style.setProperty('--mid-soft',    softenColor(p.mid, 0.86));
    root.style.setProperty('--bad-soft',    softenColor(p.bad, 0.88));
  }
  root.style.setProperty('--type-num', p.det);
  root.style.setProperty('--type-str', p.llm);
  root.style.setProperty('--type-dat', p.good);
}

function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
  const [phase, setPhase] = useStateA('welcome'); // welcome | preview | running | done
  const [dataset, setDataset] = useStateA(null);
  const [sessionId, setSessionId] = useStateA(null);
  const [error, setError] = useStateA(null);
  const [progress, setProgress] = useStateA({
    activeIdx: -1,
    statuses: window.PIPELINE_NODES.map(() => 'pending'),
    elapsed:  window.PIPELINE_NODES.map(() => 0),
    log: [],
  });
  const [dragging, setDragging] = useStateA(false);
  const eventSourceRef = useRefA(null);

  // Apply theme + palette together so every brand color follows the theme's bg/fg
  useEffectA(() => {
    document.documentElement.setAttribute('data-theme', t.theme);
    applyPalette(t.palette, t.theme);
  }, [t.theme, t.palette]);

  // Drag-drop CSV anywhere on the page
  useEffectA(() => {
    const onOver  = (e) => { e.preventDefault(); setDragging(true); };
    const onLeave = (e) => { if (e.target === document.documentElement || e.clientX === 0) setDragging(false); };
    const onDrop  = (e) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer?.files?.[0];
      if (f) handleUpload(f);
    };
    window.addEventListener('dragover', onOver);
    window.addEventListener('dragleave', onLeave);
    window.addEventListener('drop', onDrop);
    return () => {
      window.removeEventListener('dragover', onOver);
      window.removeEventListener('dragleave', onLeave);
      window.removeEventListener('drop', onDrop);
    };
  }, []);

  // ─── Backend calls ─────────────────────────────────────────────────────
  const handleUpload = async (file) => {
    setError(null);
    const fd = new FormData();
    fd.append('file', file);
    try {
      const resp = await fetch('/upload', { method: 'POST', body: fd });
      if (!resp.ok) throw new Error(`Upload failed: ${resp.status}`);
      const j = await resp.json();
      setSessionId(j.session_id);
      setDataset(j);
      setPhase('preview');
    } catch (e) {
      setError(`Upload error: ${e.message}`);
    }
  };

  const handleUseDemo = async () => {
    setError(null);
    try {
      const resp = await fetch('/demo', { method: 'POST' });
      if (!resp.ok) throw new Error(`Demo load failed: ${resp.status}`);
      const j = await resp.json();
      setSessionId(j.session_id);
      setDataset(j);
      setPhase('preview');
    } catch (e) {
      setError(`Demo error: ${e.message}`);
    }
  };

  // ─── Live pipeline runner (SSE) ────────────────────────────────────────
  const runPipeline = () => {
    if (!sessionId) {
      setError('No session_id; load a dataset first.');
      return;
    }
    setPhase('running');
    setProgress({
      activeIdx: 0,
      statuses: window.PIPELINE_NODES.map(() => 'pending'),
      elapsed:  window.PIPELINE_NODES.map(() => 0),
      log: [],
    });

    // Mark the first node as "running" immediately
    setProgress(p => ({
      ...p,
      statuses: p.statuses.map((s, i) => i === 0 ? 'running' : s),
    }));

    const es = new EventSource(`/run/${sessionId}`);
    eventSourceRef.current = es;
    // Flag: once `complete` fires, server closes the SSE → browser emits a
    // native `error` event for the EOF. Without this flag we'd treat the
    // normal end-of-stream as a pipeline failure and overwrite the results.
    let completed = false;

    es.addEventListener('node_done', (ev) => {
      const data = JSON.parse(ev.data);
      const idx = window.PIPELINE_NODES.findIndex(n => n.id === data.node);
      if (idx < 0) return;
      setProgress(p => ({
        ...p,
        activeIdx: Math.min(idx + 1, window.PIPELINE_NODES.length - 1),
        statuses: p.statuses.map((s, i) =>
          i === idx ? 'done' : (i === idx + 1 ? 'running' : s)
        ),
        elapsed: p.elapsed.map((e, i) => i === idx ? data.elapsed * 1000 : e),
        log: [...p.log, `${data.node}: ${data.message}`],
      }));
    });

    es.addEventListener('complete', (ev) => {
      completed = true;
      const payload = JSON.parse(ev.data);
      // Replace RESULTS with real backend data so the design's ResultsScreen reads it
      window.RESULTS = {
        dataset_name:             payload.dataset_name,
        reliability_score:        payload.reliability_score,
        post_reliability_score:   payload.post_reliability_score,
        sub_scores:               payload.sub_scores,
        post_sub_scores:          payload.post_sub_scores,
        weights:                  payload.weights,
        severity_breakdown:       payload.severity_breakdown,
        post_severity_breakdown:  payload.post_severity_breakdown,
        issues:                   payload.issues,
        correction_log:           payload.correction_log,
        audit_trail:              payload.audit_trail,
      };
      window.FIXED_PREVIEW       = payload.fixed_preview;
      window.LATEST_PROVIDER     = payload.provider;
      window.LATEST_HTML_REPORT  = payload.html_report;
      window.LATEST_SESSION_ID   = sessionId;     // used by ExportMenu to build download URLs
      es.close();
      eventSourceRef.current = null;
      setTimeout(() => setPhase('done'), 350);
    });

    es.addEventListener('error', (ev) => {
      // Normal EOF after `complete` — ignore the browser's native error event
      if (completed) return;
      let msg = 'Stream error';
      try { msg = JSON.parse(ev.data).message; } catch (_) {}
      setError(msg);
      es.close();
      eventSourceRef.current = null;
      setPhase('preview');
    });
  };

  const reset = () => {
    if (eventSourceRef.current) { eventSourceRef.current.close(); eventSourceRef.current = null; }
    setDataset(null);
    setSessionId(null);
    setError(null);
    setProgress({
      activeIdx: -1,
      statuses: window.PIPELINE_NODES.map(() => 'pending'),
      elapsed:  window.PIPELINE_NODES.map(() => 0),
      log: [],
    });
    setPhase('welcome');
  };

  return (
    <div className="app">
      <TopBar phase={phase} onReset={reset} />

      {error && (
        <div style={{
          background: '#ffd6d6', color: '#7a1f1f', padding: '10px 16px',
          borderLeft: '4px solid #c0392b', fontFamily: 'JetBrains Mono, monospace',
          fontSize: 13, margin: '8px 16px', borderRadius: 4
        }}>
          ⚠ {error}
        </div>
      )}

      {phase === 'welcome' && <WelcomeScreen onUpload={handleUpload} onUseDemo={handleUseDemo} />}
      {phase === 'preview' && <PreviewScreen dataset={dataset} onRun={runPipeline} onCancel={reset} />}
      {phase === 'running' && <PipelineScreen progress={progress} onSkip={() => {}} />}
      {phase === 'done'    && <ResultsScreen onReset={reset} />}

      {dragging && <div className="drop-overlay">Drop CSV to load</div>}

      {DEV_MODE && (
        <TweaksPanel>
          <TweakSection label="Appearance" />
          <TweakRadio label="Theme" value={t.theme}
            options={[
              { value: 'light', label: 'Light' },
              { value: 'dark',  label: 'Dark' },
              { value: 'hc',    label: 'High contrast' },
            ]}
            onChange={(v) => setTweak('theme', v)} />
          <TweakSelect label="Color palette" value={t.palette}
            options={Object.entries(PALETTES).map(([id, p]) => ({ value: id, label: p.name }))}
            onChange={(v) => setTweak('palette', v)} />
          <TweakSection label="Demo" />
          <TweakButton label="Restart from welcome" onClick={reset} />
          {phase === 'preview' && (
            <TweakButton label="Run pipeline now" onClick={runPipeline} />
          )}
        </TweaksPanel>
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
