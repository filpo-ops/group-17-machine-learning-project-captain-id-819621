// Main app — orchestrates phases, streams real pipeline events from the FastAPI backend.

const { useState: useStateA, useEffect: useEffectA, useRef: useRefA } = React;

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "theme": "light"
}/*EDITMODE-END*/;

// Tweaks panel is only visible when ?dev=1 is set in the URL.
const DEV_MODE = new URLSearchParams(window.location.search).get('dev') === '1';

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

  useEffectA(() => {
    document.documentElement.setAttribute('data-theme', t.theme);
  }, [t.theme]);

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
      // Replace mock RESULTS with real data so the existing ResultsScreen reads it
      window.RESULTS = {
        reliability_score:  payload.reliability_score,
        sub_scores:         payload.sub_scores,
        weights:            payload.weights,
        severity_breakdown: payload.severity_breakdown,
        issues:             payload.issues,
        correction_log:     payload.correction_log,
        audit_trail:        payload.audit_trail,
      };
      window.FIXED_PREVIEW = payload.fixed_preview;
      window.LATEST_PROVIDER = payload.provider;
      window.LATEST_HTML_REPORT = payload.html_report;
      window.LATEST_SESSION_ID = sessionId;     // used by ExportMenu to build download URLs
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
