// Welcome, Preview, Pipeline screens

const { useState, useEffect, useRef, useMemo } = React;

// ─── Logo / mark ───────────────────────────────────────────────────────────
function Mark({ size = 22 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 22 22" fill="none" aria-hidden="true">
      <rect x="1" y="1" width="9" height="9" rx="1.5" stroke="currentColor" strokeWidth="1.2" />
      <rect x="12" y="1" width="9" height="9" rx="1.5" stroke="currentColor" strokeWidth="1.2" opacity="0.45" />
      <rect x="1" y="12" width="9" height="9" rx="1.5" stroke="currentColor" strokeWidth="1.2" opacity="0.45" />
      <rect x="12" y="12" width="9" height="9" rx="1.5" stroke="currentColor" strokeWidth="1.2" />
      <circle cx="16.5" cy="16.5" r="1.6" fill="currentColor" />
    </svg>
  );
}

function TopBar({ phase, onReset }) {
  const phaseLabels = { welcome: 'Idle', preview: 'Dataset loaded', running: 'Pipeline running', done: 'Run complete' };
  return (
    <header className="topbar">
      <div className="topbar-left">
        <Mark size={20} />
        <div className="topbar-title">
          <div className="topbar-name">Agents for Data Quality</div>
          <div className="topbar-sub">Hybrid deterministic · LLM remediation pipeline</div>
        </div>
      </div>
      <div className="topbar-right">
        <span className={`phase-pill phase-${phase}`}>
          <span className="phase-dot"></span>
          {phaseLabels[phase]}
        </span>
        {phase !== 'welcome' && (
          <button className="btn-ghost" onClick={onReset}>New run</button>
        )}
      </div>
    </header>
  );
}

// ─── Welcome screen ────────────────────────────────────────────────────────
function WelcomeScreen({ onUpload, onUseDemo }) {
  const fileRef = useRef(null);
  return (
    <div className="screen welcome">
      <div className="welcome-grid">
        <section className="welcome-hero">
          <div className="eyebrow">CSV Quality Assessment · v0.4</div>
          <h1 className="hero-title">
            A reliability score for your data,<br/>
            grounded in <span className="hero-em">deterministic checks</span> and <span className="hero-em">surgical LLM reasoning</span>.
          </h1>
          <p className="hero-lede">
            Upload a CSV. Nine pipeline nodes — five deterministic Python steps and four
            LLM agents — produce a reliability score from&nbsp;0 to&nbsp;100, a corrected
            dataset, and an auditable log of every decision.
          </p>

          <div className="hero-cta">
            <input
              ref={fileRef}
              type="file"
              accept=".csv,text/csv"
              style={{ display: 'none' }}
              onChange={(e) => { if (e.target.files?.[0]) onUpload(e.target.files[0]); }}
            />
            <button className="btn-primary" onClick={() => fileRef.current?.click()}>
              Upload CSV
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M7 1v9M3 6l4-4 4 4M2 12h10" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/></svg>
            </button>
            <button className="btn-secondary" onClick={onUseDemo}>
              Try with sample.csv
            </button>
            <span className="cta-hint">Drag &amp; drop a file anywhere on this page</span>
          </div>
        </section>

        <aside className="welcome-side">
          <div className="side-card">
            <div className="side-card-h">Why hybrid</div>
            <ul className="side-list">
              <li><strong>Deterministic first.</strong> Rule discovery and 9 audit tools catch what they're built to catch — fast, reproducible, no model variance.</li>
              <li><strong>LLM where it earns its place.</strong> Four agents (Schema, Completeness, Consistency, Anomaly) plan fixes only when domain reasoning is required.</li>
              <li><strong>Auditable.</strong> Every correction logged with the agent that proposed it, the action applied, and a rationale.</li>
            </ul>
          </div>

          <div className="side-card">
            <div className="side-card-h">ISO-8000 dimensions scored</div>
            <div className="dim-grid">
              <div className="dim-row"><span>Validity</span>      <span className="mono">20%</span></div>
              <div className="dim-row"><span>Completeness</span>  <span className="mono">30%</span></div>
              <div className="dim-row"><span>Consistency</span>   <span className="mono">25%</span></div>
              <div className="dim-row"><span>Uniqueness</span>    <span className="mono">15%</span></div>
              <div className="dim-row"><span>Accuracy</span>      <span className="mono">10%</span></div>
            </div>
          </div>
        </aside>
      </div>

      <div className="welcome-footer">
        <div className="ft-cell">
          <div className="ft-num">9</div>
          <div className="ft-lbl">pipeline nodes</div>
        </div>
        <div className="ft-cell">
          <div className="ft-num">5</div>
          <div className="ft-lbl">deterministic steps</div>
        </div>
        <div className="ft-cell">
          <div className="ft-num">4</div>
          <div className="ft-lbl">LLM agents</div>
        </div>
        <div className="ft-cell">
          <div className="ft-num">5</div>
          <div className="ft-lbl">ISO-8000 dimensions</div>
        </div>
        <div className="ft-cell">
          <div className="ft-num mono">0–100</div>
          <div className="ft-lbl">reliability score</div>
        </div>
      </div>
    </div>
  );
}

// ─── Dataset preview ───────────────────────────────────────────────────────
function PreviewScreen({ dataset, onRun, onCancel }) {
  const cols = dataset.columns;
  const previewCols = cols.slice(0, 8);
  return (
    <div className="screen preview">
      <div className="screen-head">
        <div>
          <div className="eyebrow">Step 1 of 2 · Review</div>
          <h2 className="screen-title">Dataset loaded</h2>
        </div>
        <div className="screen-actions">
          <button className="btn-ghost" onClick={onCancel}>Cancel</button>
          <button className="btn-primary" onClick={onRun}>
            Run pipeline
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M3 1l8 6-8 6V1z" fill="currentColor"/></svg>
          </button>
        </div>
      </div>

      <div className="preview-meta">
        <div className="pm-cell pm-file">
          <div className="pm-lbl">File</div>
          <div className="pm-val mono">{dataset.filename}</div>
          <div className="pm-sub">{dataset.size}</div>
        </div>
        <div className="pm-cell">
          <div className="pm-lbl">Rows</div>
          <div className="pm-val mono">{dataset.rows.toLocaleString()}</div>
        </div>
        <div className="pm-cell">
          <div className="pm-lbl">Columns</div>
          <div className="pm-val mono">{dataset.cols}</div>
        </div>
        <div className="pm-cell pm-types">
          <div className="pm-lbl">Type distribution</div>
          <div className="type-bar">
            <div className="type-seg type-numeric" style={{ flex: dataset.types.numeric }} title={`${dataset.types.numeric} numeric`} />
            <div className="type-seg type-string"  style={{ flex: dataset.types.string }}  title={`${dataset.types.string} string`} />
            <div className="type-seg type-date"    style={{ flex: dataset.types.date }}    title={`${dataset.types.date} date`} />
          </div>
          <div className="type-legend">
            <span><i className="dot type-numeric"></i> {dataset.types.numeric} numeric</span>
            <span><i className="dot type-string"></i>  {dataset.types.string} string</span>
            <span><i className="dot type-date"></i>    {dataset.types.date} date</span>
          </div>
        </div>
      </div>

      <div className="table-card">
        <div className="table-card-h">
          <span>First {dataset.preview.length} rows</span>
          <span className="table-card-hint mono">{previewCols.length} of {cols.length} columns shown</span>
        </div>
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                {previewCols.map(c => <th key={c}>{c}</th>)}
                <th className="th-more">+{cols.length - previewCols.length}</th>
              </tr>
            </thead>
            <tbody>
              {dataset.preview.map((row, i) => (
                <tr key={i}>
                  {previewCols.map(c => (
                    <td key={c} className={!row[c] || row[c] === '—' ? 'td-null' : ''}>
                      {row[c] || <span className="null-mark">∅</span>}
                    </td>
                  ))}
                  <td className="td-more">…</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="preview-foot">
        <span className="mono dim">Ready to run · estimated 18s on this dataset</span>
      </div>
    </div>
  );
}

window.WelcomeScreen = WelcomeScreen;
window.PreviewScreen = PreviewScreen;
window.TopBar = TopBar;
window.Mark = Mark;
