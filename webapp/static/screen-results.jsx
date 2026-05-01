// Results dashboard

const { useState: useStateR, useEffect: useEffectR, useRef: useRefR, useMemo: useMemoR } = React;

function useCountUp(target, duration = 1200, start = 0) {
  const [val, setVal] = useStateR(start);
  useEffectR(() => {
    let raf, t0;
    const step = (t) => {
      if (!t0) t0 = t;
      const p = Math.min(1, (t - t0) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      setVal(start + (target - start) * eased);
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target, duration]);
  return val;
}

function ResultsScreen({ onReset }) {
  const R = window.RESULTS;
  const [selectedIssue, setSelectedIssue] = useStateR(null);
  const [tab, setTab] = useStateR('issues');

  // The verdict reflects the post-fix score when available — that's the score
  // that describes the *delivered* dataset quality, not the input quality.
  const hasPost = R.post_reliability_score != null;
  const displayScore = hasPost ? R.post_reliability_score : R.reliability_score;
  const score = useCountUp(displayScore, 1400);
  const preScore = useCountUp(R.reliability_score, 1400);
  const verdict = displayScore >= 70 ? 'high' : displayScore >= 40 ? 'medium' : 'low';
  const verdictLabel = { high: 'High reliability', medium: 'Medium reliability', low: 'Low reliability' }[verdict];
  const delta = hasPost ? +(R.post_reliability_score - R.reliability_score).toFixed(1) : null;

  return (
    <div className="screen results">
      <div className="screen-head">
        <div>
          <div className="eyebrow">Run complete · sample.csv</div>
          <h2 className="screen-title">Results</h2>
        </div>
        <div className="screen-actions">
          <ExportMenu />
          <button className="btn-ghost" onClick={onReset}>New run</button>
        </div>
      </div>

      {/* Hero: score + sub-scores side by side */}
      <div className="results-hero">
        <div className={`score-card score-${verdict}`}>
          <div className="score-card-h">{hasPost ? 'Reliability — post-fix' : 'Reliability score'}</div>
          <div className="score-card-num">
            <span className="score-num mono">{score.toFixed(1)}</span>
            <span className="score-denom mono">/ 100</span>
          </div>
          <div className="score-verdict">
            <span className="verdict-dot"></span>
            {verdictLabel}
          </div>
          {hasPost && (
            <div className="score-delta mono" style={{
              marginTop: 8, fontSize: 13,
              color: delta > 0 ? '#0a6e2c' : delta < 0 ? '#a31515' : '#666'
            }}>
              before {preScore.toFixed(1)} → after {score.toFixed(1)}
              <span style={{ marginLeft: 6, fontWeight: 600 }}>
                ({delta >= 0 ? '+' : ''}{delta.toFixed(1)})
              </span>
            </div>
          )}
          <div className="score-formula mono">
            weighted sum · 5 ISO-8000 dimensions
          </div>
        </div>

        <div className="subscores-card">
          <div className="subscores-h">
            <span>Sub-scores by dimension {hasPost && <span className="dim" style={{fontWeight:400}}>· post-fix</span>}</span>
            <span className="mono dim">weight × score = contribution</span>
          </div>
          <div className="subscores-grid">
            {Object.entries(hasPost ? R.post_sub_scores : R.sub_scores).map(([dim, val]) => (
              <SubScoreRow
                key={dim} dim={dim} value={val} weight={R.weights[dim]}
                preValue={hasPost ? R.sub_scores[dim] : null}
              />
            ))}
          </div>
        </div>

        <div className="severity-card">
          <div className="severity-h">Severity breakdown</div>
          <div className="severity-list">
            {Object.entries(R.severity_breakdown).map(([sev, count]) => {
              const postCount = hasPost ? (R.post_severity_breakdown?.[sev] ?? 0) : null;
              const resolved = hasPost ? (count - postCount) : null;
              return (
                <div key={sev} className={`sev-row sev-${sev}`}>
                  <span className="sev-dot"></span>
                  <span className="sev-label">{sev}</span>
                  <span className="sev-count mono">
                    {hasPost ? `${count} → ${postCount}` : count}
                    {hasPost && resolved > 0 && (
                      <span style={{ color: '#0a6e2c', marginLeft: 4, fontSize: 11 }}>
                        (-{resolved})
                      </span>
                    )}
                  </span>
                </div>
              );
            })}
          </div>
          <div className="severity-foot mono dim">
            {R.issues.length} issues pre-fix
            {hasPost && ` · ${(R.post_severity_breakdown ? Object.values(R.post_severity_breakdown).reduce((a,b)=>a+b,0) : 0)} residual`}
          </div>
        </div>
      </div>

      {/* Tabs: issues | corrections | fixed preview | audit */}
      <div className="results-tabs">
        <div className="tab-bar">
          {[
            { id: 'issues',     label: 'Detected issues',  count: R.issues.length },
            { id: 'log',        label: 'Correction log',   count: R.correction_log.length },
            { id: 'fixed',      label: 'Fixed dataset' },
            { id: 'audit',      label: 'Audit trail',      count: R.audit_trail.length },
          ].map(t => (
            <button key={t.id}
              className={`tab ${tab === t.id ? 'tab-active' : ''}`}
              onClick={() => setTab(t.id)}>
              {t.label}
              {t.count != null && <span className="tab-count mono">{t.count}</span>}
            </button>
          ))}
        </div>

        <div className="tab-body">
          {tab === 'issues' && <IssuesTable issues={R.issues} onSelect={setSelectedIssue} selected={selectedIssue} />}
          {tab === 'log' && <CorrectionLog log={R.correction_log} />}
          {tab === 'fixed' && <FixedPreview rows={window.FIXED_PREVIEW} />}
          {tab === 'audit' && <AuditTrail trail={R.audit_trail} />}
        </div>
      </div>

      {selectedIssue && (
        <IssuePanel issue={selectedIssue} onClose={() => setSelectedIssue(null)} />
      )}
    </div>
  );
}

function SubScoreRow({ dim, value, weight, preValue }) {
  const animVal = useCountUp(value, 1100);
  const contribution = (value * weight).toFixed(1);
  const tone = value >= 70 ? 'good' : value >= 40 ? 'mid' : 'bad';
  const hasDelta = preValue != null && Math.abs(preValue - value) >= 0.5;
  return (
    <div className="ss-row">
      <div className="ss-row-top">
        <span className="ss-dim">{dim}</span>
        <span className="mono ss-weight">×{weight.toFixed(2)}</span>
        <span className={`mono ss-val ss-val-${tone}`}>{Math.round(animVal)}</span>
      </div>
      <div className="ss-bar">
        <div className={`ss-bar-fill ss-${tone}`} style={{ width: `${animVal}%` }} />
      </div>
      <div className="ss-contrib mono dim">
        contributes {contribution} pts
        {hasDelta && (
          <span style={{ marginLeft: 6, color: value >= preValue ? '#0a6e2c' : '#a31515' }}>
            · {Math.round(preValue)} → {Math.round(value)}
          </span>
        )}
      </div>
    </div>
  );
}

function IssuesTable({ issues, onSelect, selected }) {
  return (
    <div className="table-card flush">
      <table className="data-table issues-table">
        <thead>
          <tr>
            <th>Tool</th>
            <th>Issue</th>
            <th>Severity</th>
            <th>Columns</th>
            <th className="num">Rows</th>
            <th>Message</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {issues.map(i => (
            <tr key={i.id}
                onClick={() => onSelect(i)}
                className={`issue-row ${selected?.id === i.id ? 'row-selected' : ''}`}>
              <td className="mono">{i.tool}</td>
              <td>{i.issue_type}</td>
              <td><span className={`sev-pill sev-${i.severity}`}>{i.severity}</span></td>
              <td className="mono">{i.columns.join(', ')}</td>
              <td className="num mono">{i.row_count.toLocaleString()}</td>
              <td className="td-msg">{i.message}</td>
              <td className="td-chev">›</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CorrectionLog({ log }) {
  const agentTone = { Schema: 'a-schema', Completeness: 'a-completeness', Consistency: 'a-consistency', Anomaly: 'a-anomaly' };
  return (
    <div className="table-card flush">
      <table className="data-table log-table">
        <thead>
          <tr>
            <th>Agent</th>
            <th>Action</th>
            <th>Column</th>
            <th className="num">Rows</th>
            <th>Rationale</th>
          </tr>
        </thead>
        <tbody>
          {log.map(c => (
            <tr key={c.id}>
              <td><span className={`agent-pill ${agentTone[c.agent]}`}>{c.agent}</span></td>
              <td className="mono">{c.action}</td>
              <td className="mono">{c.column}</td>
              <td className="num mono">{c.rows_affected.toLocaleString()}</td>
              <td className="td-rationale">{c.rationale}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FixedPreview({ rows }) {
  const cols = ['id', 'cognome', 'nome', 'data_nascita', 'comune', 'spesa', 'data_pagamento', 'flag_attivo'];
  return (
    <div className="table-card flush">
      <div className="table-card-h">
        <span>First {rows.length} rows of corrected dataset</span>
        <span className="mono dim"><span className="mod-mark">*</span> = value modified by remediation</span>
      </div>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>{cols.map(c => <th key={c}>{c}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                {cols.map(c => {
                  const v = r[c] || '';
                  const modified = v.toString().includes('*');
                  return <td key={c} className={modified ? 'td-modified' : ''}>{v}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function AuditTrail({ trail }) {
  return (
    <div className="audit-trail-card">
      {trail.map((line, i) => (
        <div key={i} className="audit-trail-line">
          <span className="audit-trail-i mono">{String(i + 1).padStart(2, '0')}</span>
          <span className="audit-trail-msg mono">{line}</span>
        </div>
      ))}
    </div>
  );
}

function IssuePanel({ issue, onClose }) {
  return (
    <>
      <div className="panel-scrim" onClick={onClose} />
      <aside className="side-panel" role="dialog">
        <div className="side-panel-h">
          <div>
            <div className="eyebrow">Issue · #{String(issue.id).padStart(3, '0')}</div>
            <div className="side-panel-title">{issue.issue_type}</div>
          </div>
          <button className="btn-icon" onClick={onClose} aria-label="Close">×</button>
        </div>

        <div className="side-panel-body">
          <div className="kv-row"><span>Tool</span> <span className="mono">{issue.tool}</span></div>
          <div className="kv-row"><span>Severity</span> <span className={`sev-pill sev-${issue.severity}`}>{issue.severity}</span></div>
          <div className="kv-row"><span>Columns</span> <span className="mono">{issue.columns.join(', ')}</span></div>
          <div className="kv-row"><span>Rows affected</span> <span className="mono">{issue.row_count.toLocaleString()}</span></div>

          <div className="panel-section">
            <div className="panel-section-h">Message</div>
            <div className="panel-msg">{issue.message}</div>
          </div>

          <div className="panel-section">
            <div className="panel-section-h">Sample affected rows</div>
            <div className="sample-rows">
              {issue.sample_rows.map(r => (
                <span key={r} className="row-chip mono">row {r.toLocaleString()}</span>
              ))}
            </div>
          </div>

          <div className="panel-section">
            <div className="panel-section-h">Resolved by</div>
            {window.RESULTS.correction_log.filter(l => l.column === issue.columns[0]).map(l => (
              <div key={l.id} className="resolution">
                <div className="res-row1">
                  <span className={`agent-pill a-${l.agent.toLowerCase()}`}>{l.agent}</span>
                  <span className="mono">{l.action}</span>
                </div>
                <div className="res-rationale">{l.rationale}</div>
              </div>
            ))}
          </div>
        </div>
      </aside>
    </>
  );
}

function ExportMenu() {
  const [open, setOpen] = useStateR(false);
  // Trigger a download from a session-id-scoped backend endpoint by clicking
  // a hidden anchor; the backend sets Content-Disposition: attachment.
  const download = (kind) => {
    const sid = window.LATEST_SESSION_ID;
    if (!sid) {
      alert("No active session — run the pipeline first.");
      return;
    }
    const url = `/download/${kind}/${sid}`;
    const a = document.createElement("a");
    a.href = url;
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setOpen(false);
  };
  return (
    <div className="export-menu">
      <button className="btn-primary" onClick={() => setOpen(!open)}>
        Export
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M2 4l3 3 3-3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/></svg>
      </button>
      {open && (
        <>
          <div className="export-scrim" onClick={() => setOpen(false)} />
          <div className="export-pop">
            <button className="export-item" onClick={() => download("fixed")}>
              <span>Fixed dataset</span><span className="mono dim">.csv</span>
            </button>
            <button className="export-item" onClick={() => download("report")}>
              <span>Quality report</span><span className="mono dim">.html</span>
            </button>
            <button className="export-item" onClick={() => download("log")}>
              <span>Correction log</span><span className="mono dim">.json</span>
            </button>
            <button className="export-item" onClick={() => download("bundle")}>
              <span>Full run bundle</span><span className="mono dim">.zip</span>
            </button>
          </div>
        </>
      )}
    </div>
  );
}

window.ResultsScreen = ResultsScreen;
