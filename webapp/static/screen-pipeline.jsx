// Pipeline execution screen

const { useState: useState2, useEffect: useEffect2, useRef: useRef2, useMemo: useMemo2 } = React;

function PipelineScreen({ progress, onSkip }) {
  // progress: { activeIdx, statuses: [...], elapsed: [...], log: [strings] }
  const nodes = window.PIPELINE_NODES;
  const detNodes = nodes.filter(n => n.kind === 'det');
  const llmNodes = nodes.filter(n => n.kind === 'llm');

  const totalElapsed = progress.elapsed.reduce((a, b) => a + b, 0);
  const completedCount = progress.statuses.filter(s => s === 'done').length;
  const pct = (completedCount / nodes.length) * 100;

  return (
    <div className="screen pipeline">
      <div className="screen-head">
        <div>
          <div className="eyebrow">Step 2 of 2 · Execution</div>
          <h2 className="screen-title">Running pipeline</h2>
        </div>
        <div className="screen-actions">
          <span className="elapsed mono">{(totalElapsed / 1000).toFixed(1)}s elapsed</span>
          <button className="btn-ghost" onClick={onSkip}>Skip animation</button>
        </div>
      </div>

      <div className="pipeline-progress-bar">
        <div className="pb-track">
          <div className="pb-fill" style={{ width: `${pct}%` }} />
        </div>
        <div className="pb-label mono">{completedCount} / {nodes.length} nodes</div>
      </div>

      <div className="pipeline-stage">
        <div className="lane lane-det">
          <div className="lane-h">
            <span className="lane-tag">DETERMINISTIC</span>
            <span className="lane-sub">Python · 5 nodes · rule-based, reproducible</span>
          </div>
          <div className="lane-track">
            {detNodes.map((node, i) => {
              const idx = nodes.findIndex(n => n.id === node.id);
              return <NodeCard key={node.id} node={node} idx={idx}
                       status={progress.statuses[idx]}
                       elapsed={progress.elapsed[idx]} />;
            })}
          </div>
        </div>

        <div className="lane lane-llm">
          <div className="lane-h">
            <span className="lane-tag lane-tag-llm">LLM AGENTS</span>
            <span className="lane-sub">DeepSeek · 4 nodes · plan fixes per ISO-8000 dimension</span>
          </div>
          <div className="lane-track">
            {llmNodes.map((node, i) => {
              const idx = nodes.findIndex(n => n.id === node.id);
              return <NodeCard key={node.id} node={node} idx={idx}
                       status={progress.statuses[idx]}
                       elapsed={progress.elapsed[idx]} />;
            })}
          </div>
        </div>

        <FlowConnectors statuses={progress.statuses} />
      </div>

      <div className="audit-stream">
        <div className="audit-stream-h">
          <span>Audit stream</span>
          <span className="mono dim">{progress.log.length} events</span>
        </div>
        <div className="audit-stream-body">
          {progress.log.slice(-8).map((line, i) => (
            <div key={progress.log.length - 8 + i} className="audit-line">
              <span className="audit-line-ts mono">{String(i).padStart(2, '0')}:00</span>
              <span className="audit-line-msg mono">{line}</span>
            </div>
          ))}
          {progress.log.length === 0 && (
            <div className="audit-line dim mono">awaiting first event…</div>
          )}
        </div>
      </div>
    </div>
  );
}

function NodeCard({ node, idx, status, elapsed }) {
  const outcome = window.NODE_OUTCOMES[node.id];
  return (
    <div className={`node-card node-${node.kind} node-${status}`}>
      <div className="node-card-top">
        <div className="node-idx mono">0{idx + 1}</div>
        <NodeStatus status={status} />
      </div>
      <div className="node-card-body">
        <div className="node-label">{node.label}</div>
        <div className="node-desc">{node.desc}</div>
      </div>
      <div className="node-card-foot">
        <span className="mono node-elapsed">
          {status === 'pending' ? '—' : status === 'running' ? `${(elapsed / 1000).toFixed(1)}s…` : `${(elapsed / 1000).toFixed(1)}s`}
        </span>
        {status === 'done' && outcome && (
          <div className="node-outcome">{outcome}</div>
        )}
      </div>
    </div>
  );
}

function NodeStatus({ status }) {
  if (status === 'done') {
    return <svg className="status-icon status-done" width="14" height="14" viewBox="0 0 14 14"><path d="M3 7l3 3 5-6" stroke="currentColor" strokeWidth="1.6" fill="none" strokeLinecap="round" strokeLinejoin="round"/></svg>;
  }
  if (status === 'running') {
    return <span className="status-spinner" />;
  }
  return <span className="status-pending" />;
}

function FlowConnectors({ statuses }) {
  // Arrows from audit (idx 2) → 4 LLM agents → remediation (idx 7)
  // Pure decoration; positioning handled in CSS via the lane structure
  return (
    <svg className="flow-svg" aria-hidden="true">
      {/* connectors are pseudo via CSS gradient lines */}
    </svg>
  );
}

window.PipelineScreen = PipelineScreen;
