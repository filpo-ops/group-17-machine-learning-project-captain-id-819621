// Pipeline execution screen — flowing horizontal layout
// Visualizes the 10-node pipeline as 3 groups: head (det) → middle (llm) → tail (det)

const { useState: useState2 } = React;

function PipelineScreen({ progress, onSkip }) {
  const nodes = window.PIPELINE_NODES;
  const llmStart = nodes.findIndex(n => n.kind === 'llm');
  const llmEnd   = nodes.length - 1 - [...nodes].reverse().findIndex(n => n.kind === 'llm');

  const head    = nodes.slice(0, llmStart);            // det before LLM
  const middle  = nodes.slice(llmStart, llmEnd + 1);   // LLM agents
  const tail    = nodes.slice(llmEnd + 1);             // det after LLM

  const totalElapsed   = progress.elapsed.reduce((a, b) => a + b, 0);
  const completedCount = progress.statuses.filter(s => s === 'done').length;
  const pct            = (completedCount / nodes.length) * 100;

  const renderGroup = (group, kindClass) => (
    <div className={`flow-group ${kindClass}`}>
      <div className="flow-group-head">
        <span className={`lane-tag ${kindClass === 'group-llm' ? 'lane-tag-llm' : ''}`}>
          {kindClass === 'group-llm' ? 'LLM AGENTS' : 'DETERMINISTIC'}
        </span>
        <span className="lane-sub">
          {kindClass === 'group-llm'
            ? `${group.length} agents · plan fixes per ISO-8000 dimension`
            : `${group.length} ${group.length === 1 ? 'step' : 'steps'} · rule-based, reproducible`}
        </span>
      </div>
      <div className="flow-group-track">
        {group.map(node => {
          const idx = nodes.findIndex(n => n.id === node.id);
          return <NodeCard key={node.id} node={node} idx={idx}
                   status={progress.statuses[idx]}
                   elapsed={progress.elapsed[idx]} />;
        })}
      </div>
    </div>
  );

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
        <div className="flow-row">
          {renderGroup(head,   'group-det')}
          <div className="flow-arrow"><FlowArrow /></div>
          {renderGroup(middle, 'group-llm')}
          <div className="flow-arrow"><FlowArrow /></div>
          {renderGroup(tail,   'group-det')}
        </div>
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

function FlowArrow() {
  return (
    <svg width="22" height="14" viewBox="0 0 22 14" fill="none" aria-hidden="true">
      <path d="M1 7h18M14 1l6 6-6 6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
}

function NodeCard({ node, idx, status, elapsed }) {
  const outcome = window.NODE_OUTCOMES[node.id];
  return (
    <div className={`node-card node-${node.kind} node-${status}`}>
      <div className="node-card-top">
        <div className="node-idx mono">{String(idx + 1).padStart(2, '0')}</div>
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

window.PipelineScreen = PipelineScreen;
