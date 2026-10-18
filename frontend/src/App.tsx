import { useCallback, useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { api } from './api';
import { artifactUrl, finiteCurves, formatBytes, formatValue, isActive, validateImpulses, validateUpload } from './domain';
import type { Curve, Impulse, Job, Mode, Status, Upload } from './domain';

const workspace = {
  simulation: { label: 'Simulation Lab', title: 'Observe. Intervene. Compare.', description: 'Compare learned trajectories with physics baselines under the same supplied impulses.', badge: 'Measured simulation' },
  forecast: { label: 'Video Forecast', title: 'Continue an observed scene.', description: 'Generate a possible continuation from an observed prefix. Physical accuracy is experimental.', badge: 'Experimental forecast' },
  backtest: { label: 'Video Backtest', title: 'Predict before seeing the answer.', description: 'Withhold the recorded future, commit a prediction, then measure the difference.', badge: 'Measured backtest' },
} as const;

function BenchMark() {
  return <svg className="bench-mark" viewBox="0 0 40 40" aria-hidden="true"><path d="M5 28L16 17M24 12l10 13M8 31h25" stroke="currentColor" strokeWidth="1.5" fill="none" /><circle cx="11" cy="22" r="4" fill="#bc3e3e" /><circle cx="24" cy="12" r="4" fill="#237851" /><circle cx="32" cy="25" r="4" fill="#285db2" /></svg>;
}

function ErrorPlot({ curves }: { curves: Curve[] }) {
  const clean = finiteCurves(curves);
  const pixel = curves.some(c => c.pixel_mse);
  if (!clean.length) return null;
  const colors = ['#1263a3', '#bc3e3e', '#237851', '#8a5314', '#6a558b'];
  const maxX = Math.max(1, ...clean.flatMap(c => c.horizons));
  const maxY = Math.max(.01, ...clean.flatMap(c => c.position_rmse_px));
  return <section className="plot-region" aria-labelledby="plot-title">
    <div className="section-heading"><h3 id="plot-title">Error over the rollout</h3><span>{pixel ? 'Pixel MSE · RGB [0,1]' : 'Position RMSE · pixels'}</span></div>
    <svg viewBox="0 0 720 200" role="img" aria-label="Prediction error versus future frame. Exact measurements are in the table below.">
      {[0, .5, 1].map(f => <g key={f}><line x1="48" y1={162 - f * 142} x2="696" y2={162 - f * 142} stroke="#dbe2e8" /><text x="38" y={166 - f * 142} textAnchor="end">{(maxY * f).toFixed(pixel ? 3 : 1)}</text></g>)}
      <line x1="48" y1="162" x2="696" y2="162" stroke="#8b9ca9" />
      {[0, .5, 1].map(f => <text key={f} x={48 + f * 648} y="184" textAnchor="middle">{Math.round(maxX * f)}</text>)}
      {clean.map((curve, i) => <polyline key={curve.model} fill="none" stroke={colors[i % colors.length]} strokeWidth="2" points={curve.horizons.map((x, j) => `${48 + x / maxX * 648},${162 - curve.position_rmse_px[j] / maxY * 142}`).join(' ')} />)}
    </svg>
    <div className="plot-key">{clean.map((c, i) => <span key={c.model}><i style={{ background: colors[i % colors.length] }} />{c.model}</span>)}</div>
  </section>;
}

function Rollouts({ job }: { job: Job }) {
  const players = useRef<(HTMLVideoElement | null)[]>([]);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState('');
  const videos = (job.result?.videos ?? []).filter(v => artifactUrl(v.url));
  const fps = job.result?.fps ?? (job.mode === 'simulation' ? 30 : 16);
  const context = job.result?.context_frames ?? (job.mode === 'simulation' ? 4 : 17);
  const boundary = context / fps;
  useEffect(() => { setTime(0); setDuration(0); setPlaying(false); setError(''); }, [job.id]);

  async function toggle() {
    if (playing) players.current.forEach(v => v?.pause());
    else {
      try { await Promise.all(players.current.filter(Boolean).map(v => v!.play())); }
      catch { players.current.forEach(v => v?.pause()); setError('Playback could not start. Download the video to open it in your media player.'); setPlaying(false); return; }
    }
    setPlaying(!playing);
  }
  function seek(value: number) {
    players.current.forEach(v => { if (v && v.readyState > 0) v.currentTime = Math.min(value, v.duration || value); });
    setTime(value);
  }
  if (!videos.length) return <div className="empty-result"><h3>No playable video was returned</h3><p>The job may contain metrics only. Download its result bundle to inspect the available artifacts.</p></div>;
  return <section aria-label="Synchronized rollout comparison">
    <div className={`rollout-grid ${videos.length === 1 ? 'single' : ''}`}>
      {videos.map((video, i) => <figure key={video.url}>
        <div className="video-stage"><video ref={el => { players.current[i] = el; }} src={artifactUrl(video.url)} preload="metadata" playsInline
          aria-label={`${video.label} rollout`} onLoadedMetadata={event => { if (i === 0) setDuration(event.currentTarget.duration); }}
          onTimeUpdate={event => { if (i === 0) { const t = event.currentTarget.currentTime; setTime(t); if (playing) players.current.slice(1).forEach(v => { if (v && Math.abs(v.currentTime - t) > .12) v.currentTime = t; }); } }}
          onEnded={() => { if (i === 0) { players.current.forEach(v => v?.pause()); setPlaying(false); } }}
          onError={() => setError('A video could not be decoded by this browser. Download the artifact to inspect it.')} /></div>
        <figcaption><span>{video.label}</span><a href={artifactUrl(video.url)} download>Download MP4</a></figcaption>
      </figure>)}
    </div>
    <div className="transport">
      <button type="button" onClick={toggle} disabled={!duration}>{playing ? 'Pause all' : 'Play all'}</button>
      <div className="ruler"><div className="ruler-labels"><span>Observed</span><span>Predicted future</span></div>
        <div className="scrub-track"><input aria-label="Rollout time" type="range" min="0" max={duration || 1} step={1 / fps} value={time} disabled={!duration} onChange={e => seek(Number(e.target.value))} />
          {duration > boundary && <span className="boundary" style={{ left: `${boundary / duration * 100}%` }} title="Observation ends; forecast begins" />}</div>
      </div>
      <output className="timecode">{time.toFixed(2)} / {duration.toFixed(2)} s</output>
    </div>
    <p className="transport-note">Forecast boundary at {boundary.toFixed(2)} s. All views share one playback cursor.</p>
    {error && <p role="alert" className="inline-error">{error}</p>}
  </section>;
}

function Results({ job }: { job: Job }) {
  const result = job.result;
  return <>
    <div className="result-heading"><div><span className={`evidence-label ${job.mode === 'forecast' ? 'experimental' : ''}`}>{workspace[job.mode].badge}</span><h2>{workspace[job.mode].label} result</h2></div>
      <a className="button" href={`/api/jobs/${encodeURIComponent(job.id)}/export`}>Download results</a></div>
    <div className="provenance"><span>Model <strong>{result?.model_version ?? 'Not reported'}</strong></span><span>Execution <strong>{result?.execution_origin ?? 'Not reported'}</strong></span><span className="job-id">{job.id}</span></div>
    <Rollouts key={job.id} job={job} />
    {job.mode === 'forecast' ? <p className="evidence-note">This is a generated continuation. No recorded future was supplied for scoring, so no accuracy or calibrated confidence is shown.</p> : <>
      <ErrorPlot curves={result?.curves ?? []} />
      {!!result?.metrics?.length && <section className="metrics-region" aria-labelledby="metrics-title"><div className="section-heading"><h3 id="metrics-title">Measured errors</h3><span>Lower is better</span></div>
        <div className="table-scroll"><table><thead><tr><th scope="col">Predictor</th><th scope="col">Future frame</th><th scope="col">Position RMSE (px)</th><th scope="col">Velocity RMSE</th><th scope="col">Pixel MSE</th></tr></thead>
          <tbody>{result.metrics.map((m, i) => <tr key={`${m.model}-${m.horizon}-${i}`}><th scope="row">{m.model}</th><td>{m.horizon}</td><td>{formatValue(m.position_rmse_px, 2)}</td><td>{formatValue(m.velocity_rmse)}</td><td>{formatValue(m.pixel_mse, 5)}</td></tr>)}</tbody></table></div>
        <p className="transport-note">A dash means the measurement is unavailable. Pixel error does not establish physical correctness.</p></section>}
      {job.mode === 'simulation' && !!result?.uncertainty?.length && <section className="metrics-region"><div className="section-heading"><h3>Calibrated hybrid uncertainty</h3><span>Conditional on shared perception</span></div>
        <div className="table-scroll"><table><thead><tr><th scope="col">Horizon</th><th scope="col">Coverage</th><th scope="col">Mean object radius (px)</th><th scope="col">Mean diameter (px)</th></tr></thead><tbody>{result.uncertainty.map(u => <tr key={u.horizon}><th scope="row">{u.horizon}</th><td>{u.coverage === undefined ? '—' : `${formatValue(u.coverage * 100, 1)}%`}</td><td>{formatValue(u.radius_px, 2)}</td><td>{formatValue(u.interval_width_px, 2)}</td></tr>)}</tbody></table></div><p className="transport-note">{result.uncertainty_note}</p></section>}
    </>}
  </>;
}

export default function App() {
  const [mode, setMode] = useState<Mode>('simulation');
  const [status, setStatus] = useState<Status | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [bundle, setBundle] = useState('');
  const [seed, setSeed] = useState(0);
  const [horizon, setHorizon] = useState(50);
  const [drag, setDrag] = useState(.2);
  const [restitution, setRestitution] = useState(.9);
  const [actions, setActions] = useState<Impulse[]>([]);
  const [execution, setExecution] = useState<'local' | 'notebook'>('local');
  const [file, setFile] = useState<File | null>(null);
  const [clipUrl, setClipUrl] = useState('');
  const [clipDuration, setClipDuration] = useState(0);
  const [cutoff, setCutoff] = useState(1.0625);
  const [upload, setUpload] = useState<Upload | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [connectionError, setConnectionError] = useState('');
  const [deleting, setDeleting] = useState<string | null>(null);
  const selected = jobs.find(j => j.id === selectedId);
  const hasActive = jobs.some(isActive) || Boolean(status?.worker.busy_job_id);
  const bundles = status?.bundles.filter(b => mode === 'simulation' ? b.kind !== 'video' : b.kind === 'video') ?? [];
  const currentBundle = bundles.some(b => b.id === bundle) ? bundle : bundles[0]?.id ?? '';

  const refresh = useCallback(async () => {
    try {
      const [nextStatus, nextJobs] = await Promise.all([api<Status>('/status'), api<Job[] | { jobs: Job[] }>('/jobs')]);
      setStatus(nextStatus); setJobs(Array.isArray(nextJobs) ? nextJobs : nextJobs.jobs); setConnectionError('');
    } catch (e) { setConnectionError(e instanceof Error ? e.message : 'The local service could not be reached.'); }
  }, []);
  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => { if (hasActive) { const timer = setInterval(() => void refresh(), 2000); return () => clearInterval(timer); } }, [hasActive, refresh]);
  useEffect(() => { if (!file) { setClipUrl(''); return; } const url = URL.createObjectURL(file); setClipUrl(url); return () => URL.revokeObjectURL(url); }, [file]);

  function switchMode(next: Mode) { setMode(next); setError(''); setSelectedId(null); }
  function updateAction(index: number, key: keyof Impulse, value: number) { setActions(previous => previous.map((a, i) => i === index ? { ...a, [key]: value } : a)); }
  function selectFile(next: File | undefined) {
    setUpload(null); setClipDuration(0); setError('');
    if (!next) { setFile(null); return; }
    const invalid = validateUpload(next);
    if (invalid) { setError(invalid); setFile(null); return; }
    setFile(next);
  }
  async function submit(event: FormEvent) {
    event.preventDefault(); setError('');
    if (mode === 'simulation') {
      const invalid = validateImpulses(actions, horizon);
      if (invalid) { setError(invalid); return; }
      if (!currentBundle) { setError('Export a compatible simulation serving bundle before running a comparison.'); return; }
    } else {
      if (!file || !clipDuration) { setError('Choose a readable video before creating a job.'); return; }
      if (cutoff <= 0 || cutoff >= clipDuration) { setError('Set the observation cut-off inside the video, leaving a recorded future.'); return; }
      if (execution === 'local' && !status?.video.available) { setError(status?.video.reason ?? 'Local video generation is unavailable. Choose notebook export.'); return; }
    }
    setBusy(true);
    try {
      let media = upload;
      if (mode !== 'simulation' && !media && file) {
        const body = new FormData(); body.append('file', file); media = await api<Upload>('/uploads', { method: 'POST', body }); setUpload(media);
      }
      const payload = mode === 'simulation'
        ? { mode, bundle_id: currentBundle, seed, horizon, actions, drag, restitution, execution: 'local' }
        : { mode, ...(currentBundle ? { bundle_id: currentBundle } : {}), seed, horizon, upload_id: media?.id, cutoff_seconds: cutoff, execution };
      const job = await api<Job>('/jobs', { method: 'POST', body: JSON.stringify(payload) });
      setSelectedId(job.id); setJobs(previous => [job, ...previous.filter(j => j.id !== job.id)]); await refresh();
    } catch (e) { setError(e instanceof Error ? e.message : 'The job could not be created.'); }
    finally { setBusy(false); }
  }
  async function cancel(job: Job) {
    try { await api(`/jobs/${encodeURIComponent(job.id)}/cancel`, { method: 'POST' }); await refresh(); }
    catch (e) { setError(e instanceof Error ? e.message : 'The job could not be cancelled.'); }
  }
  async function remove(job: Job) {
    try { await api(`/jobs/${encodeURIComponent(job.id)}`, { method: 'DELETE' }); setDeleting(null); if (selectedId === job.id) setSelectedId(null); await refresh(); }
    catch (e) { setError(e instanceof Error ? e.message : 'The result could not be deleted.'); }
  }
  async function importResult(job: Job, resultFile: File | undefined) {
    if (!resultFile) return;
    setBusy(true); setError('');
    try { const body = new FormData(); body.append('file', resultFile); await api(`/jobs/${encodeURIComponent(job.id)}/import`, { method: 'POST', body }); await refresh(); }
    catch (e) { setError(e instanceof Error ? e.message : 'The notebook result could not be imported.'); }
    finally { setBusy(false); }
  }

  return <div className="application">
    <a className="skip-link" href="#workspace">Skip to workspace</a>
    <header className="app-header"><div className="product"><BenchMark /><div><h1>Neural World Model</h1><p>Local laboratory</p></div></div>
      <div className="system-readings"><span className={`service-state ${connectionError ? 'unavailable' : ''}`}><i />{connectionError ? 'Service disconnected' : status ? `Worker ${status.worker.state}` : 'Connecting to local service'}</span>
        {status && <span className="storage-reading">{formatBytes(status.storage.used_bytes)} / {formatBytes(status.storage.limit_bytes)} managed · {formatBytes(status.storage.free_bytes)} free</span>}
        <button type="button" className="quiet" onClick={() => void refresh()}>Refresh</button></div>
    </header>
    <nav className="workspace-nav" aria-label="Workspaces">{(Object.keys(workspace) as Mode[]).map(key => <button key={key} type="button" aria-pressed={mode === key} onClick={() => switchMode(key)}>{workspace[key].label}</button>)}<span>Runs stay on this computer</span></nav>
    {connectionError && <div className="connection-banner" role="alert"><strong>The local service is unavailable.</strong> {connectionError} Start the application with <code>run.bat app</code>, then refresh.</div>}
    <main id="workspace" className="workbench">
      <aside className="inspector" aria-label="Experiment controls"><form onSubmit={submit}>
        <div className="inspector-heading"><h2>{mode === 'simulation' ? 'Scenario' : 'Source video'}</h2><span className="evidence-label small">{mode === 'forecast' ? 'Experimental' : 'Measured'}</span></div>
        {mode === 'simulation' ? <>
          <p className="field-help">Three identified discs. Four observed frames. The same future actions for every predictor.</p>
          <label>Serving model<select value={currentBundle} onChange={e => setBundle(e.target.value)} disabled={!bundles.length}>{!bundles.length && <option value="">No compatible bundle</option>}{bundles.map(b => <option key={b.id} value={b.id}>{b.label}</option>)}</select></label>
          <label>Scenario seed<input type="number" min="0" max="2147483647" step="1" required value={seed} onChange={e => setSeed(Number(e.target.value))} /></label>
          <fieldset><legend>Ground-truth physics</legend><div className="paired-fields"><label>Drag (s⁻¹)<input type="number" min="0" max="1" step=".05" value={drag} onChange={e => setDrag(Number(e.target.value))} required /></label><label>Restitution<input type="number" min="0" max="1" step=".05" value={restitution} onChange={e => setRestitution(Number(e.target.value))} required /></label></div>
            <p className="field-help">{drag === .2 && restitution === .9 ? 'Training conditions: drag 0.2, restitution 0.9.' : 'Changed physics. Calibration may not transfer to this condition.'}</p></fieldset>
        </> : <>
          <p className="field-help">MP4 or WebM · up to 100 MiB, 30 s, 1080p. Uploaded footage stays local unless you export it.</p>
          <label className="file-control">Choose video<input type="file" accept=".mp4,.webm,video/mp4,video/webm" onChange={e => selectFile(e.target.files?.[0])} /></label>
          {file && <p className="filename">{file.name}</p>}
          {clipUrl && <div className="prefix-preview"><video src={clipUrl} controls preload="metadata" aria-label="Observed video prefix preview"
            onLoadedMetadata={e => { const v = e.currentTarget; const invalid = validateUpload(file!, { duration: v.duration, width: v.videoWidth, height: v.videoHeight }); if (invalid) { setError(invalid); setFile(null); } else { setClipDuration(v.duration); setCutoff(Math.min(1.0625, v.duration / 2)); } }}
            onError={() => { setError('The browser could not read this video. Try a supported MP4 or WebM encoding.'); setClipDuration(0); }}
            onTimeUpdate={e => { if (e.currentTarget.currentTime > cutoff) { e.currentTarget.pause(); e.currentTarget.currentTime = cutoff; } }} /><span>Observed prefix preview</span></div>}
          <label>Observation cut-off (seconds)<input type="number" min="1" max={clipDuration > 0 ? Math.max(1, clipDuration - .01) : 30} step=".0625" required value={cutoff} onChange={e => setCutoff(Number(e.target.value))} /><span className="field-help">17 samples ending at this point enter the predictor.</span></label>
          <label>Execution<select value={execution} onChange={e => setExecution(e.target.value as 'local' | 'notebook')}><option value="local">This computer</option><option value="notebook">Export notebook job</option></select></label>
          {execution === 'local' && !status?.video.available && <p className="availability-note">{status?.video.reason ?? 'Checking video model availability.'}</p>}
          {execution === 'notebook' && <p className="field-help">Download a job package, run it in your chosen notebook, then import its result. No cloud upload happens automatically.</p>}
        </>}
        <fieldset><legend>Forecast horizon</legend><div className="horizon-options">{[10, 20, 50].map(value => <label key={value}><input type="radio" name="horizon" value={value} checked={horizon === value} onChange={() => setHorizon(value)} /><span>{value} frames</span></label>)}</div></fieldset>
        {mode === 'simulation' ? <fieldset className="impulses"><legend><span className="ball-dot red" />Red-ball impulses</legend><p className="field-help">Frame 1 acts immediately after the observed context. No row means zero impulse.</p>
          {actions.length > 0 && <div className="impulse-labels" aria-hidden="true"><span>Frame</span><span>Δv x</span><span>Δv y</span><span /></div>}
          {actions.map((action, i) => <div className="impulse-row" key={i}><input aria-label={`Impulse ${i + 1} frame`} type="number" min="1" max={horizon} step="1" required value={action.frame} onChange={e => updateAction(i, 'frame', Number(e.target.value))} /><input aria-label={`Impulse ${i + 1} horizontal velocity change`} type="number" step=".05" required value={action.dx} onChange={e => updateAction(i, 'dx', Number(e.target.value))} /><input aria-label={`Impulse ${i + 1} vertical velocity change`} type="number" step=".05" required value={action.dy} onChange={e => updateAction(i, 'dy', Number(e.target.value))} /><button type="button" className="remove-impulse" aria-label={`Remove impulse ${i + 1}`} onClick={() => setActions(previous => previous.filter((_, j) => j !== i))}>×</button></div>)}
          <button type="button" className="text-button" onClick={() => setActions(previous => [...previous, { frame: 1, dx: .2, dy: 0 }])}>+ Add impulse</button></fieldset> : <label>Generation seed<input type="number" min="0" max="2147483647" step="1" required value={seed} onChange={e => setSeed(Number(e.target.value))} /></label>}
        <button className="primary run-button" type="submit" disabled={busy || !status || (mode === 'simulation' && !currentBundle)}>{busy ? 'Preparing job…' : mode === 'simulation' ? 'Run comparison' : execution === 'notebook' ? 'Prepare notebook job' : mode === 'forecast' ? 'Generate forecast' : 'Run backtest'}</button>
        <p className="field-help queue-note">One GPU job runs at a time. Up to four jobs can wait.</p>
      </form></aside>
      <section className="experiment-view" aria-label="Experiment results">
        {error && <div className="error-banner" role="alert"><div><strong>Action needs attention</strong><p>{error}</p></div><button type="button" aria-label="Dismiss error" onClick={() => setError('')}>×</button></div>}
        {selected ? selected.result ? <Results job={selected} /> : <div className="pending-result"><span className="evidence-label">{workspace[selected.mode].badge}</span><h2>{workspace[selected.mode].label}</h2><p className={`job-state state-${selected.status}`}>{selected.status}</p><p>{selected.message ?? (selected.status === 'queued' ? 'Waiting for the worker. This job will start when the current job releases the GPU.' : 'The result will appear here when the worker finishes.')}</p>
          {selected.error && <p className="inline-error" role="alert">{selected.error}</p>}
          {(isActive(selected) || selected.status === 'waiting_notebook') && <><progress max="1" value={selected.progress ?? 0} aria-label="Job progress" /><button type="button" onClick={() => void cancel(selected)}>Cancel job</button></>}
          {selected.status === 'waiting_notebook' && <div className="notebook-actions"><a className="button" href={`/api/jobs/${encodeURIComponent(selected.id)}/export`}>Download notebook job</a><a href="/api/video-notebook" download>Download runner notebook</a><a href="/api/source-bundle" download>Download source bundle</a><label className="file-control">Import completed result<input type="file" accept=".zip" disabled={busy} onChange={e => void importResult(selected, e.target.files?.[0])} /></label><p className="field-help">Open the runner locally or in Colab, follow its source-bundle setup, then run this job and import the result ZIP. Runtime and billing are chosen manually.</p></div>}
        </div> : <div className="ready-view"><div className="view-heading"><span className={`evidence-label ${mode === 'forecast' ? 'experimental' : ''}`}>{workspace[mode].badge}</span><h2>{workspace[mode].title}</h2><p>{workspace[mode].description}</p></div>
          <div className="empty-bench"><div className="bench-crosshair" aria-hidden="true"><span /><span /></div><h3>{mode === 'simulation' ? 'Set up a physical experiment' : 'Choose an observed video'}</h3><p>{mode === 'simulation' ? 'Choose a seed and horizon, add an impulse if needed, then run a comparison. Actual rollout videos and measurements will appear here.' : mode === 'forecast' ? 'Select footage and mark where observation ends. The model generates a possible future from that prefix.' : 'Select footage and mark the observation cut-off. The withheld future is used only after the prediction is saved.'}</p></div>
          <div className="method-strip">{mode === 'simulation' ? <><span><b>4</b> observed frames</span><span><b>10 / 20 / 50</b> future frames</span><span><i className="ball-dot red" /><i className="ball-dot green" /><i className="ball-dot blue" />Fixed object identities</span></> : <><span>Observed prefix only</span><span>{mode === 'forecast' ? 'No accuracy score without a recorded future' : 'Prediction committed before scoring'}</span></>}</div>
        </div>}
      </section>
    </main>
    <section className="ledger" aria-labelledby="ledger-title"><div className="section-heading"><div><h2 id="ledger-title">Job ledger</h2><span>{jobs.length} saved {jobs.length === 1 ? 'job' : 'jobs'}</span></div><span>{hasActive ? 'Updating every 2 seconds' : 'Queue idle'}</span></div>
      {jobs.length === 0 ? <p className="empty-ledger">No experiments yet. Configure a scenario or choose a video to create the first job.</p> : <div className="table-scroll"><table><thead><tr><th scope="col">Created</th><th scope="col">Workspace</th><th scope="col">Job</th><th scope="col">State</th><th scope="col">Progress</th><th scope="col">Actions</th></tr></thead><tbody>{jobs.map(job => <tr key={job.id} className={selectedId === job.id ? 'selected-row' : ''}><td>{new Date(job.created_at).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</td><th scope="row"><button className="text-button" onClick={() => { setMode(job.mode); setSelectedId(job.id); setError(''); }}>{workspace[job.mode]?.label ?? job.mode}</button></th><td className="job-id">{job.id.slice(0, 12)}</td><td><span className={`job-state state-${job.status}`}>{job.status.replaceAll('_', ' ')}</span></td><td>{isActive(job) ? <progress max="1" value={job.progress ?? 0} aria-label={`${job.id} progress`} /> : job.message ?? '—'}</td><td className="job-actions">{(isActive(job) || job.status === 'waiting_notebook') ? <button onClick={() => void cancel(job)}>Cancel</button> : deleting === job.id ? <><span>Delete files?</span><button className="danger" onClick={() => void remove(job)}>Delete</button><button onClick={() => setDeleting(null)}>Keep</button></> : <>{job.status === 'succeeded' && <a href={`/api/jobs/${encodeURIComponent(job.id)}/export`}>Export</a>}{['failed', 'interrupted', 'cancelled'].includes(job.status) && <button onClick={async () => { try { const next = await api<Job>(`/jobs/${encodeURIComponent(job.id)}/retry`, {method: 'POST'}); setSelectedId(next.id); await refresh(); } catch (e) { setError(e instanceof Error ? e.message : 'Retry failed'); } }}>Retry</button>}<button onClick={() => setDeleting(job.id)}>Delete</button></>}</td></tr>)}</tbody></table></div>}
    </section>
    <footer>Local production beta <span>Video generation is experimental. Inspect measured results before drawing conclusions.</span></footer>
  </div>;
}
