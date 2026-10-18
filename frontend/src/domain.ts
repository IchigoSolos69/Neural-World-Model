export type Mode = 'simulation' | 'forecast' | 'backtest';
export type Impulse = { frame: number; dx: number; dy: number };
export type Bundle = { id: string; label: string; kind: string; default_model?: string };
export type Metric = { model: string; horizon: number; position_rmse_px?: number; velocity_rmse?: number; pixel_mse?: number };
export type Curve = { model: string; horizons: number[]; position_rmse_px?: number[]; pixel_mse?: number[] };
export type Result = {
  videos?: { label: string; url: string }[];
  metrics?: Metric[];
  curves?: Curve[];
  uncertainty?: { horizon: number; coverage?: number; radius_px?: number; interval_width_px?: number }[];
  model_version?: string;
  execution_origin?: string;
  fps?: number;
  context_frames?: number;
  uncertainty_note?: string;
};
export type Job = {
  id: string; mode: Mode; status: string; progress?: number; message?: string; error?: string;
  result?: Result; created_at: string;
};
export type Status = {
  storage: { used_bytes: number; limit_bytes: number; free_bytes: number };
  bundles: Bundle[];
  worker: { state: string; busy_job_id?: string | null };
  video: { available: boolean; reason?: string };
  jobs?: Job[];
};
export type Upload = { id: string; duration?: number; width?: number; height?: number; fps?: number };

export const isActive = (job: Job) => ['queued', 'running', 'cancelling', 'pending'].includes(job.status);
export const formatBytes = (value: number) => `${(value / 2 ** 30).toFixed(1)} GiB`;
export const formatValue = (value: number | undefined, digits = 3) =>
  typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';

export function validateUpload(file: { name: string; size: number }, metadata?: { duration: number; width: number; height: number }) {
  if (!/\.(mp4|webm)$/i.test(file.name)) return 'Choose an MP4 or WebM video.';
  if (file.size > 100 * 2 ** 20) return 'The video exceeds the 100 MiB upload limit.';
  if (file.size === 0) return 'The selected file is empty.';
  if (metadata) {
    if (!Number.isFinite(metadata.duration) || metadata.duration <= 0) return 'The video has no readable duration.';
    if (metadata.duration > 30) return 'Choose a video no longer than 30 seconds.';
    if (Math.max(metadata.width, metadata.height) > 1920 || Math.min(metadata.width, metadata.height) > 1080) return 'Choose a video at 1080p or below.';
  }
  return null;
}

export function validateImpulses(actions: Impulse[], horizon: number) {
  if (actions.some(a => !Number.isInteger(a.frame) || a.frame < 1 || a.frame > horizon)) return `Impulse frames must be between 1 and ${horizon}.`;
  if (actions.some(a => !Number.isFinite(a.dx) || !Number.isFinite(a.dy))) return 'Each impulse needs a finite horizontal and vertical value.';
  if (new Set(actions.map(a => a.frame)).size !== actions.length) return 'Use only one impulse per future frame.';
  if (actions.some(a => Math.abs(a.dx) > 1 || Math.abs(a.dy) > 1)) return 'Impulse values must be between -1 and 1.';
  return null;
}

export function artifactUrl(url: string): string | undefined {
  return /^\/api\/artifacts\/[a-zA-Z0-9_-]+\/[a-zA-Z0-9_.-]+$/.test(url) ? url : undefined;
}

export function finiteCurves(curves: Curve[] = []) {
  return curves.filter(c => Array.isArray(c.horizons) && (Array.isArray(c.position_rmse_px) || Array.isArray(c.pixel_mse)))
    .map(c => {
      const source = c.position_rmse_px ?? c.pixel_mse ?? [];
      const points = c.horizons.map((x, i) => [x, source[i]])
        .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y) && x >= 0 && y >= 0);
      return { model: c.model, horizons: points.map(p => p[0]), position_rmse_px: points.map(p => p[1]) };
    }).filter(c => c.horizons.length > 0);
}
