import test from 'node:test';
import assert from 'node:assert/strict';
import { artifactUrl, finiteCurves, isActive, validateImpulses, validateUpload } from './domain.ts';

test('uploads reject unsupported, oversized and unreadable media', () => {
  assert.match(validateUpload({ name: 'clip.exe', size: 10 })!, /MP4/);
  assert.match(validateUpload({ name: 'clip.mp4', size: 101 * 2 ** 20 })!, /100 MiB/);
  assert.match(validateUpload({ name: 'clip.mp4', size: 0 })!, /empty/);
  assert.match(validateUpload({ name: 'clip.mp4', size: 100 }, { duration: Infinity, width: 640, height: 480 })!, /duration/);
  assert.match(validateUpload({ name: 'clip.mp4', size: 100 }, { duration: 31, width: 640, height: 480 })!, /30 seconds/);
  assert.match(validateUpload({ name: 'clip.mp4', size: 100 }, { duration: 1, width: 3840, height: 2160 })!, /1080p/);
  assert.equal(validateUpload({ name: 'portrait.webm', size: 100 }, { duration: 5, width: 1080, height: 1920 }), null);
});

test('impulse timing is one based and bounded by the requested horizon', () => {
  assert.equal(validateImpulses([{ frame: 1, dx: .2, dy: 0 }], 10), null);
  assert.match(validateImpulses([{ frame: 0, dx: .2, dy: 0 }], 10)!, /between 1 and 10/);
  assert.match(validateImpulses([{ frame: 11, dx: .2, dy: 0 }], 10)!, /between/);
  assert.match(validateImpulses([{ frame: 1, dx: NaN, dy: 0 }], 10)!, /finite/);
});

test('result media stays on the local artifact endpoint', () => {
  assert.equal(artifactUrl('/api/artifacts/job-1/prediction.mp4'), '/api/artifacts/job-1/prediction.mp4');
  for (const url of ['https://example.com/clip.mp4', '//example.com/x', '/api/artifacts/../secret', '/api/artifacts/id/../secret']) assert.equal(artifactUrl(url), undefined);
});

test('curves drop nonfinite points instead of drawing fabricated values', () => {
  assert.deepEqual(finiteCurves([{ model: 'hybrid', horizons: [1, 2, 3], position_rmse_px: [0, NaN, 4] }]), [{ model: 'hybrid', horizons: [1, 3], position_rmse_px: [0, 4] }]);
  assert.equal(isActive({ id: 'a', mode: 'simulation', status: 'running', created_at: '' }), true);
  assert.equal(isActive({ id: 'a', mode: 'simulation', status: 'interrupted', created_at: '' }), false);
});
