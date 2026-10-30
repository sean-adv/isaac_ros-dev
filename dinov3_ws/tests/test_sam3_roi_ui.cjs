// Run with node --test dinov3_ws/tests/test_sam3_roi_ui.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {regionBounds, previewRect} = require('../src/dinov3_cases/dinov3_cases/static/sam3_roi.js');

test('zero size, automatic center, explicit zero center and clipped regions', () => {
  assert.deepEqual(regionBounds([1920, 1080], {width: 0, height: 0, center_x: 0, center_y: 0}), [0, 0, 1920, 1080]);
  assert.deepEqual(regionBounds([1920, 1080], {width: 640, height: 480, center_x: -1, center_y: -1}), [640, 300, 1280, 780]);
  assert.deepEqual(regionBounds([1920, 1080], {width: 640, height: 480, center_x: 0, center_y: 0}), [0, 0, 320, 240]);
  assert.deepEqual(regionBounds([1920, 1080], {width: 4000, height: 3000, center_x: -1, center_y: -1}), [0, 0, 1920, 1080]);
});

test('4:3 native camera mapped through stretched 960x540 stream and letterboxing', () => {
  const rect = previewRect([400, 300, 1200, 900], [1600, 1200], [1000, 700], [960, 540]);
  assert.deepEqual(rect, {left: 250, top: 209.375, width: 500, height: 281.25});
});

test('pillarboxing and responsive resize keep the ROI on video content', () => {
  const rect = previewRect([480, 270, 1440, 810], [1920, 1080], [1000, 400], [960, 540]);
  assert.ok(Math.abs(rect.left - 322.2222222222) < 1e-6);
  assert.equal(rect.top, 100);
  assert.ok(Math.abs(rect.width - 355.5555555556) < 1e-6);
  assert.equal(rect.height, 200);
  assert.deepEqual(previewRect([0, 0, 1920, 1080], [1920, 1080], [480, 270], [960, 540]),
    {left: 0, top: 0, width: 480, height: 270});
});
