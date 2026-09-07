import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  perspectivePoint,
  capturePixel,
  orbitCameraPoint,
  cameraViewport,
} from '../src/lib/perspective-camera.ts';
import { rotateCloudPoint } from '../src/lib/depth-ruler.ts';
import { projectedPlane, occludePlane } from '../src/lib/plane-occlusion.ts';

const manifest = JSON.parse(
  readFileSync(new URL('../public/results/manifest.json', import.meta.url)),
);
let maxDragDelta = 0;
let worst = 0,
  checked = 0;
for (const scene of manifest.scenes) {
  const raw = readFileSync(
    new URL(`../public/results/${scene.id}.bin`, import.meta.url),
  );
  const points = new Float32Array(
    raw.buffer,
    raw.byteOffset,
    raw.byteLength / 4,
  );
  for (let i = 0; i < points.length; i += 7) {
    const index = i / 7,
      u = (index % 200) * 8,
      v = Math.floor(index / 200) * 8;
    for (const col of [2, 3]) {
      const z = points[i + col],
        ratio = z / points[i + 2];
      const pixel = capturePixel(
        points[i] * ratio,
        points[i + 1] * ratio,
        z,
        manifest.projection,
      );
      worst = Math.max(worst, Math.abs(pixel.u - u), Math.abs(pixel.v - v));
      const x = points[i] * ratio,
        y = points[i + 1] * ratio;
      const front = cameraViewport(
        orbitCameraPoint(x, y, z, { yaw: 0, pitch: 0, distance: 0.7 }),
        manifest.projection,
      );
      const s = Math.min(920 / 1596, 520 / 1190);
      assert.ok(Math.abs(front.u - (500 + (pixel.u - 797.5) * s)) < 1e-9);
      assert.ok(Math.abs(front.v - (295 + (pixel.v - 594.5) * s)) < 1e-9);
      if (index % 35 === 0) {
        const drag = cameraViewport(
          orbitCameraPoint(x, y, z, { yaw: 1e-7, pitch: 1e-7, distance: 0.7 }),
          manifest.projection,
        );
        maxDragDelta = Math.max(
          maxDragDelta,
          Math.hypot(drag.u - front.u, drag.v - front.v),
        );
        const top = orbitCameraPoint(x, y, z, {
          yaw: 0,
          pitch: Math.PI / 2,
          distance: 18,
        });
        assert.ok(top.d > 0 && Object.values(top).every(Number.isFinite));
      }
      checked++;
    }
  }
}
assert.equal(manifest.scenes.length, 5);
assert.ok(worst < 0.001);
assert.ok(
  maxDragDelta < 0.01,
  'Front-to-drag must have no finite pose/framing jump',
);
for (const yaw of [-0.35, 0, 0.35])
  for (const pitch of [-0.25, 0.12, 0.8, Math.PI / 2]) {
    const p = (x, z) => {
      const point = perspectivePoint(
        rotateCloudPoint(x, 0.06, z, 0.7, yaw, pitch),
        1.5,
      );
      return { u: 500 + point.x * 800, v: 295 + point.y * 800, d: point.d };
    };
    const plane = projectedPlane(
      p(-0.05, 0.2),
      p(0.05, 0.2),
      p(0.05, 1.2),
      true,
    );
    assert.ok(plane);
    for (const z of [0.2, 0.5, 0.9, 1.2]) {
      const point = p(0, z);
      assert.ok(
        Math.abs(
          1 / (plane.dx * point.u + plane.dy * point.v + plane.offset) -
            point.d,
        ) < 1e-10,
      );
    }
  }
const layer = new Uint8ClampedArray(8).fill(255);
occludePlane(layer, new Float32Array([0.5, 2]), 2, {
  dx: 0,
  dy: 0,
  offset: 1,
  perspective: true,
});
assert.equal(layer[3], 0);
assert.equal(layer[7], 255);
assert.ok(
  perspectivePoint({ x: 1, y: 0, d: 1 }, 1).x >
    perspectivePoint({ x: 1, y: 0, d: 2 }, 1).x,
);
console.log({
  scenes: manifest.scenes.length,
  checked,
  maximumOriginalPixelError: worst,
  perspectivePlaneViews: 12,
  maximumInfinitesimalDragPixels: maxDragDelta,
});
