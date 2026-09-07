/** Positive camera-space depth; callers must place the camera outside the scene. */
export function perspectivePoint(
  p: { x: number; y: number; d: number },
  distance: number,
) {
  const d = distance + p.d;
  if (d <= 0) throw new Error('Point behind perspective camera');
  return { x: p.x / d, y: p.y / d, d };
}

export function capturePixel(
  x: number,
  y: number,
  z: number,
  k: { fx: number; fy: number; cx: number; cy: number },
) {
  return { u: (k.fx * x) / z + k.cx, v: (k.fy * y) / z + k.cy };
}

export const orbitTargetDepth = 0.7;

/** One capture-anchored camera. At zero angles/dolly its pose is exactly identity. */
export function orbitCameraPoint(
  x: number,
  y: number,
  z: number,
  view: { yaw: number; pitch: number; distance: number },
) {
  const zz = z - orbitTargetDepth;
  const rx = Math.cos(view.yaw) * x + Math.sin(view.yaw) * zz;
  const rz = -Math.sin(view.yaw) * x + Math.cos(view.yaw) * zz;
  const ry = Math.cos(view.pitch) * y - Math.sin(view.pitch) * rz;
  const d =
    Math.sin(view.pitch) * y + Math.cos(view.pitch) * rz + view.distance;
  return { x: rx / d, y: ry / d, d };
}

/** Fixed intrinsics and viewport mapping: never depends on scene bounds or angles. */
export function cameraViewport(
  p: { x: number; y: number },
  k: { fx: number; fy: number; cx: number; cy: number },
  width = 1000,
  height = 590,
) {
  const s = Math.min((width - 80) / 1596, (height - 70) / 1190);
  return {
    u: width / 2 + (k.fx * p.x + k.cx - 797.5) * s,
    v: height / 2 + (k.fy * p.y + k.cy - 594.5) * s,
  };
}
