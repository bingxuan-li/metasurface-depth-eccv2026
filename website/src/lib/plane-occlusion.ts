type ScreenPoint = { u: number; v: number; d: number };

/** Perspective planes interpolate reciprocal depth, not depth, in screen XY. */
export function projectedPlane(
  a: ScreenPoint,
  b: ScreenPoint,
  c: ScreenPoint,
  perspective = false,
) {
  if (perspective) {
    a = { ...a, d: 1 / a.d };
    b = { ...b, d: 1 / b.d };
    c = { ...c, d: 1 / c.d };
  }
  const ux = b.u - a.u,
    uy = b.v - a.v;
  const vx = c.u - a.u,
    vy = c.v - a.v;
  const det = ux * vy - uy * vx;
  if (Math.abs(det) < 1e-6) return null; // Edge-on: no stable visible face.
  const dx = ((b.d - a.d) * vy - (c.d - a.d) * uy) / det;
  const dy = (ux * (c.d - a.d) - vx * (b.d - a.d)) / det;
  return { dx, dy, offset: a.d - dx * a.u - dy * a.v, perspective };
}

/** Conservative footprint includes the antialiased edge of the same 2px splat. */
export function writePointDepth(
  depth: Float32Array,
  width: number,
  height: number,
  p: ScreenPoint,
) {
  for (
    let y = Math.max(0, Math.floor(p.v - 1));
    y < Math.min(height, Math.ceil(p.v + 1));
    y++
  ) {
    for (
      let x = Math.max(0, Math.floor(p.u - 1));
      x < Math.min(width, Math.ceil(p.u + 1));
      x++
    ) {
      const i = y * width + x;
      depth[i] = Math.min(depth[i], p.d);
    }
  }
}

/** Masks every guide fragment, including ruler text, against the nearest point. */
export function occludePlane(
  rgba: Uint8ClampedArray,
  depth: Float32Array,
  width: number,
  plane: ReturnType<typeof projectedPlane>,
) {
  for (let i = 0; i < depth.length; i++) {
    if (!rgba[i * 4 + 3]) continue;
    let d = plane
      ? plane.dx * ((i % width) + 0.5) +
        plane.dy * (Math.floor(i / width) + 0.5) +
        plane.offset
      : Infinity;
    if (plane?.perspective) d = d > 0 ? 1 / d : Infinity;
    if (!plane || depth[i] <= d + 1e-5) rgba[i * 4 + 3] = 0;
  }
}
