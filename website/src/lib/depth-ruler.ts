/** Camera-coordinate Z ruler. Its support plane is a visual guide, not a fitted table. */
export const depthTicks = Array.from({ length: 11 }, (_, i) => (i + 2) / 10);

export function rotateCloudPoint(
  x: number,
  y: number,
  z: number,
  center: number,
  yaw: number,
  pitch: number,
) {
  const xx = Math.cos(yaw) * x + Math.sin(yaw) * (z - center);
  const zz = -Math.sin(yaw) * x + Math.cos(yaw) * (z - center);
  return {
    x: xx,
    y: Math.cos(pitch) * y - Math.sin(pitch) * zz,
    d: Math.sin(pitch) * y + Math.cos(pitch) * zz,
  };
}
