import { useEffect, useRef, useState } from 'react';
import Image from './components/image';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { depthTicks } from './lib/depth-ruler';
import { orbitCameraPoint, cameraViewport } from './lib/perspective-camera';
import { projectedPlane, writePointDepth, occludePlane } from './lib/plane-occlusion';
import realManifest from '../public/results/manifest.json';

const palette = [
  [165, 0, 38],
  [244, 109, 67],
  [254, 224, 139],
  [230, 245, 152],
  [102, 194, 165],
  [50, 136, 189],
  [94, 79, 162],
];
function color(z: number) {
  const t = Math.max(0, Math.min(1, z - 0.2)) * 6;
  const lo = Math.min(5, Math.floor(t));
  return `rgb(${palette[lo].map((v, j) => Math.round(v * (1 - t + lo) + palette[lo + 1][j] * (t - lo))).join(',')})`;
}

export function PublicResultsGallery() {
  const visibleScenes = realManifest.scenes;
  const [activeScene, setScene] = useState(visibleScenes[0].id);
  const current =
    visibleScenes.find((s) => s.id === activeScene) ?? visibleScenes[0];
  const scene = current.id;
  const [cloud, setCloud] = useState<{
    id: string;
    points: Float32Array;
  } | null>(null);
  const [error, setError] = useState('');
  const [view, setView] = useState({ yaw: 0, pitch: 0, distance: 0.7 });
  const [source, setSource] = useState('prediction');
  const [shading, setShading] = useState('input');
  const [showRuler, setShowRuler] = useState(true);
  const canvas = useRef<HTMLCanvasElement>(null);
  const drag = useRef<{ x: number; y: number } | null>(null);
  const ready = cloud?.id === scene;

  useEffect(() => {
    const abort = new AbortController();
    fetch(`./results/${scene}.bin?format=model-input-v1`, {
      signal: abort.signal,
    })
      .then((r) => {
        if (!r.ok) throw new Error('unavailable');
        return r.arrayBuffer();
      })
      .then((buffer) => {
        const points = new Float32Array(buffer);
        if (
          points.length !== current.points * 7 ||
          !points.every(Number.isFinite)
        )
          throw new Error('invalid');
        if (!canvas.current?.getContext('2d')) throw new Error('unsupported');
        setCloud({ id: scene, points });
      })
      .catch((e) => {
        if (e.name !== 'AbortError')
          setError(
            '3D data could not load. Select another scene or reload the page.',
          );
      });
    return () => abort.abort();
  }, [scene, current.points]);

  useEffect(() => {
    const element = canvas.current;
    if (!element || !cloud || cloud.id !== scene) return;
    const ctx = element.getContext('2d');
    if (!ctx) return;
    const { points: p } = cloud;
    const width = 1000,
      height = 590;
    ctx.fillStyle = '#f4f5f6';
    ctx.fillRect(0, 0, width, height);
    const geometry = [];
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    for (let i = 0; i < p.length; i += 7) {
      const z = source === 'gt' ? p[i + 3] : p[i + 2];
      const ratio = z / p[i + 2];
      const x = p[i] * ratio,
        y = p[i + 1] * ratio;
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
      geometry.push({ x, y, z, r: p[i + 4], g: p[i + 5], b: p[i + 6] });
    }
    const pointColor = (p: { z: number; r: number; g: number; b: number }) =>
      shading === 'depth'
        ? color(p.z)
        : `rgb(${Math.round(p.r * 255)} ${Math.round(p.g * 255)} ${Math.round(p.b * 255)})`;
    // One pose path for Front, drag and Top. Neither intrinsics nor framing changes on drag.
    const rotatePoint = (x: number, y: number, z: number) =>
      orbitCameraPoint(x, y, z, view);
    const rulerX = (minX + maxX) / 2;
    const rulerY = maxY + 0.012;
    const rulerHalfWidth = 0.008;
    // A flat XZ rectangle: every top-surface vertex has the same camera Y.
    const rulerCorners = [
      [rulerX - rulerHalfWidth, 0.19],
      [rulerX + rulerHalfWidth, 0.19],
      [rulerX + rulerHalfWidth, 1.21],
      [rulerX - rulerHalfWidth, 1.21],
    ].map(([x, z]) => rotatePoint(x, rulerY, z));
    const ruler = depthTicks.map((z) => ({
      ...rotatePoint(rulerX, rulerY, z),
      z,
    }));
    const rotated = geometry.map((p) => ({
      ...rotatePoint(p.x, p.y, p.z),
      fill: pointColor(p),
    }));
    const screen = (p: { x: number; y: number }) =>
      cameraViewport(p, realManifest.projection, width, height);
    const projected = rotated
      .filter((p) => p.d > 0.001)
      .map((p) => {
        return {
          ...screen(p),
          d: p.d,
          fill: p.fill,
        };
      })
      .sort((a, b) => b.d - a.d);
    const pointDepth = new Float32Array(width * height).fill(Infinity);
    for (const p of projected) {
      if (p.u < 0 || p.u > width || p.v < 0 || p.v > height) continue;
      ctx.fillStyle = p.fill;
      ctx.fillRect(p.u - 1, p.v - 1, 2, 2);
      writePointDepth(pointDepth, width, height, p);
    }
    const mainContext = ctx;
    {
      const guide = document.createElement('canvas');
      guide.width = width;
      guide.height = height;
      const ctx = guide.getContext('2d');
      if (!ctx) return;
      const onGround = (x: number, z: number) =>
        screen(rotatePoint(x, rulerY, z));
      // Sparse, unlabelled lateral divisions: nominal XY, not calibrated distances.
      const gridLeft = minX - 0.01,
        gridRight = maxX + 0.01;
      ctx.strokeStyle = '#d8dde3';
      ctx.lineWidth = 0.8;
      const gridLine = (x0: number, z0: number, x1: number, z1: number) => {
        const a = onGround(x0, z0),
          b = onGround(x1, z1);
        ctx.beginPath();
        ctx.moveTo(a.u, a.v);
        ctx.lineTo(b.u, b.v);
        ctx.stroke();
      };
      for (const z of depthTicks) gridLine(gridLeft, z, gridRight, z);
      for (let i = 0; i <= 6; i++) {
        const x = gridLeft + ((gridRight - gridLeft) * i) / 6;
        gridLine(x, 0.2, x, 1.2);
      }
      const marks = ruler.map((p) => ({ ...screen(p), z: p.z }));
      const length = Math.hypot(
        marks[10].u - marks[0].u,
        marks[10].v - marks[0].v,
      );
      if (showRuler) {
        // Flat ruler on the horizontal reference plane, not a screen-facing ribbon.
        const face = rulerCorners.map(screen);
        ctx.beginPath();
        face.forEach((p, i) =>
          i === 0 ? ctx.moveTo(p.u, p.v) : ctx.lineTo(p.u, p.v),
        );
        ctx.closePath();
        ctx.fillStyle = '#e5e8eb';
        ctx.fill();
        ctx.strokeStyle = '#727b85';
        ctx.lineWidth = 1.2;
        ctx.stroke();
        ctx.strokeStyle = '#58616d';
        ctx.fillStyle = '#39424e';
        ctx.lineWidth = 1.2;
        ctx.font = '14px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const onRuler = (x: number, z: number) =>
          screen(rotatePoint(x, rulerY, z));
        // Minor divisions every 2 cm, major divisions every 10 cm, all coplanar.
        for (let i = 0; i <= 50; i++) {
          const z = (10 + i) / 50;
          const major = i % 5 === 0;
          const a = onRuler(rulerX - rulerHalfWidth, z);
          const b = onRuler(
            rulerX - rulerHalfWidth + (major ? 0.006 : 0.003),
            z,
          );
          ctx.beginPath();
          ctx.moveTo(a.u, a.v);
          ctx.lineTo(b.u, b.v);
          ctx.stroke();
        }
        if (length < 100) {
          // Keep status text out of the geometry layer and its depth test.
        } else {
          const labelEvery = length >= 520 ? 1 : length >= 270 ? 2 : 5;
          marks.forEach((p, i) => {
            const major = i % labelEvery === 0 || i === 10;
            const label = onRuler(rulerX + 0.003, p.z);
            if (major) ctx.fillText(p.z.toFixed(1), label.u, label.v);
          });
        }
      }
      const planePoints = rulerCorners
        .slice(0, 3)
        .map((p) => ({ ...screen(p), d: p.d }));
      const plane = projectedPlane(
        planePoints[0],
        planePoints[1],
        planePoints[2],
        true,
      );
      const pixels = ctx.getImageData(0, 0, width, height);
      occludePlane(pixels.data, pointDepth, width, plane);
      ctx.putImageData(pixels, 0, 0);
      mainContext.drawImage(guide, 0, 0);
      mainContext.font = '14px system-ui, sans-serif';
      mainContext.textAlign = 'center';
      mainContext.fillStyle = '#39424e';
      mainContext.fillText(
        !showRuler
          ? 'Reference grid · perspective'
          : length < 100 || !plane
            ? 'Z: 0.2–1.2 m · rotate to view'
            : 'Depth Z (m)',
        width / 2,
        height - 24,
      );
    }
  }, [cloud, scene, view, source, shading, showRuler]);

  function rotate(yaw: number, pitch = 0) {
    setView((v) => ({
      ...v,
      yaw: Math.max(-0.35, Math.min(0.35, v.yaw + yaw)),
      pitch: Math.max(-0.25, Math.min(Math.PI / 2, v.pitch + pitch)),
    }));
  }

  return (
    <section className="figure-section real-explorer" id="real-scenes">
      <h2>Real Scenes</h2>
      <p className="section-intro">
        Large model
      </p>
      <div className="wide">
        <div className="scene-picker">
          <span id="scene-label">Scene</span>
          <Select
            value={scene}
            onValueChange={(value) => {
              if (value) {
                setError('');
                setScene(value);
                setView({ yaw: 0, pitch: 0, distance: 0.7 });
              }
            }}
          >
            <SelectTrigger
              aria-labelledby="scene-label"
              className="scene-select"
            >
              <SelectValue>
                {current.label}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {visibleScenes.map((s) => (
                <SelectItem key={s.id} value={s.id}>
                  {s.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="scene-thumbnails" aria-label="Featured scenes">
          {visibleScenes.map((s) => (
            <button
              type="button"
              key={s.id}
              onClick={() => {
                setError('');
                setScene(s.id);
                setView({ yaw: 0, pitch: 0, distance: 0.7 });
              }}
              aria-label={`Show ${s.label}`}
              aria-pressed={scene === s.id}
            >
              <Image
                unoptimized
                src={`./results/${s.id}-input.webp`}
                width={399}
                height={298}
                alt={s.label}
                loading="lazy"
              />
              <span className="thumbnail-number">
                {s.label}
              </span>
            </button>
          ))}
        </div>
        <div className="depth-panels">
          {[
            ['input', 'Polarization I₁'],
            ['input2', 'Polarization I₂'],
            ['prediction', 'Ours · Large'],
            ['gt', 'Depth labels'],
          ].map(([key, label]) => (
            <figure key={key}>
              <Image
                unoptimized
                src={`./results/${scene}-${key}.webp`}
                width={399}
                height={298}
                alt={`${label}: ${current.label}`}
                loading="lazy"
              />
              <figcaption>{label}</figcaption>
            </figure>
          ))}
        </div>
        <div className="depth-scale">
          <span>0.2 m</span>
          <span className="depth-gradient" />
          <span>1.2 m</span>
        </div>
        <h3 className="cloud-heading">Single-view point cloud</h3>
        <div className="cloud-controls">
          <Select
            value={source}
            onValueChange={(value) => {
              if (value) setSource(value);
            }}
          >
            <SelectTrigger aria-label="Geometry source">
              <SelectValue>
                {source === 'prediction'
                  ? 'Predicted depth'
                  : 'Depth labels (GT)'}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="prediction">Predicted depth</SelectItem>
              <SelectItem value="gt">Depth labels (GT)</SelectItem>
            </SelectContent>
          </Select>
          <Select
            value={shading}
            onValueChange={(value) => {
              if (value) setShading(value);
            }}
          >
            <SelectTrigger aria-label="Point colors">
              <SelectValue>
                {shading === 'depth'
                  ? 'Depth colors'
                  : 'Model input (pseudo-RGB)'}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="depth">Depth colors</SelectItem>
              <SelectItem value="input">Model input (pseudo-RGB)</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <figure className="cloud-figure">
          <div className="cloud-stage">
            {!ready && (
              <Image
                unoptimized
                src={`./results/${scene}-cloud.webp`}
                width={900}
                height={560}
                alt={`Rendered predicted point cloud: ${current.label}`}
              />
            )}
            <canvas
              ref={canvas}
              width={1000}
              height={590}
              style={{ visibility: ready ? 'visible' : 'hidden' }}
              aria-label={`${source === 'gt' ? 'Depth-label' : 'Predicted'} point cloud of ${current.label}. Depth ruler from 0.2 to 1.2 metres, ticks every 0.1 metre. Use the rotation buttons below or drag to change viewpoint.`}
              onPointerDown={(e) => {
                drag.current = { x: e.clientX, y: e.clientY };
                e.currentTarget.setPointerCapture(e.pointerId);
              }}
              onPointerMove={(e) => {
                if (drag.current) {
                  rotate(
                    (e.clientX - drag.current.x) * 0.0015,
                    (e.clientY - drag.current.y) * 0.0015,
                  );
                  drag.current = { x: e.clientX, y: e.clientY };
                }
              }}
              onPointerUp={() => {
                drag.current = null;
              }}
              onPointerCancel={() => {
                drag.current = null;
              }}
            >
              Single-view point cloud. The static preview is available from the
              scene thumbnails.
            </canvas>
            <span className="cloud-source-label">
              {ready
                ? source === 'gt'
                  ? 'GT labels'
                  : 'Large prediction'
                : 'Prediction preview'}{' '}
              · approximate projection
            </span>
          </div>
          <div className="view-controls" aria-label="Point cloud viewpoint">
            <button
              type="button"
              onClick={() => rotate(-0.04)}
              disabled={!ready}
            >
              ← Left
            </button>
            <button
              type="button"
              onClick={() => rotate(0.04)}
              disabled={!ready}
            >
              Right →
            </button>
            <button
              type="button"
              onClick={() => rotate(0, 0.04)}
              disabled={!ready}
            >
              ↑ Up
            </button>
            <button
              type="button"
              onClick={() => rotate(0, -0.04)}
              disabled={!ready}
            >
              ↓ Down
            </button>
            <button
              type="button"
              onClick={() =>
                setView((v) => ({
                  ...v,
                  distance: Math.max(0.65, v.distance / 1.15),
                }))
              }
              disabled={!ready}
              aria-label="Zoom in"
            >
              ＋
            </button>
            <button
              type="button"
              onClick={() =>
                setView((v) => ({
                  ...v,
                  distance: Math.min(60, v.distance * 1.15),
                }))
              }
              disabled={!ready}
              aria-label="Zoom out"
            >
              −
            </button>
            <button
              type="button"
              onClick={() => {
                setView({ yaw: 0, pitch: 0, distance: 0.7 });
              }}
              disabled={!ready}
            >
              Front view
            </button>
            <button
              type="button"
              disabled={!ready}
              onClick={() => {
                setView({ yaw: 0, pitch: Math.PI / 2, distance: 18 });
              }}
            >
              Top view
            </button>
            <button
              type="button"
              disabled={!ready}
              aria-pressed={showRuler}
              onClick={() => setShowRuler((shown) => !shown)}
            >
              {showRuler ? 'Hide ruler' : 'Show ruler'}
            </button>
            <button
              type="button"
              onClick={() => {
                setView({ yaw: 0, pitch: 0, distance: 0.7 });
              }}
              disabled={!ready}
            >
              Reset
            </button>
          </div>
          <output className="cloud-status">
            {error ||
              (ready
                ? 'Drag to rotate, or use the controls.'
                : 'Loading interactive point cloud…')}
          </output>
          <figcaption>
            Point clouds from predicted depth, colored by the model input.
            Camera intrinsics are approximate; missing surfaces are not filled.
          </figcaption>
        </figure>
        <details className="technical-notes">
          <summary>Rendering and data notes</summary>
          <p>
            This gallery contains a subset of the 42 evaluation scenes.{' '}
            Predictions use the Large mixed-training checkpoint. The depth color
            scale spans 0.2–1.2 m. Input colors are pseudo-RGB channels formed
            from the two monochrome captures.
          </p>
          <p>
            29,800 uniformly sampled points per scene, with no GT mask,
            denoising, or Z exaggeration. X = (u − 797.5)Z / 15666.67; Y = (v −
            594.5)Z / 15666.67. This nominal projection uses the paper’s 37.6 mm
            sensor distance and 2.4 μm pitch, assuming a centered crop without
            resizing. Effective intrinsics after preprocessing are unverified.
            GT consists of manually annotated object-distance regions, not
            scanned surfaces.
          </p>
        </details>
      </div>
    </section>
  );
}
