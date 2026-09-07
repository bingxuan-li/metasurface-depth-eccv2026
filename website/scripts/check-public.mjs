import assert from 'node:assert/strict';
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, resolve } from 'node:path';

const root = resolve('dist');
const manifest = JSON.parse(readFileSync(join(root, 'results/manifest.json'), 'utf8'));
assert.equal(
  readFileSync(join(root, 'results/manifest.json'), 'utf8'),
  readFileSync(resolve('src/data/real-scenes.json'), 'utf8'),
  'Published manifest must match the single source',
);
assert.equal(manifest.scenes.length, 5);
assert.deepEqual(manifest.scenes.map(s => s.id), ['camera', 'cat', 'dragon', 'three-objects', 'four-objects']);
assert.equal(readdirSync(join(root, 'results')).filter(f => f.endsWith('.bin')).length, 5);
for (const scene of manifest.scenes) {
  assert.equal(statSync(join(root, 'results', scene.id + '.bin')).size, scene.points * 7 * 4);
  for (const suffix of ['input', 'input2', 'prediction', 'gt', 'cloud'])
    assert(existsSync(join(root, 'results', `${scene.id}-${suffix}.webp`)));
}
function walk(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap(e => e.isDirectory() ? walk(join(dir,e.name)) : [join(dir,e.name)]);
}
for (const path of walk(root).filter(p => /\.(js|html|json)$/.test(p))) {
  const text = readFileSync(path, 'utf8');
  assert(!/\/api\/selection|\/review\b|chatgpt\.site|Watch on YouTube|Review scenes|More Real-World Results|Rendering and data notes|Geometry source|Depth labels \(GT\)|Dataset \(forthcoming\)/.test(text), `Private or obsolete UI in ${path}`);
}
const html = readFileSync(join(root,'index.html'),'utf8');
assert(html.includes('<h1>') && html.includes('youtube-nocookie.com/embed/Qkb4nXjKlwU'));
assert(!/\p{Script=Han}/u.test(html), 'Public project page must remain English-only');
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
assert.equal(new Set(ids).size, ids.length, 'Duplicate page anchor');
for (const match of html.matchAll(/href="#([^"]+)"/g))
  assert(ids.includes(match[1]), `Missing anchor: ${match[1]}`);
for (const url of [
  'https://github.com/bingxuan-li/metasurface-depth-eccv2026',
  'https://huggingface.co/Bingxuan111/metasurface-depth-eccv2026',
  'https://huggingface.co/datasets/Bingxuan111/metasurface-real-eccv2026',
]) assert(html.includes(`href="${url}"`), `Missing resource: ${url}`);
// The canvas has fixed 1000 x 590 drawing coordinates; CSS must not stretch it.
const css = readFileSync(resolve('src/globals.css'), 'utf8');
const ratios = [...css.matchAll(/\.cloud-stage\s*\{[^}]*aspect-ratio:\s*([^;]+);/g)]
  .map(match => match[1].replace(/\s/g, ''));
assert.deepEqual(ratios, ['1000/590'], 'Keep one undistorted canvas ratio at every breakpoint');
for (const match of html.matchAll(/(?:src|href)="\.\/([^"?#]+)"/g))
  assert(existsSync(join(root,match[1])), `Missing asset: ${match[1]}`);
console.log('PASS: five selected scenes, manifest parity, assets, anchors, public links, English copy, canvas ratio, no obsolete UI');
