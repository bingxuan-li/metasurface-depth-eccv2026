import assert from 'node:assert/strict';
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, resolve } from 'node:path';

const root = resolve('dist');
const manifest = JSON.parse(readFileSync(join(root, 'results/manifest.json'), 'utf8'));
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
  assert(!/\/api\/selection|\/review\b|chatgpt\.site|Watch on YouTube|Review scenes/.test(text), `Private or obsolete UI in ${path}`);
}
const html = readFileSync(join(root,'index.html'),'utf8');
assert(html.includes('<h1>') && html.includes('youtube-nocookie.com/embed/Qkb4nXjKlwU'));
for (const match of html.matchAll(/(?:src|href)="\.\/([^"?#]+)"/g))
  assert(existsSync(join(root,match[1])), `Missing asset: ${match[1]}`);
console.log('PASS: static page, five selected scenes, all local assets, no review/API links');
