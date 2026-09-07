import { copyFile, readFile, writeFile } from 'node:fs/promises';
import { render } from '../.prerender/prerender.js';

const path = new URL('../dist/index.html', import.meta.url);
const template = await readFile(path, 'utf8');
if (!template.includes('<!--prerender-->')) throw new Error('Missing prerender marker');
await writeFile(path, template.replace('<!--prerender-->', render()));
// Keep one editable manifest; publish the identical provenance file with the assets.
await copyFile(
  new URL('../src/data/real-scenes.json', import.meta.url),
  new URL('../dist/results/manifest.json', import.meta.url),
);
