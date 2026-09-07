import { readFile, writeFile } from 'node:fs/promises';
import { render } from '../.prerender/prerender.js';

const path = new URL('../dist/index.html', import.meta.url);
const template = await readFile(path, 'utf8');
if (!template.includes('<!--prerender-->')) throw new Error('Missing prerender marker');
await writeFile(path, template.replace('<!--prerender-->', render()));
