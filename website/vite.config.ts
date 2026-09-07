import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/postcss';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  base: process.env.PAGES_BASE_PATH || '/',
  plugins: [react()],
  resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },
  css: { postcss: { plugins: [tailwindcss()] } },
});
