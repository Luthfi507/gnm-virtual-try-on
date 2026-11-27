import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { defineConfig } from 'vite';

const here = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  base: './',
  server: { port: 5173 },
  build: {
    target: 'es2022',
    rollupOptions: {
      input: {
        main: resolve(here, 'index.html'),
      },
    },
    chunkSizeWarningLimit: 2048,
  },
});
