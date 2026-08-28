import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { fileURLToPath, URL } from 'node:url';

// Tauri 期望 dev server 在 9090 端口，且使用 strictPort 避免端口漂移。
export default defineConfig({
  plugins: [svelte()],
  clearScreen: false,
  resolve: {
    alias: {
      $lib: fileURLToPath(new URL('./src/lib', import.meta.url)),
    },
  },
  server: {
    port: 9090,
    strictPort: true,
  },
  // Tauri webview 通过自定义协议加载，需相对路径。
  base: './',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
