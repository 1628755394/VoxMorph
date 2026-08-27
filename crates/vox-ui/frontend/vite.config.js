import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

// Tauri 期望 dev server 在 1420 端口，且使用 strictPort 避免端口漂移。
export default defineConfig({
  plugins: [svelte()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
  },
  // Tauri webview 通过自定义协议加载，需相对路径。
  base: './',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
