import react from '@vitejs/plugin-react';
import { copyFile } from 'fs/promises';
import { mkdir } from 'fs/promises';
import path, { dirname, join } from "path";
import { fileURLToPath } from "url";
import { defineConfig } from 'vite';
;

function viteStaticCopyPyodide() {
  return {
    name: "vite-plugin-pyodide",
    generateBundle: async () => {
      const assetsDir = "dist/assets";
      await mkdir(assetsDir, { recursive: true });
      const files = [
        "pyodide-lock.json",
        "pyodide.asm.js",
        "pyodide.asm.wasm",
        "python_stdlib.zip",
      ];
      const modulePath = fileURLToPath(import.meta.resolve("pyodide"));
      for (const file of files) {
        await copyFile(
          join(dirname(modulePath), file),
          join(assetsDir, file),
        );
      }
    },
  }
  // const pyodideDir = dirname(fileURLToPath(import.meta.resolve("pyodide")));
  // return viteStaticCopy({
  //   targets: [
  //     {
  //       src: [join(pyodideDir, "*")].concat(PYODIDE_EXCLUDE),
  //       dest: "assets",
  //     },
  //   ],
  // });
}

// https://vite.dev/config/
export default defineConfig({
  optimizeDeps: { exclude: ["pyodide"] },
  server: {
    fs: {
      // Allow serving files from node_modules
      allow: ['..', 'node_modules']
    }
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },

  plugins: [react(), viteStaticCopyPyodide()],
})
