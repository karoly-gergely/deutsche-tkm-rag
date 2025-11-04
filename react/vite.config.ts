import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const basename = env.VITE_BASENAME || "/deutsche-tkm-rag";

  return {
    server: {
      host: "::",
      port: 5173,
    },
    plugins: [react()],
    base: mode === "production" ? `${basename}/` : "/",
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    build: {
      outDir: "dist",
      sourcemap: false,
      chunkSizeWarningLimit: 1000,
    },
  };
});
