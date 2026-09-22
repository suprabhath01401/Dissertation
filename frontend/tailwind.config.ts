import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        legal: "#3b82f6",     // blue-500
        app: "#22c55e",       // green-500
        cluster: "#a855f7",   // purple-500
        laptop: "#6b7280",    // gray-500
      },
    },
  },
  plugins: [],
} satisfies Config;
