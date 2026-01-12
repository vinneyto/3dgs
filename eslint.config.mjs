import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // Migration sandbox (old Vite app lives here for now):
    "tmp/**",
    // Generated WASM bindings (wasm-pack output):
    "app/_shared/wasm/pkg/**",
  ]),
  // These demos intentionally mutate Three/WebGPU objects (uniforms, BufferAttributes, etc).
  // The immutability rule is not a good fit here.
  {
    files: ["app/_shared/**/*.{ts,tsx}", "app/(pages)/**/*.{ts,tsx}"],
    rules: {
      "react-hooks/immutability": "off",
    },
  },
]);

export default eslintConfig;
