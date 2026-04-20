import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next for the package-local app.
  globalIgnores([
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    "tmp/**",
    "app/_shared/wasm/pkg/**",
    "external/**",
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
