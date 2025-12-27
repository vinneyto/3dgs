import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const projectRoot = process.cwd();
const wasmDir = path.join(projectRoot, "wasm");
const wasmSrcDir = path.join(wasmDir, "src");
const wasmCargoToml = path.join(wasmDir, "Cargo.toml");
const wasmCargoLock = path.join(wasmDir, "Cargo.lock");

let building = false;
let pending = false;
let debounceTimer = null;

function runBuild() {
  if (building) {
    pending = true;
    return;
  }
  building = true;
  pending = false;

  const cmd = process.platform === "win32" ? "npm.cmd" : "npm";
  const child = spawn(cmd, ["run", "wasm:build"], { stdio: "inherit" });
  child.on("exit", (code) => {
    building = false;
    if (pending) runBuild();
    if (code !== 0) {
      // Keep watching even if the build fails.
      console.warn(
        `[wasm:watch] build failed (exit ${code}). Watching for changes...`
      );
    }
  });
}

function scheduleBuild() {
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => runBuild(), 150);
}

function watchDir(dir) {
  // Recursive watch is supported on macOS/Windows; on Linux it may be non-recursive.
  // For our use (macOS), this works well and keeps dependencies at zero.
  return fs.watch(dir, { recursive: true }, (_eventType, filename) => {
    if (!filename) return;
    // Ignore target/ and generated pkg output.
    const f = String(filename);
    if (
      f.includes("target/") ||
      f.includes("target\\") ||
      f.includes(".wasm-pack")
    )
      return;
    scheduleBuild();
  });
}

console.log("[wasm:watch] Watching Rust sources for changes...");
console.log(`- ${path.relative(projectRoot, wasmSrcDir)}`);
console.log(`- ${path.relative(projectRoot, wasmCargoToml)}`);
console.log(`- ${path.relative(projectRoot, wasmCargoLock)}`);
console.log("[wasm:watch] Running initial build...");

runBuild();

const watchers = [];
watchers.push(watchDir(wasmSrcDir));

watchers.push(
  fs.watch(wasmCargoToml, () => scheduleBuild()),
  fs.watch(wasmCargoLock, () => scheduleBuild())
);

process.on("SIGINT", () => {
  for (const w of watchers) w.close();
  process.exit(0);
});
