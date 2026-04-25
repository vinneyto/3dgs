#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';

const repoRoot = process.cwd();
const movedFromPrefix = path.join(repoRoot, 'packages', 'examples');
const nextRoot = path.join(repoRoot, '.next');

function walk(dir, acc = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(p, acc);
    else if (p.endsWith('.nft.json')) acc.push(p);
  }
  return acc;
}

function toPosix(p) {
  return p.split(path.sep).join('/');
}

if (!fs.existsSync(nextRoot)) {
  console.error('No .next directory found, nothing to retarget');
  process.exit(1);
}

const nftFiles = walk(nextRoot);
let updatedCount = 0;
let entryCount = 0;

for (const filePath of nftFiles) {
  const raw = fs.readFileSync(filePath, 'utf8');
  const json = JSON.parse(raw);
  if (!Array.isArray(json.files)) continue;

  const relFromNextRoot = path.relative(nextRoot, filePath);
  const newFileDir = path.dirname(filePath);
  const oldFileDir = path.dirname(path.join(movedFromPrefix, '.next', relFromNextRoot));

  const rewritten = json.files.map((rel) => {
    entryCount += 1;
    const targetAbs = path.normalize(path.join(oldFileDir, rel));
    const newRel = path.relative(newFileDir, targetAbs);
    return toPosix(newRel);
  });

  json.files = rewritten;
  fs.writeFileSync(filePath, JSON.stringify(json));
  updatedCount += 1;
}

console.log(`Retargeted NFT paths in ${updatedCount} files (${entryCount} entries)`);
