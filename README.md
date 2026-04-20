# 3dgs monorepo

Repository converted to an npm workspace monorepo. Right now it contains a
single package:

- `packages/gallery` - the Next.js demo gallery application

Other top-level directories such as `docs/` and `mlcv/` stay in the repository,
but are not npm workspace packages.

## Workspace layout

```text
.
|-- package.json
|-- packages/
|   `-- gallery/
|       |-- app/
|       |-- public/
|       |-- scripts/
|       |-- wasm/
|       `-- package.json
|-- docs/
`-- mlcv/
```

## Getting started

Install dependencies from the repository root:

```bash
npm install
```

Start the gallery from the workspace root:

```bash
npm run dev
```

The root scripts proxy into `@3dgs/gallery`, so the following also work from
the root:

```bash
npm run build
npm run lint
npm run wasm:build
npm run wasm:watch
```

You can also run commands directly against the package:

```bash
npm run dev --workspace @3dgs/gallery
```

## Gallery package

The demo gallery now lives in `packages/gallery`. Its main sources are:

- `packages/gallery/app` - App Router pages and shared UI/demo code
- `packages/gallery/public` - static assets
- `packages/gallery/wasm` - Rust + wasm-pack sources
- `packages/gallery/scripts` - local development scripts
