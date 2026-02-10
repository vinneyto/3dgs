# Roadmap

## Core TODOs

- [x] Add radix sort
- [x] Add spherical harmonics support
- [ ] Move the splat storage structure into a dedicated module
- [ ] Add a raycast module (BVH or brute-force)
- [ ] Add SOG2 loading support (see SPARK_JS_SOG2_LOD_RESEARCH.md — spark.js supports SOG v1+v2 via pcsogs.ts: ZIP with meta.json + WebP images, codebook decoding for v2)
- [ ] Add GPU frustum culling
- [ ] Decouple the renderer from React and move it into a standalone library
- [ ] Add support for placing multiple Gaussian Splatting objects in a scene
- [ ] Add streaming loading of LOD tiles (NOTE: SOG2 format does NOT have built-in LOD. spark.js uses SPZ chunks + RAD format for LOD in splat-quick-lod branch. Consider similar approach or custom LOD over SOG2 chunks)
- [ ] Optimize splat data storage
- [ ] Optimize radix sort
- [ ] Optimize GPU frustum culling
- [ ] Tile-based compute renderer
