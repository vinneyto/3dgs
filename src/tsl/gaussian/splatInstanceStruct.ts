import { struct } from "three/tsl";

export const SplatInstanceStruct = struct(
  {
    center: "vec4",
    covA: "vec4",
    covB: "vec4",
    colorOpacity: "vec4",
  },
  "SplatInstance"
);
