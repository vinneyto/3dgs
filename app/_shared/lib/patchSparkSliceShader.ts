import { Color, Vector3 } from "three";
import type { SparkRenderer } from "@sparkjsdev/spark";

export type SparkSliceShaderUniforms = {
  sliceHighlightEnabled: { value: boolean };
  slicePlaneOrigin: { value: Vector3 };
  slicePlaneNormal: { value: Vector3 };
  sliceHighlightColor: { value: Color };
  sliceHighlightWidth: { value: number };
  sliceHighlightStrength: { value: number };
};

type SparkUniformMap = Record<string, { value: unknown }> &
  SparkSliceShaderUniforms;
type MutableUniformMap = Record<string, { value: unknown }>;

const SLICE_SHADER_FLAG = "sliceHighlightEnabled";

export function patchSparkSliceShader(
  spark: SparkRenderer,
): SparkSliceShaderUniforms {
  const uniforms = spark.uniforms as unknown as MutableUniformMap;
  if (SLICE_SHADER_FLAG in uniforms) {
    return uniforms as SparkSliceShaderUniforms;
  }

  const material = spark.material;
  const sliceUniforms = uniforms as SparkUniformMap;

  sliceUniforms.sliceHighlightEnabled = { value: true };
  sliceUniforms.slicePlaneOrigin = { value: new Vector3(0, 0, 0) };
  sliceUniforms.slicePlaneNormal = { value: new Vector3(0, 0, 1) };
  sliceUniforms.sliceHighlightColor = {
    value: new Color("#ff8a00").convertSRGBToLinear(),
  };
  sliceUniforms.sliceHighlightWidth = { value: 0.08 };
  sliceUniforms.sliceHighlightStrength = { value: 2.25 };

  if (!material.vertexShader.includes("vSliceCenter")) {
    material.vertexShader = material.vertexShader
      .replace(
        "flat out uint vSplatIndex;\n",
        [
          "flat out uint vSplatIndex;",
          "flat out vec3 vSliceCenter;",
          "flat out vec3 vSliceAxisX;",
          "flat out vec3 vSliceAxisY;",
          "",
        ].join("\n"),
      )
      .replace(
        [
          "    vec3 center, scales;",
          "    vec4 quaternion, rgba;",
          "    unpackSplatEncoding(packed, center, scales, quaternion, rgba, rgbMinMaxLnScaleMinMax);",
          "",
        ].join("\n"),
        [
          "    vec3 center, scales;",
          "    vec4 quaternion, rgba;",
          "    unpackSplatEncoding(packed, center, scales, quaternion, rgba, rgbMinMaxLnScaleMinMax);",
          "",
          "    vSliceCenter = center;",
          "    vSliceAxisX = quatVec(quaternion, vec3(scales.x, 0.0, 0.0));",
          "    vSliceAxisY = quatVec(quaternion, vec3(0.0, scales.y, 0.0));",
          "",
        ].join("\n"),
      );
  }

  if (!material.fragmentShader.includes("vSliceCenter")) {
    material.fragmentShader = material.fragmentShader
      .replace(
        [
          "flat in uint vSplatIndex;",
          "",
          "void main() {",
        ].join("\n"),
        [
          "flat in uint vSplatIndex;",
          "flat in vec3 vSliceCenter;",
          "flat in vec3 vSliceAxisX;",
          "flat in vec3 vSliceAxisY;",
          "",
          "uniform bool sliceHighlightEnabled;",
          "uniform vec3 slicePlaneOrigin;",
          "uniform vec3 slicePlaneNormal;",
          "uniform vec3 sliceHighlightColor;",
          "uniform float sliceHighlightWidth;",
          "uniform float sliceHighlightStrength;",
          "",
          "void main() {",
        ].join("\n"),
      )
      .replace(
        [
          "    if (encodeLinear) {",
          "        rgba.rgb = srgbToLinear(rgba.rgb);",
          "    }",
          "",
          "    if (stochastic) {",
        ].join("\n"),
        [
          "    if (encodeLinear) {",
          "        rgba.rgb = srgbToLinear(rgba.rgb);",
          "    }",
          "",
          "    if (sliceHighlightEnabled) {",
          "        vec3 planeNormal = normalize(slicePlaneNormal);",
          "        vec3 sliceEvalPos = vSliceCenter + vSplatUv.x * vSliceAxisX + vSplatUv.y * vSliceAxisY;",
          "        float sliceWidth = max(sliceHighlightWidth, 1e-5);",
          "        float planeDistance = abs(dot(sliceEvalPos - slicePlaneOrigin, planeNormal));",
          "        float sliceHighlight = 1.0 - smoothstep(0.0, sliceWidth, planeDistance);",
          "        sliceHighlight *= rgba.a * sliceHighlightStrength;",
          "        rgba.rgb += sliceHighlightColor * sliceHighlight;",
          "    }",
          "",
          "    if (stochastic) {",
        ].join("\n"),
      );
  }

  material.uniforms = sliceUniforms;
  material.needsUpdate = true;

  return sliceUniforms;
}
