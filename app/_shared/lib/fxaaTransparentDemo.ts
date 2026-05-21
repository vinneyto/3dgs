import {
  AmbientLight,
  Clock,
  DirectionalLight,
  Group,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  PerspectiveCamera,
  Scene,
  TorusGeometry,
  TorusKnotGeometry,
  WebGLRenderer,
} from "three";
import { RotatingCubes } from "@/app/_shared/lib/RotatingCubes";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import { FXAAShader } from "three/examples/jsm/shaders/FXAAShader.js";

export class FxaaTransparentDemo {
  private readonly renderer: WebGLRenderer;
  private readonly scene: Scene;
  private readonly camera: PerspectiveCamera;
  private readonly composer: EffectComposer;
  private readonly fxaaPass: ShaderPass;

  private rafId: number | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private readonly clock = new Clock();

  private readonly group = new Group();
  private readonly rotatingCubes = new RotatingCubes();

  constructor(renderer: WebGLRenderer) {
    this.renderer = renderer;

    this.scene = new Scene();
    this.camera = new PerspectiveCamera(55, 1, 0.1, 100);
    this.camera.position.set(0, 0.4, 3.4);

    const ambient = new AmbientLight(0xffffff, 0.5);
    const key = new DirectionalLight(0xffffff, 1.15);
    key.position.set(2, 3, 4);

    const fill = new DirectionalLight(0x9ecbff, 0.55);
    fill.position.set(-3, 1.5, -2);

    const geo = new TorusKnotGeometry(0.58, 0.2, 256, 32);
    const mat = new MeshStandardMaterial({
      color: "#7cc6ff",
      metalness: 0.1,
      roughness: 0.3,
    });

    const mesh = new Mesh(geo, mat);
    this.group.add(mesh);

    const ring = new Mesh(
      new TorusGeometry(1.2, 0.02, 12, 128),
      new MeshBasicMaterial({
        color: "#ffd36b",
        transparent: true,
        opacity: 0.8,
      }),
    );
    ring.rotation.x = Math.PI / 2;
    this.group.add(ring);

    this.scene.add(ambient, key, fill, this.rotatingCubes.group, this.group);

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));

    this.fxaaPass = new ShaderPass(FXAAShader);
    this.composer.addPass(this.fxaaPass);
    this.composer.addPass(new OutputPass());

    this.renderer.setClearColor(0x000000, 0);
  }

  attach(container: HTMLElement) {
    this.resize(container.clientWidth, container.clientHeight);

    this.resizeObserver = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const { width, height } = entry.contentRect;
      this.resize(Math.max(1, width), Math.max(1, height));
    });
    this.resizeObserver.observe(container);

    this.start();
  }

  detach() {
    this.stop();
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
  }

  private resize(width: number, height: number) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(width, height, false);

    this.composer.setPixelRatio(dpr);
    this.composer.setSize(width, height);

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();

    this.fxaaPass.material.uniforms["resolution"].value.set(
      1 / (width * dpr),
      1 / (height * dpr),
    );
  }

  private start() {
    if (this.rafId !== null) return;

    const tick = () => {
      const t = this.clock.getElapsedTime();
      this.group.rotation.y = t * 0.55;
      this.group.rotation.x = Math.sin(t * 0.7) * 0.18;
      this.rotatingCubes.update(t);
      this.composer.render();
      this.rafId = window.requestAnimationFrame(tick);
    };

    this.rafId = window.requestAnimationFrame(tick);
  }

  private stop() {
    if (this.rafId !== null) {
      window.cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  }
}
