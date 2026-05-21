import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import { FXAAShader } from "three/examples/jsm/shaders/FXAAShader.js";

export class FxaaTransparentDemo {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene: THREE.Scene;
  private readonly camera: THREE.PerspectiveCamera;
  private readonly composer: EffectComposer;
  private readonly fxaaPass: ShaderPass;

  private rafId: number | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private readonly clock = new THREE.Clock();

  private readonly group = new THREE.Group();

  constructor(renderer: THREE.WebGLRenderer) {
    this.renderer = renderer;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 100);
    this.camera.position.set(0, 0.4, 3.4);

    const ambient = new THREE.AmbientLight(0xffffff, 0.45);
    const key = new THREE.DirectionalLight(0xffffff, 1.0);
    key.position.set(2, 3, 4);

    const geo = new THREE.TorusKnotGeometry(0.58, 0.2, 256, 32);
    const mat = new THREE.MeshStandardMaterial({
      color: "#7cc6ff",
      metalness: 0.1,
      roughness: 0.3,
    });

    const mesh = new THREE.Mesh(geo, mat);
    this.group.add(mesh);

    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(1.2, 0.02, 12, 128),
      new THREE.MeshBasicMaterial({
        color: "#ffd36b",
        transparent: true,
        opacity: 0.8,
      }),
    );
    ring.rotation.x = Math.PI / 2;
    this.group.add(ring);

    this.scene.add(ambient, key, this.group);

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));

    this.fxaaPass = new ShaderPass(FXAAShader);
    this.composer.addPass(this.fxaaPass);

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
