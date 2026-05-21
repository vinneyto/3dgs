import {
  BoxGeometry,
  ColorRepresentation,
  Group,
  MathUtils,
  Mesh,
  MeshStandardMaterial,
} from "three";

export class RotatingCubes {
  readonly group = new Group();
  private readonly cubes: Mesh[] = [];

  constructor() {
    const geo = new BoxGeometry(0.42, 0.42, 0.42);
    const colors: ColorRepresentation[] = ["#ff8a65", "#4fc3f7", "#aed581", "#ce93d8"];

    for (let i = 0; i < 4; i += 1) {
      const cube = new Mesh(
        geo,
        new MeshStandardMaterial({
          color: colors[i],
          metalness: 0.15,
          roughness: 0.35,
        }),
      );

      const angle = (i / 4) * Math.PI * 2;
      cube.position.set(Math.cos(angle) * 1.15, (i % 2 === 0 ? 0.28 : -0.28), Math.sin(angle) * 1.15);
      cube.rotation.set(MathUtils.degToRad(i * 15), MathUtils.degToRad(i * 20), 0);

      this.cubes.push(cube);
      this.group.add(cube);
    }
  }

  update(t: number) {
    for (let i = 0; i < this.cubes.length; i += 1) {
      const cube = this.cubes[i];
      cube.rotation.x += 0.01 + i * 0.002;
      cube.rotation.y += 0.013 + i * 0.002;
      cube.position.y = (i % 2 === 0 ? 0.28 : -0.28) + Math.sin(t * 1.6 + i) * 0.08;
    }
  }
}
