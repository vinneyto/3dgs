This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.

## Plasma PIC demo planning (draft)

This repository now includes a placeholder page for the future WebGPU Particle-In-Cell plasma simulation demo:

- Route: `/plasma-pic`
- File: `app/(pages)/plasma-pic/page.tsx`

### Questions to finalize before implementation

1. **Physical model scope**
   - Electrostatic PIC only (Poisson + E field), or full electromagnetic PIC (Maxwell + B field)?
   - Confirm if magnetic effects are out of scope for MVP.

2. **Field boundary conditions (cube side = 0.1 m)**
   - Dirichlet (`phi=0` on walls), Neumann (`dphi/dn=0`), or periodic boundaries?

3. **Particle boundary conditions**
   - Absorb on walls, reflect on walls, or periodic wrapping?
   - Should wall charge accumulation be modeled?

4. **Initial neutrality and species setup**
   - Must `N_e = N_p` initially?
   - Are electron/proton spatial distributions identical at start?

5. **Collisions**
   - Collisionless plasma for MVP, or include collision model?

6. **Cylinder initialization parameters**
   - Required inputs: center `(x,y,z)`, axis direction, radius `R`, height `H`, density `n`, and/or particle counts?

7. **Initial velocity distribution**
   - Zero velocities, Maxwellian (`T_e`, `T_p`), and/or bulk drift velocity?

8. **Macro-particle strategy**
   - Inputs as macro-particle counts + physical density (for particle weight), or normalized units only?

9. **Grid sizing from Debye length**
   - Compute `dx` from Debye length and clamp to `Nx,Ny,Nz <= 64`?
   - Behavior when desired resolution exceeds 64 per axis?

10. **Time step policy**
    - Auto `dt` from plasma frequency / Courant-like constraints?
    - Allow manual override?

11. **Charge deposition / field gather shape function**
    - NGP, CIC, or TSC? (CIC recommended for MVP.)

12. **Relativistic particle pusher**
    - Relativistic momentum update (`p = gamma m v`) with leapfrog-like integrator?

13. **Units and normalization**
    - Pure SI units in shaders, or normalized plasma units for numerical stability?

14. **Poisson solver in WebGPU compute**
    - Preferred solver for MVP: Jacobi / weighted Jacobi / RBGS / CG / FFT-based?

15. **Poisson iteration control**
    - Fixed iteration count per step, or residual-based convergence criterion?

16. **Sub-iterations per simulation step**
    - Is multiple Poisson iterations per step acceptable for stability at lower FPS?

17. **Demo platform and architecture**
    - Confirm target is browser demo with Next.js + WebGPU.

18. **Visualization requirements for MVP**
    - Particle cloud only, or also slices/maps for charge density, potential, and electric field?

19. **Controls/UI requirements**
    - Need live controls for density, temperatures, dt, solver iterations, pause/reset/step?

20. **Performance target**
    - Expected macro-particle count and FPS target (e.g., 50k/100k/500k at 30/60 FPS)?

21. **GPU data layout constraints**
    - Storage buffers only, or 3D textures allowed for grid fields?

22. **Validation criteria**
    - Which diagnostics are mandatory: charge conservation, Poisson residual, energy trend, etc.?

23. **Reference validation scenario**
    - Should we include a known test case with approximate analytical behavior for solver sanity checks?

### Working process for this feature

- We can keep this section as the living checklist in the PR.
- You provide answers item-by-item.
- After all items are clarified, we use the finalized checklist as the implementation prompt.
