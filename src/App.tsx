import { Navigate, NavLink, Route, Routes } from "react-router-dom";
import "./App.css";
import { CovariancePage } from "./pages/CovariancePage.tsx";
import { InstancedSplatsPage } from "./pages/InstancedSplatsPage.tsx";
import { BillboardCirclePage } from "./pages/BillboardCirclePage.tsx";
import { BillboardCircleStoragePage } from "./pages/BillboardCircleStoragePage.tsx";
import { SplatComparePage } from "./pages/SplatComparePage.tsx";
import { SplatQuadPage } from "./pages/SplatQuadPage.tsx";
import { PlyHeaderPage } from "./pages/PlyHeaderPage.tsx";
import { PlyEllipsoidsPage } from "./pages/PlyEllipsoidsPage.tsx";
import { RustWasmPlyParsePage } from "./pages/RustWasmPlyParsePage.tsx";

export default function App() {
  return (
    <div className="appShell">
      <aside className="sidebar">
        <div className="sidebarHeader">3DGS examples</div>
        <nav className="nav">
          <NavLink
            className={({ isActive }) => `navLink${isActive ? " active" : ""}`}
            to="/covariance"
          >
            Covariance (debug)
          </NavLink>
          <NavLink
            className={({ isActive }) => `navLink${isActive ? " active" : ""}`}
            to="/splat-quad"
          >
            Splat quad (debug)
          </NavLink>
          <NavLink
            className={({ isActive }) => `navLink${isActive ? " active" : ""}`}
            to="/compare"
          >
            Compare (ellipsoid + sprite)
          </NavLink>
          <NavLink
            className={({ isActive }) => `navLink${isActive ? " active" : ""}`}
            to="/instanced"
          >
            Instanced splats (storage buffer)
          </NavLink>
          <NavLink
            className={({ isActive }) => `navLink${isActive ? " active" : ""}`}
            to="/billboard-circles"
          >
            Billboard circles (TSL)
          </NavLink>
          <NavLink
            className={({ isActive }) => `navLink${isActive ? " active" : ""}`}
            to="/billboard-circles-buffer"
          >
            Billboard circles (storage buffer)
          </NavLink>
          <NavLink
            className={({ isActive }) => `navLink${isActive ? " active" : ""}`}
            to="/ply-header"
          >
            PLY header (cactus)
          </NavLink>
          <NavLink
            className={({ isActive }) => `navLink${isActive ? " active" : ""}`}
            to="/ply-ellipsoids"
          >
            PLY ellipsoids (instanced)
          </NavLink>
          <NavLink
            className={({ isActive }) => `navLink${isActive ? " active" : ""}`}
            to="/rust-wasm"
          >
            PLY parse (Rust WASM)
          </NavLink>
        </nav>
      </aside>

      <main className="content">
        <Routes>
          <Route path="/" element={<Navigate to="/covariance" replace />} />
          <Route path="/covariance" element={<CovariancePage />} />
          <Route path="/splat-quad" element={<SplatQuadPage />} />
          <Route path="/compare" element={<SplatComparePage />} />
          <Route path="/instanced" element={<InstancedSplatsPage />} />
          <Route path="/billboard-circles" element={<BillboardCirclePage />} />
          <Route
            path="/billboard-circles-buffer"
            element={<BillboardCircleStoragePage />}
          />
          <Route path="/ply-header" element={<PlyHeaderPage />} />
          <Route path="/ply-ellipsoids" element={<PlyEllipsoidsPage />} />
          <Route path="/rust-wasm" element={<RustWasmPlyParsePage />} />
          <Route path="/gaussian-splat" element={<Navigate to="/covariance" replace />} />
          <Route path="*" element={<Navigate to="/covariance" replace />} />
        </Routes>
      </main>
    </div>
  );
}
