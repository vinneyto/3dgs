import { Navigate, NavLink, Route, Routes } from "react-router-dom";
import "./App.css";
import { CovariancePage } from "./pages/CovariancePage.tsx";
import { SplatQuadPage } from "./pages/SplatQuadPage.tsx";

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
        </nav>
      </aside>

      <main className="content">
        <Routes>
          <Route path="/" element={<Navigate to="/covariance" replace />} />
          <Route path="/covariance" element={<CovariancePage />} />
          <Route path="/splat-quad" element={<SplatQuadPage />} />
          <Route path="/gaussian-splat" element={<Navigate to="/covariance" replace />} />
          <Route path="*" element={<Navigate to="/covariance" replace />} />
        </Routes>
      </main>
    </div>
  );
}
