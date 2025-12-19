import { Navigate, NavLink, Route, Routes } from "react-router-dom";
import "./App.css";
import { GaussianSplatPage } from "./pages/GaussianSplatPage.tsx";

export default function App() {
  return (
    <div className="appShell">
      <aside className="sidebar">
        <div className="sidebarHeader">3DGS examples</div>
        <nav className="nav">
          <NavLink
            className={({ isActive }) => `navLink${isActive ? " active" : ""}`}
            to="/gaussian-splat"
          >
            Gaussian Splat
          </NavLink>
        </nav>
      </aside>

      <main className="content">
        <Routes>
          <Route path="/" element={<Navigate to="/gaussian-splat" replace />} />
          <Route path="/gaussian-splat" element={<GaussianSplatPage />} />
          <Route path="*" element={<Navigate to="/gaussian-splat" replace />} />
        </Routes>
      </main>
    </div>
  );
}
