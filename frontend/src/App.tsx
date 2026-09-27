import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";
import { CommandPalette } from "./components/CommandPalette";
import { Layout } from "./components/Layout";
import { Spinner } from "./components/ui";

const Dashboard = lazy(() => import("./pages/Dashboard"));
const Studio = lazy(() => import("./pages/Studio"));
const Forge = lazy(() => import("./pages/Forge"));
const Miner = lazy(() => import("./pages/Miner"));
const Library = lazy(() => import("./pages/Library"));
const ImportPage = lazy(() => import("./pages/ImportPage"));
const BrainPage = lazy(() => import("./pages/BrainPage"));
const Explorer = lazy(() => import("./pages/Explorer"));
const DataPage = lazy(() => import("./pages/DataPage"));
const SettingsPage = lazy(() => import("./pages/SettingsPage"));

export default function App() {
  return (
    <Layout>
      <Suspense fallback={<div className="p-6"><Spinner size={18} /></div>}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/studio" element={<Studio />} />
          <Route path="/forge" element={<Forge />} />
          <Route path="/miner" element={<Miner />} />
          <Route path="/library" element={<Library />} />
          <Route path="/brain" element={<BrainPage />} />
          <Route path="/import" element={<ImportPage />} />
          <Route path="/explorer" element={<Explorer />} />
          <Route path="/data" element={<DataPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<Dashboard />} />
        </Routes>
      </Suspense>
      <CommandPalette />
    </Layout>
  );
}
