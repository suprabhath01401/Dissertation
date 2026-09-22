import { useEffect, useState } from "react";
import { fetchHealth } from "../api/client";
import type { Constraint, HealthStatus, Session } from "../types";
import { DocumentsPanel } from "./DocumentsPanel";
import { MemoryPanel } from "./MemoryPanel";

interface Props {
  sessions: Session[];
  activeSessionId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onUpload: () => void;
  constraints: Constraint[];
  onRemoveConstraint: (id: string) => void;
}

function StatusDot({ status }: { status: string }) {
  const color = status === "ok" ? "bg-green-400" : status === "unknown" ? "bg-yellow-400" : "bg-red-400";
  return <span className={`inline-block w-2 h-2 rounded-full ${color}`} />;
}

export function Sidebar({
  sessions, activeSessionId, onSelect, onNew, onDelete, onUpload, constraints, onRemoveConstraint,
}: Props) {
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    const check = () => fetchHealth().then(setHealth).catch(console.error);
    check();
    const id = setInterval(check, 30_000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="w-64 flex-shrink-0 flex flex-col border-r border-gray-200 bg-gray-50 h-screen">
      {/* Header */}
      <div className="p-3 border-b border-gray-200 flex items-center justify-between">
        <span className="font-semibold text-gray-800 text-sm">Legal RAG</span>
        <button
          onClick={onNew}
          className="text-xs bg-blue-600 text-white px-2.5 py-1 rounded-lg hover:bg-blue-700 transition-colors"
        >
          + New
        </button>
      </div>

      {/* Sessions */}
      <div className="flex-1 overflow-y-auto py-2">
        {sessions.map((s) => (
          <div
            key={s.id}
            className={`group flex items-center gap-1 px-3 py-2 cursor-pointer text-sm rounded-lg mx-1 ${
              activeSessionId === s.id ? "bg-blue-100 text-blue-900" : "hover:bg-gray-100 text-gray-700"
            }`}
            onClick={() => onSelect(s.id)}
          >
            <span className="flex-1 truncate">{s.title ?? "Chat"}</span>
            <button
              onClick={(e) => { e.stopPropagation(); onDelete(s.id); }}
              className="opacity-0 group-hover:opacity-40 hover:!opacity-80 text-gray-500 text-xs"
            >
              ✕
            </button>
          </div>
        ))}
      </div>

      {/* Documents */}
      <div className="border-t border-gray-200 pt-2 pb-1">
        <div className="flex items-center justify-between px-3 mb-1">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Documents</p>
          <button onClick={onUpload} className="text-xs text-blue-600 hover:underline">Upload</button>
        </div>
        <DocumentsPanel />
      </div>

      {/* Permanent constraints */}
      <MemoryPanel constraints={constraints} onRemove={onRemoveConstraint} />

      {/* Status bar */}
      {health && (
        <div className="border-t border-gray-200 px-3 py-2 flex flex-wrap gap-x-3 gap-y-1">
          {(["qdrant", "postgres", "ollama", "cluster"] as const).map((k) => (
            <span key={k} className="flex items-center gap-1 text-xs text-gray-500">
              <StatusDot status={health[k]} />
              {k}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
