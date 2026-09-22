import { useEffect, useState } from "react";
import { deleteDocument } from "../api/client";
import { fetchDocuments } from "../api/client";
import type { ChatDocument } from "../types";

export function DocumentsPanel() {
  const [docs, setDocs] = useState<ChatDocument[]>([]);
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    fetchDocuments().then(setDocs).catch(console.error);
  }, []);

  const handleDeleteRequest = (id: string) => {
    setPendingDelete(id);
    setNotice(null);
  };

  const handleDeleteConfirm = async (id: string) => {
    setPendingDelete(null);
    setDeleting(id);
    try {
      const result = await deleteDocument(id);
      setDocs((prev) => prev.filter((d) => d.id !== id));
      const parts = [`Deleted "${result.filename}"`];
      if (result.sessions_deleted > 0)
        parts.push(`${result.sessions_deleted} chat session${result.sessions_deleted !== 1 ? "s" : ""} removed`);
      setNotice(parts.join(" · "));
      setTimeout(() => setNotice(null), 4000);
    } catch (e) {
      setNotice("Delete failed");
      setTimeout(() => setNotice(null), 3000);
    } finally {
      setDeleting(null);
    }
  };

  const handleCancel = () => setPendingDelete(null);

  if (!docs.length) return (
    <p className="text-xs text-gray-400 px-3 mt-2">No documents ingested yet.</p>
  );

  return (
    <div className="px-3 mt-2 space-y-1">
      {notice && (
        <p className="text-xs text-amber-600 bg-amber-50 rounded px-2 py-1">{notice}</p>
      )}
      {docs.map((d) => (
        <div key={d.id} className="text-xs text-gray-700">
          {pendingDelete === d.id ? (
            <div className="bg-red-50 rounded px-2 py-1 space-y-1">
              <p className="text-red-700 font-medium truncate" title={d.filename}>
                Delete &ldquo;{d.filename}&rdquo;?
              </p>
              <p className="text-red-500 opacity-80">
                Removes {d.chunk_count} vectors, the source file, and{" "}
                {typeof d.sessions_using === "number"
                  ? `${d.sessions_using} chat session${d.sessions_using === 1 ? "" : "s"}`
                  : "any chat sessions"}{" "}
                that used it.
              </p>
              <div className="flex gap-2 pt-0.5">
                <button
                  onClick={() => handleDeleteConfirm(d.id)}
                  className="text-red-600 font-semibold hover:text-red-800"
                >
                  Delete all
                </button>
                <button
                  onClick={handleCancel}
                  className="text-gray-400 hover:text-gray-600"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-1">
              <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${d.source_type === "legal" ? "bg-blue-500" : "bg-green-500"}`} />
              <span className="flex-1 truncate" title={d.filename}>{d.filename}</span>
              <span className="opacity-40 text-gray-400">{d.chunk_count}c</span>
              {deleting === d.id ? (
                <span className="opacity-40 text-gray-400 animate-pulse">…</span>
              ) : (
                <button
                  onClick={() => handleDeleteRequest(d.id)}
                  className="opacity-20 hover:opacity-60 text-gray-500"
                  title="Delete document"
                >
                  ✕
                </button>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
