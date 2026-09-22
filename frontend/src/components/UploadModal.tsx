import { useCallback, useState } from "react";
import { uploadDocument } from "../api/client";

interface Props {
  onClose: () => void;
}

export function UploadModal({ onClose }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [sourceType, setSourceType] = useState<"legal" | "app_docs">("legal");
  const [status, setStatus] = useState<"idle" | "uploading" | "done" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  }, []);

  const handleSubmit = async () => {
    if (!file) return;
    setStatus("uploading");
    try {
      await uploadDocument(file, sourceType);
      setStatus("done");
      setTimeout(onClose, 1500);
    } catch (e) {
      setStatus("error");
      setErrorMsg(String(e));
    }
  };

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-md p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold text-gray-800">Upload Document</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-xl">✕</button>
        </div>

        <div
          onDrop={handleDrop}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
            dragOver ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-gray-400"
          }`}
          onClick={() => document.getElementById("file-input")?.click()}
        >
          <input
            id="file-input"
            type="file"
            accept=".pdf,.md,.txt"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && setFile(e.target.files[0])}
          />
          {file ? (
            <p className="text-sm font-medium text-gray-700">{file.name}</p>
          ) : (
            <p className="text-sm text-gray-400">Drop a PDF or Markdown file here, or click to browse</p>
          )}
        </div>

        <div className="mt-4 flex gap-3">
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="radio"
              name="source_type"
              value="legal"
              checked={sourceType === "legal"}
              onChange={() => setSourceType("legal")}
            />
            <span className="text-blue-600 font-medium">Legal Document</span>
          </label>
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="radio"
              name="source_type"
              value="app_docs"
              checked={sourceType === "app_docs"}
              onChange={() => setSourceType("app_docs")}
            />
            <span className="text-green-600 font-medium">App Documentation</span>
          </label>
        </div>

        {status === "error" && (
          <p className="mt-2 text-xs text-red-600">{errorMsg}</p>
        )}
        {status === "done" && (
          <p className="mt-2 text-xs text-green-600">Upload started! Ingestion running in background.</p>
        )}

        <button
          onClick={handleSubmit}
          disabled={!file || status === "uploading"}
          className="mt-4 w-full py-2.5 bg-blue-600 text-white rounded-xl text-sm font-medium
                     hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {status === "uploading" ? "Uploading…" : "Upload"}
        </button>
      </div>
    </div>
  );
}
