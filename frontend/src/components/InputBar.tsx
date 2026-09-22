import { useRef, useState } from "react";

interface Props {
  onSend: (text: string, useMixtral: boolean) => void;
  onUpload: () => void;
  disabled?: boolean;
}

export function InputBar({ onSend, onUpload, disabled }: Props) {
  const [text, setText] = useState("");
  const [useMixtral, setUseMixtral] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed, useMixtral);
    setText("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  return (
    <div className="border-t border-gray-200 bg-white">
      {/* Model selector */}
      <div className="flex items-center gap-2 px-3 pt-2">
        <span className="text-xs text-gray-400">Model:</span>
        <button
          onClick={() => setUseMixtral(false)}
          className={`px-2.5 py-0.5 rounded-full text-xs font-medium transition-colors ${
            !useMixtral
              ? "bg-blue-100 text-blue-700"
              : "text-gray-400 hover:text-gray-600"
          }`}
        >
          Local (llama3.1)
        </button>
        <button
          onClick={() => setUseMixtral(true)}
          className={`px-2.5 py-0.5 rounded-full text-xs font-medium transition-colors ${
            useMixtral
              ? "bg-purple-100 text-purple-700"
              : "text-gray-400 hover:text-gray-600"
          }`}
        >
          Mixtral 8×7B
        </button>
        {useMixtral && (
          <span className="text-xs text-amber-600 bg-amber-50 rounded px-2 py-0.5">
            Cluster · responses take 5–10 min
          </span>
        )}
      </div>

      {/* Input row */}
      <div className="flex items-end gap-2 p-3">
        <button
          onClick={onUpload}
          className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
          title="Upload document"
        >
          📎
        </button>

        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            e.target.style.height = "auto";
            e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`;
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          placeholder={useMixtral ? "Ask a question (Mixtral — slow but powerful)…" : "Ask a legal question…"}
          disabled={disabled}
          rows={1}
          className={`flex-1 resize-none rounded-xl border px-4 py-2.5 text-sm
                     focus:outline-none focus:ring-2 disabled:opacity-50
                     overflow-hidden min-h-[42px] ${
                       useMixtral
                         ? "border-purple-300 focus:ring-purple-400"
                         : "border-gray-300 focus:ring-blue-500"
                     }`}
        />

        <button
          onClick={submit}
          disabled={!text.trim() || disabled}
          className={`px-4 py-2.5 text-white rounded-xl text-sm font-medium
                     disabled:opacity-40 disabled:cursor-not-allowed transition-colors ${
                       useMixtral
                         ? "bg-purple-600 hover:bg-purple-700"
                         : "bg-blue-600 hover:bg-blue-700"
                     }`}
        >
          {disabled && useMixtral ? "Waiting…" : "Send"}
        </button>
      </div>
    </div>
  );
}
