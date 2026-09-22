import { useState } from "react";
import { ChatWindow } from "./components/ChatWindow";
import { ConstraintStrip } from "./components/ConstraintStrip";
import { EvalDashboard } from "./components/EvalDashboard";
import { InputBar } from "./components/InputBar";
import { Sidebar } from "./components/Sidebar";
import { UploadModal } from "./components/UploadModal";
import { useChat } from "./hooks/useChat";
import { useSession } from "./hooks/useSession";

const USER_ID = "default";

export default function App() {
  const { sessions, activeSessionId, setActiveSessionId, newSession, removeSession, selectSession } =
    useSession(USER_ID);

  const { messages, streamingContent, isStreaming, sources, routePath, constraints, temporalFacts, sendMessage, removeConstraint } =
    useChat(activeSessionId, USER_ID);

  const [showUpload, setShowUpload] = useState(false);
  const [view, setView] = useState<"chat" | "eval">("chat");

  const handleSend = (text: string, useMixtral: boolean) => {
    const session = sessions.find((s) => s.id === activeSessionId);
    const title = session?.title ?? text.slice(0, 60);
    sendMessage(text, activeSessionId, title, useMixtral);
  };

  const handleNew = async () => {
    await newSession();
    setView("chat");
  };

  return (
    <div className="flex h-screen bg-white text-gray-900">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelect={(id) => { selectSession(id); setView("chat"); }}
        onNew={handleNew}
        onDelete={removeSession}
        onUpload={() => setShowUpload(true)}
        constraints={constraints}
        onRemoveConstraint={removeConstraint}
      />

      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Nav tabs */}
        <div className="border-b border-gray-200 px-4 flex gap-4 bg-white">
          {(["chat", "eval"] as const).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`py-3 text-sm font-medium border-b-2 transition-colors ${
                view === v ? "border-blue-600 text-blue-600" : "border-transparent text-gray-500 hover:text-gray-700"
              }`}
            >
              {v === "chat" ? "Chat" : "Evaluation"}
            </button>
          ))}
        </div>

        {view === "eval" ? (
          <EvalDashboard />
        ) : (
          <>
            <ChatWindow
              messages={messages}
              streamingContent={streamingContent}
              isStreaming={isStreaming}
            />
            <ConstraintStrip
              constraints={constraints.filter((c) => !c.is_permanent)}
              sessionId={activeSessionId}
              onRemove={removeConstraint}
            />
            <InputBar
              onSend={handleSend}
              onUpload={() => setShowUpload(true)}
              disabled={isStreaming}
            />
          </>
        )}
      </div>

      {showUpload && <UploadModal onClose={() => setShowUpload(false)} />}
    </div>
  );
}
