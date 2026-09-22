import { useEffect, useRef } from "react";
import type { Message } from "../types";
import { MessageBubble } from "./MessageBubble";

interface Props {
  messages: Message[];
  streamingContent: string;
  isStreaming: boolean;
}

export function ChatWindow({ messages, streamingContent, isStreaming }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4">
      {messages.length === 0 && !isStreaming && (
        <div className="h-full flex items-center justify-center text-gray-400 text-sm">
          <p>Ask a question about your legal documents or app documentation.</p>
        </div>
      )}

      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}

      {isStreaming && streamingContent && (
        <div className="flex justify-start mb-4">
          <div className="max-w-[80%] bg-gray-100 text-gray-900 rounded-2xl rounded-bl-sm border border-gray-200 px-4 py-3 text-sm whitespace-pre-wrap">
            {streamingContent}
            <span className="inline-block w-2 h-4 bg-gray-400 ml-0.5 animate-pulse" />
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
