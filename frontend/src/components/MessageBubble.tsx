import type { Message } from "../types";
import { DateCard } from "./DateCard";
import { ModelBadge } from "./ModelBadge";
import { SourceBadge } from "./SourceBadge";

interface Props {
  message: Message;
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div className={`max-w-[80%] ${isUser ? "items-end" : "items-start"} flex flex-col`}>
        <div
          className={`rounded-2xl px-4 py-3 text-sm whitespace-pre-wrap ${
            isUser
              ? "bg-blue-600 text-white rounded-br-sm"
              : "bg-gray-100 text-gray-900 rounded-bl-sm border border-gray-200"
          }`}
        >
          {message.content}
        </div>

        {!isUser && (
          <>
            <div className="mt-1 flex flex-wrap items-center gap-1">
              {message.route_path && (
                <ModelBadge routePath={message.route_path} modelUsed={message.model_used} />
              )}
              {message.sources?.map((s, i) => (
                <SourceBadge key={i} source={s} />
              ))}
            </div>

            {message.temporal_facts && <DateCard facts={message.temporal_facts} />}
          </>
        )}
      </div>
    </div>
  );
}
