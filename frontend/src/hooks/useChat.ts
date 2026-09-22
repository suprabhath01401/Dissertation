import { useCallback, useEffect, useRef, useState } from "react";
import { fetchConstraints, fetchMessages, streamChat } from "../api/client";
import type { Constraint, Message, Source, TemporalFacts } from "../types";

interface ChatState {
  messages: Message[];
  streamingContent: string;
  isStreaming: boolean;
  sources: Source[];
  routePath: string | null;
  constraints: Constraint[];
  temporalFacts: TemporalFacts | null;
}

export function useChat(activeSessionId: string | null, userId = "default") {
  const [state, setState] = useState<ChatState>({
    messages: [],
    streamingContent: "",
    isStreaming: false,
    sources: [],
    routePath: null,
    constraints: [],
    temporalFacts: null,
  });
  const abortRef = useRef<(() => void) | null>(null);

  // Load messages + constraints whenever the active session changes
  useEffect(() => {
    if (!activeSessionId) {
      setState((prev) => ({ ...prev, messages: [], constraints: [], streamingContent: "" }));
      return;
    }
    fetchMessages(activeSessionId)
      .then((msgs) => setState((prev) => ({ ...prev, messages: msgs })))
      .catch(console.error);
    fetchConstraints(activeSessionId)
      .then((cons) =>
        setState((prev) => ({
          ...prev,
          constraints: cons.map((c: any) => ({
            id: c.id,
            type: c.constraint_type,
            value: c.value,
            is_permanent: c.is_permanent,
          })),
        }))
      )
      .catch(console.error);
  }, [activeSessionId]);

  const sendMessage = useCallback(
    (text: string, sessionId: string | null, title: string, useMixtral = false) => {
      if (abortRef.current) abortRef.current();

      const userMsg: Message = {
        id: crypto.randomUUID(),
        session_id: sessionId ?? "",
        role: "user",
        content: text,
        created_at: new Date().toISOString(),
      };

      setState((prev) => ({
        ...prev,
        messages: [...prev.messages, userMsg],
        streamingContent: "",
        isStreaming: true,
        sources: [],
        routePath: null,
        temporalFacts: null,
      }));

      let accumulated = "";
      let msgSources: Source[] = [];
      let msgRoute: string | null = null;
      let msgTemporal: TemporalFacts | null = null;
      let newSessionId = sessionId;

      const abort = streamChat(sessionId, userId, text, title, useMixtral, (type, data) => {
        if (type === "token") {
          accumulated += (data as { text: string }).text;
          setState((prev) => ({ ...prev, streamingContent: accumulated }));
        } else if (type === "source") {
          msgSources = [...msgSources, data as Source];
          setState((prev) => ({ ...prev, sources: msgSources }));
        } else if (type === "route") {
          msgRoute = (data as { path: string }).path;
          setState((prev) => ({ ...prev, routePath: msgRoute }));
        } else if (type === "constraint") {
          const c = data as { id: string; type: string; value: string; is_permanent: boolean };
          setState((prev) => ({
            ...prev,
            constraints: [
              ...prev.constraints.filter((x) => x.id !== c.id),
              { id: c.id, type: c.type, value: c.value, is_permanent: c.is_permanent },
            ],
          }));
        } else if (type === "status") {
          setState((prev) => ({ ...prev, streamingContent: (data as { message: string }).message }));
        } else if (type === "temporal_facts") {
          msgTemporal = data as TemporalFacts;
          setState((prev) => ({ ...prev, temporalFacts: msgTemporal }));
        } else if (type === "saved") {
          newSessionId = (data as { session_id: string }).session_id;
        } else if (type === "done") {
          const assistantMsg: Message = {
            id: (data as { message_id?: string }).message_id ?? crypto.randomUUID(),
            session_id: newSessionId ?? "",
            role: "assistant",
            content: accumulated,
            model_used: msgRoute === "cluster" ? "mixtral:8x7b" : "llama3.1:8b",
            route_path: msgRoute ?? undefined,
            sources: msgSources,
            temporal_facts: msgTemporal ?? undefined,
            created_at: new Date().toISOString(),
          };
          setState((prev) => ({
            ...prev,
            messages: [...prev.messages, assistantMsg],
            streamingContent: "",
            isStreaming: false,
          }));
        } else if (type === "error") {
          setState((prev) => ({
            ...prev,
            streamingContent: `Error: ${(data as { message: string }).message}`,
            isStreaming: false,
          }));
        }
      });

      abortRef.current = abort;
    },
    [userId]
  );

  const removeConstraint = useCallback((constraintId: string) => {
    setState((prev) => ({
      ...prev,
      constraints: prev.constraints.filter((c) => c.id !== constraintId),
    }));
  }, []);

  return { ...state, sendMessage, removeConstraint };
}
