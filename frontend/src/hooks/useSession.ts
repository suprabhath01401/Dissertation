import { useCallback, useEffect, useState } from "react";
import { createSession, deleteSession, fetchSessions } from "../api/client";
import type { Session } from "../types";

export function useSession(userId = "default") {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  useEffect(() => {
    fetchSessions(userId).then(setSessions).catch(console.error);
  }, [userId]);

  const newSession = useCallback(async (title = "New Chat") => {
    const s = await createSession(userId, title);
    setSessions((prev) => [s, ...prev]);
    setActiveSessionId(s.id);
    return s;
  }, [userId]);

  const removeSession = useCallback(async (id: string) => {
    await deleteSession(id);
    setSessions((prev) => prev.filter((s) => s.id !== id));
    if (activeSessionId === id) setActiveSessionId(null);
  }, [activeSessionId]);

  const selectSession = useCallback((id: string) => {
    setActiveSessionId(id);
  }, []);

  return { sessions, activeSessionId, setActiveSessionId, newSession, removeSession, selectSession };
}
