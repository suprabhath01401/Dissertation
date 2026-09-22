import type { ChatDocument, EvalResult, EvalRun, HealthStatus, Message, Session } from "../types";

const BASE = "/api";

export async function fetchSessions(userId = "default"): Promise<Session[]> {
  const r = await fetch(`${BASE}/sessions?user_id=${userId}`);
  return r.json();
}

export async function createSession(userId = "default", title = "New Chat"): Promise<Session> {
  const r = await fetch(`${BASE}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, title }),
  });
  return r.json();
}

export async function deleteSession(sessionId: string): Promise<void> {
  await fetch(`${BASE}/sessions/${sessionId}`, { method: "DELETE" });
}

export async function fetchMessages(sessionId: string): Promise<Message[]> {
  const r = await fetch(`${BASE}/sessions/${sessionId}/messages`);
  return r.json();
}

export async function fetchConstraints(sessionId: string) {
  const r = await fetch(`${BASE}/sessions/${sessionId}/constraints`);
  return r.json();
}

export async function deleteConstraint(sessionId: string, constraintId: string): Promise<void> {
  await fetch(`${BASE}/sessions/${sessionId}/constraints`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ constraint_id: constraintId }),
  });
}

export async function fetchDocuments(): Promise<ChatDocument[]> {
  const r = await fetch(`${BASE}/documents`);
  return r.json();
}

export interface DeleteDocumentResult {
  ok: boolean;
  filename: string;
  vectors_deleted: boolean;
  file_deleted: boolean;
  sessions_deleted: number;
}

export async function deleteDocument(docId: string): Promise<DeleteDocumentResult> {
  const r = await fetch(`${BASE}/documents/${docId}`, { method: "DELETE" });
  return r.json();
}

export async function uploadDocument(file: File, sourceType: string): Promise<{ filename: string; status: string }> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("source_type", sourceType);
  const r = await fetch(`${BASE}/upload`, { method: "POST", body: fd });
  return r.json();
}

export async function fetchEvalResults(limit?: number): Promise<EvalResult[]> {
  const qs = limit ? `?limit=${limit}` : "";
  const r = await fetch(`${BASE}/evaluation/results${qs}`);
  return r.json();
}

export async function fetchEvalRuns(): Promise<EvalRun[]> {
  const r = await fetch(`${BASE}/evaluation/runs`);
  return r.json();
}

export async function runEvaluation(
  dataset: string,
  config = "all",
  n?: number,
): Promise<{ status: string; dataset: string; config: string }> {
  const params = new URLSearchParams({ dataset, config });
  if (n && n > 0) params.set("n", String(n));
  const r = await fetch(`${BASE}/evaluation/run?${params}`, { method: "POST" });
  return r.json();
}

export async function fetchEvalStatus(): Promise<{ running: string[] }> {
  const r = await fetch(`${BASE}/evaluation/status`);
  return r.json();
}

export async function stopEvaluation(
  dataset: string,
  config: string,
): Promise<{ status: string; dataset: string; config: string }> {
  const r = await fetch(
    `${BASE}/evaluation/stop?dataset=${encodeURIComponent(dataset)}&config=${encodeURIComponent(config)}`,
    { method: "POST" },
  );
  return r.json();
}

export async function deleteEvalResults(
  dataset?: string,
): Promise<{ deleted: number; dataset: string }> {
  const qs = dataset ? `?dataset=${encodeURIComponent(dataset)}` : "";
  const r = await fetch(`${BASE}/evaluation/results${qs}`, { method: "DELETE" });
  return r.json();
}

export async function fetchHealth(): Promise<HealthStatus> {
  const r = await fetch(`${BASE}/health`);
  return r.json();
}

export function streamChat(
  sessionId: string | null,
  userId: string,
  message: string,
  title: string,
  useMixtral: boolean,
  onEvent: (type: string, data: unknown) => void
): () => void {
  const controller = new AbortController();

  fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, user_id: userId, message, title, use_mixtral: useMixtral }),
    signal: controller.signal,
  }).then(async (response) => {
    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop() ?? "";

      for (const part of parts) {
        const lines = part.trim().split("\n");
        let eventType = "message";
        let eventData = "";
        for (const line of lines) {
          if (line.startsWith("event: ")) eventType = line.slice(7).trim();
          if (line.startsWith("data: ")) eventData = line.slice(6).trim();
        }
        if (eventData) {
          try {
            onEvent(eventType, JSON.parse(eventData));
          } catch {
            onEvent(eventType, eventData);
          }
        }
      }
    }
  }).catch((err) => {
    if (err.name !== "AbortError") onEvent("error", { message: String(err) });
  });

  return () => controller.abort();
}
