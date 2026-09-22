export interface Source {
  source_type: "legal" | "app_docs";
  filename: string;
  page: number | string;
  chunk_index: number;
  sac_summary?: string;
  // The retrieved chunk's raw text, sent by the backend's "source" SSE event
  // and persisted on stored messages (see backend/retrieval/hybrid_retriever.py's
  // retrieve() and backend/main.py's chat stream). Used for the expanded
  // citation view.
  text?: string;
  score: number;
}

export interface Constraint {
  id: string;
  type: string;
  value: string;
  is_permanent: boolean;
}

// Real shape of a "temporal_facts" SSE payload / stored Message.temporal_facts,
// per backend/temporal/logic_circuits.py's run_temporal_pipeline() and
// backend/temporal/parser.py's TemporalData model.
export interface TemporalExplicitDate {
  date: string; // ISO-8601
  context: string;
}

export interface TemporalTriggerEvent {
  event: string;
  date: string | null; // ISO-8601 or null if unknown
}

export interface TemporalParallelResult {
  earliest: string | null; // ISO-8601
  latest: string | null; // ISO-8601
  span_days: number;
  conflicts: string[];
}

export interface TemporalSeriesResult {
  trigger_event: string | null;
  trigger_date: string | null; // ISO-8601
  deadline: string | null; // ISO-8601
  rule_applied: string; // e.g. "+30 days"
  error?: string; // present when no trigger event with a known date was found
}

export interface TemporalFacts {
  parallel: TemporalParallelResult;
  series: TemporalSeriesResult;
  relative_refs: string[];
  explicit_dates: TemporalExplicitDate[];
  trigger_events: TemporalTriggerEvent[];
}

export interface Message {
  id: string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  model_used?: string;
  route_path?: string;
  sources?: Source[];
  temporal_facts?: TemporalFacts;
  created_at: string;
}

export interface Session {
  id: string;
  user_id: string;
  title?: string;
  summary?: string;
  created_at: string;
  updated_at: string;
  is_active: boolean;
}

export interface ChatDocument {
  id: string;
  filename: string;
  source_type: string;
  file_path: string;
  chunk_count: number;
  ingested_at: string;
  // Count of distinct chat sessions whose messages cite this document, so the
  // pre-delete confirmation can warn "N chat sessions" up front. Optional
  // because it depends on a backend field that may not always be present.
  sessions_using?: number;
}

export interface EvalResult {
  id: string;
  dataset: string;
  config: string;
  metric: string;
  score: number;
  run_at: string;
  notes?: string;
}

export interface EvalRun {
  id: string;
  run_id: string;
  dataset: string;
  config: string;
  metric: string;
  score: number;
  n_samples: number;
  run_at: string;
  notes?: string;
}

export interface HealthStatus {
  qdrant: string;
  postgres: string;
  ollama: string;
  cluster: string;
}

export interface StreamEvent {
  type: "token" | "source" | "route" | "constraint" | "temporal_facts" | "saved" | "done" | "error";
  data: unknown;
}
