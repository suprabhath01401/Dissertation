import { useCallback, useEffect, useRef, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import {
  deleteEvalResults,
  fetchEvalResults,
  fetchEvalRuns,
  fetchEvalStatus,
  runEvaluation,
  stopEvaluation,
} from "../api/client";
import type { EvalResult, EvalRun } from "../types";

const DATASETS = [
  "contractnli", "timeqa", "locomo", "locomoplus", "casehold", "cuad", "all",
];
const CONFIGS = [
  "vanilla_rag", "long_ctx_only", "self_route_base", "full_system", "mixtral_cluster",
];
const CONFIG_COLORS: Record<string, string> = {
  vanilla_rag:     "#94a3b8",
  long_ctx_only:   "#60a5fa",
  self_route_base: "#a78bfa",
  full_system:     "#34d399",
  mixtral_cluster: "#f97316",
};
const CONFIG_LABELS: Record<string, string> = {
  vanilla_rag:     "dense retrieval + plain prompt",
  long_ctx_only:   "15 chunks concatenated, no routing",
  self_route_base: "hybrid retrieval + routing, basic prompt",
  full_system:     "full pipeline (SAC + routing + expert prompt)",
  mixtral_cluster: "Mixtral 8×7B on Warwick GPU cluster",
};

// ── Metric metadata ──────────────────────────────────────────────────────────
// The current 13 metrics, across the 6 finalised benchmarks (locomo =
// factual memory, categories 1-5; locomoplus = cognitive memory, category
// 6/"cognitive" — two halves of what an earlier design merged into one
// "locomoplus" dataset, now split into separate benchmark names but both
// still judged by the same category-prompted judge_score function).
// macro_f1 and AUPR/precision_at_recall are dataset-level aggregates (one
// row per (dataset, config) run, not per sample) — they still show up as
// ordinary rows here, just far less frequently than the per-sample metrics.

const METRIC_META: Record<string, { label: string; benchmarks: string }> = {
  rouge_l:                   { label: "ROUGE-L",               benchmarks: "contractnli · timeqa · locomo · locomoplus · casehold · cuad (supplementary)" },
  accuracy:                  { label: "Accuracy",              benchmarks: "contractnli · casehold" },
  macro_f1:                  { label: "Macro-F1",              benchmarks: "casehold (primary, dataset-level aggregate)" },
  micro_f1:                  { label: "Micro-F1",              benchmarks: "contractnli (evidence spans)" },
  exact_match:               { label: "Exact Match",           benchmarks: "timeqa · cuad · locomoplus (supplementary)" },
  token_f1:                  { label: "Token F1",              benchmarks: "timeqa · cuad" },
  AUPR:                      { label: "AUPR",                  benchmarks: "cuad (primary, dataset-level aggregate)" },
  "precision_at_recall_r0.8": { label: "Precision @ R=0.8",    benchmarks: "cuad · contractnli (dataset-level aggregate)" },
  "precision_at_recall_r0.9": { label: "Precision @ R=0.9",    benchmarks: "cuad (dataset-level aggregate)" },
  judge_score:               { label: "LLM Judge Score",       benchmarks: "locomo (primary, factual memory) · locomoplus (cognitive memory)" },
  constraint_consistency:    { label: "Constraint Consistency",benchmarks: "locomoplus (cognitive memory, RQ2)" },
  temporal_consistency:      { label: "Temporal Consistency",  benchmarks: "timeqa (RQ3)" },
  perturbation_consistency:  { label: "Perturbation Consistency", benchmarks: "timeqa (primary, RQ3)" },
  retrieval_recall:          { label: "Retrieval Recall",      benchmarks: "contractnli" },
};

// ── Notes parsing ─────────────────────────────────────────────────────────────

function parseNotes(notes?: string): Record<string, string> {
  if (!notes) return {};
  try { return JSON.parse(notes); } catch { return {}; }
}

function NotesTag({ notes }: { notes?: string }) {
  const meta = parseNotes(notes);
  const entries = Object.entries(meta);
  if (!entries.length) return null;
  return (
    <span className="inline-flex flex-wrap gap-1">
      {entries.map(([k, v]) => (
        <span key={k} className="text-[10px] bg-gray-100 text-gray-500 rounded px-1 py-0.5">
          {k}: {String(v)}
        </span>
      ))}
    </span>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

export function EvalDashboard() {
  const [results, setResults]             = useState<EvalResult[]>([]);
  const [recentResults, setRecentResults] = useState<EvalResult[]>([]);
  const [runs, setRuns]                   = useState<EvalRun[]>([]);
  const [filterDataset, setFilterDataset] = useState("all");
  const [filterMetric, setFilterMetric]   = useState("all");
  const [runDataset, setRunDataset]       = useState("contractnli");
  const [runConfig, setRunConfig]         = useState("full_system");
  const [runN, setRunN]                   = useState<string>("");
  const [runningSet, setRunningSet]       = useState<string[]>([]);
  const [notice, setNotice]               = useState<string | null>(null);
  const [noticeKind, setNoticeKind]       = useState<"info" | "ok" | "err">("info");
  const [confirmDelete, setConfirmDelete] = useState(false);
  const wasRunningRef                     = useRef(false);

  const flash = (msg: string, kind: "info" | "ok" | "err" = "info") => {
    setNotice(msg); setNoticeKind(kind);
    setTimeout(() => setNotice(null), 6000);
  };

  const refresh = useCallback(() => {
    fetchEvalResults().then(setResults).catch(console.error);
    fetchEvalResults(20).then(setRecentResults).catch(console.error);
    fetchEvalRuns().then(setRuns).catch(console.error);
  }, []);

  // Single always-on poll every 5 s — directly reflects server truth
  // (spec 7d: "the frontend EvalDashboard polls /status every 5 seconds
  // unconditionally"). Detects running→idle transition to refresh results
  // and show a notice.
  useEffect(() => {
    refresh();
    fetchEvalStatus()
      .then(({ running }) => {
        setRunningSet(running);
        wasRunningRef.current = running.length > 0;
      })
      .catch(console.error);

    const interval = setInterval(() => {
      fetchEvalStatus()
        .then(({ running }) => {
          const isRunning = running.length > 0;
          setRunningSet(running);
          if (wasRunningRef.current && !isRunning) {
            refresh();
            flash("Benchmark complete — results updated.", "ok");
          }
          wasRunningRef.current = isRunning;
        })
        .catch(console.error);
    }, 5000);

    return () => clearInterval(interval);
  }, [refresh]);

  const handleRun = async () => {
    try {
      const n = runN ? parseInt(runN, 10) : undefined;
      const res = await runEvaluation(runDataset, runConfig, n);
      if (res.status === "already_running") {
        flash(`"${runDataset}:${runConfig}" is already running.`, "info");
      } else {
        setRunningSet((prev) => [...new Set([...prev, `${runDataset}:${runConfig}`])]);
        const nLabel = n ? ` (n=${n})` : "";
        flash(`Started "${runDataset}" / "${runConfig}"${nLabel}.`, "ok");
      }
    } catch (err) {
      flash(`Failed to start: ${String(err)}`, "err");
    }
  };

  const handleStop = async (dataset: string, config: string) => {
    await stopEvaluation(dataset, config);
    flash(`Stop requested for "${dataset}:${config}" — halts after current sample.`, "info");
  };

  const handleDelete = async () => {
    const target = filterDataset === "all" ? undefined : filterDataset;
    const label  = target ?? "all datasets";
    try {
      const { deleted } = await deleteEvalResults(target);
      setConfirmDelete(false);
      refresh();
      flash(`Deleted ${deleted} result rows for ${label}.`, "ok");
    } catch {
      flash("Delete failed — check the server logs.", "err");
    }
  };

  // ── Derive filter options ──────────────────────────────────────────────────
  const availableDatasets = ["all", ...new Set(results.map((r) => r.dataset))];
  const availableMetrics  = ["all", ...new Set(results.map((r) => r.metric))];

  const filtered = results.filter(
    (r) =>
      (filterDataset === "all" || r.dataset === filterDataset) &&
      (filterMetric  === "all" || r.metric  === filterMetric),
  );

  // Latest 20 raw samples only — for spot-checking, not for the chart's averages.
  const filteredRecent = recentResults.filter(
    (r) =>
      (filterDataset === "all" || r.dataset === filterDataset) &&
      (filterMetric  === "all" || r.metric  === filterMetric),
  );

  // Pivot for chart: metric → { config: avgScore }
  const pivot: Record<string, Record<string, { sum: number; n: number }>> = {};
  for (const r of filtered) {
    if (!pivot[r.metric]) pivot[r.metric] = {};
    const b = pivot[r.metric][r.config] ?? { sum: 0, n: 0 };
    b.sum += r.score; b.n += 1;
    pivot[r.metric][r.config] = b;
  }
  const chartData = Object.entries(pivot).map(([metric, byConfig]) => {
    const row: Record<string, string | number> = {
      metric: METRIC_META[metric]?.label ?? metric,
    };
    for (const [cfg, { sum, n }] of Object.entries(byConfig))
      row[cfg] = parseFloat((sum / n).toFixed(4));
    return row;
  });

  // ── Run summary: one row per (run, dataset, config, metric) — final score only ──
  const filteredRuns = runs.filter(
    (r) =>
      (filterDataset === "all" || r.dataset === filterDataset) &&
      (filterMetric  === "all" || r.metric  === filterMetric),
  );
  const runOrder = [...new Set(
    [...runs].sort((a, b) => a.run_at.localeCompare(b.run_at)).map((r) => r.run_id),
  )];
  const runNumber = Object.fromEntries(runOrder.map((id, i) => [id, i + 1]));
  const sortedRuns = [...filteredRuns].sort((a, b) => {
    if (a.run_id !== b.run_id) return b.run_at.localeCompare(a.run_at);
    if (a.dataset !== b.dataset) return a.dataset.localeCompare(b.dataset);
    if (a.config !== b.config) return a.config.localeCompare(b.config);
    return a.metric.localeCompare(b.metric);
  });

  const anyRunning = runningSet.length > 0;

  return (
    <div className="flex-1 overflow-y-auto p-6 max-w-6xl mx-auto">
      <h1 className="text-2xl font-bold text-gray-800 mb-1">Evaluation Dashboard</h1>
      <p className="text-sm text-gray-500 mb-6">
        6 benchmarks × 5 RAG configurations. All scores 0–1; higher is better.
      </p>

      {/* ── Run controls ──────────────────────────────────────────────────── */}
      <div className="bg-gray-50 border border-gray-200 rounded-xl p-4 mb-3 flex flex-wrap items-end gap-3">
        <div>
          <label className="text-xs font-medium text-gray-500 block mb-1">Dataset</label>
          <select
            value={runDataset}
            onChange={(e) => setRunDataset(e.target.value)}
            className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm"
          >
            {DATASETS.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>
        <div>
          <label className="text-xs font-medium text-gray-500 block mb-1">Config</label>
          <select
            value={runConfig}
            onChange={(e) => setRunConfig(e.target.value)}
            className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm"
          >
            <option value="all">all configs</option>
            {CONFIGS.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <div>
          <label className="text-xs font-medium text-gray-500 block mb-1">Max samples (n)</label>
          <input
            type="number"
            min={1}
            placeholder="all"
            value={runN}
            onChange={(e) => setRunN(e.target.value)}
            className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm w-24"
          />
        </div>
        <button
          onClick={handleRun}
          className="px-4 py-1.5 rounded-lg text-sm font-medium bg-emerald-600 text-white hover:bg-emerald-700"
        >
          Run Evaluation
        </button>
        {notice && (
          <p className={`text-xs rounded px-3 py-1.5 ${
            noticeKind === "ok"  ? "text-emerald-700 bg-emerald-50 border border-emerald-200" :
            noticeKind === "err" ? "text-red-700 bg-red-50 border border-red-200" :
                                   "text-blue-700 bg-blue-50 border border-blue-200"
          }`}>
            {notice}
          </p>
        )}
      </div>

      {/* ── Persistent status bar ─────────────────────────────────────────── */}
      <div className={`rounded-xl border px-4 py-3 mb-4 text-sm transition-colors ${
        anyRunning
          ? "bg-amber-50 border-amber-300 text-amber-900"
          : "bg-gray-50 border-gray-200 text-gray-400"
      }`}>
        {anyRunning ? (
          <div className="flex flex-wrap items-center gap-4">
            <span className="flex items-center gap-2 font-semibold">
              <svg className="animate-spin w-4 h-4 shrink-0" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
              </svg>
              Running evaluations:
            </span>
            {runningSet.map((key) => {
              const colonIdx = key.indexOf(":");
              const ds  = key.slice(0, colonIdx);
              const cfg = key.slice(colonIdx + 1);
              return (
                <span key={key} className="flex items-center gap-2 bg-amber-100 rounded-lg px-2.5 py-1">
                  <span className="font-mono text-xs font-medium">{key}</span>
                  <button
                    onClick={() => handleStop(ds, cfg)}
                    className="text-xs font-medium text-red-600 hover:text-red-800 border border-red-300 rounded px-1.5 py-0.5 hover:bg-red-50"
                  >
                    Stop
                  </button>
                </span>
              );
            })}
          </div>
        ) : (
          <span>No evaluations running — select a dataset and config above, then click Run.</span>
        )}
      </div>

      {/* ── Filter + Bulk delete ───────────────────────────────────────────── */}
      <div className="flex flex-wrap gap-4 items-end mb-6">
        <div>
          <label className="text-sm font-medium text-gray-600 block mb-1">Filter dataset</label>
          <select
            value={filterDataset}
            onChange={(e) => { setFilterDataset(e.target.value); setConfirmDelete(false); }}
            className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm"
          >
            {availableDatasets.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>
        <div>
          <label className="text-sm font-medium text-gray-600 block mb-1">Filter metric</label>
          <select
            value={filterMetric}
            onChange={(e) => setFilterMetric(e.target.value)}
            className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm"
          >
            {availableMetrics.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>

        {/* Bulk delete */}
        <div className="ml-auto flex items-center gap-2">
          {!confirmDelete ? (
            <button
              onClick={() => setConfirmDelete(true)}
              disabled={filtered.length === 0}
              className="px-3 py-1.5 rounded-lg text-sm font-medium text-red-600 border border-red-200 hover:bg-red-50 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Clear Results
            </button>
          ) : (
            <>
              <span className="text-xs text-red-600">
                Delete {filterDataset === "all" ? "ALL" : `"${filterDataset}"`} results?
              </span>
              <button
                onClick={handleDelete}
                className="px-3 py-1.5 rounded-lg text-sm font-medium bg-red-600 text-white hover:bg-red-700"
              >
                Yes, delete
              </button>
              <button
                onClick={() => setConfirmDelete(false)}
                className="px-3 py-1.5 rounded-lg text-sm font-medium text-gray-600 border border-gray-300 hover:bg-gray-50"
              >
                Cancel
              </button>
            </>
          )}
        </div>
      </div>

      {/* ── Chart ─────────────────────────────────────────────────────────── */}
      {chartData.length === 0 ? (
        <div className="text-center py-16 text-gray-400 border border-dashed border-gray-200 rounded-2xl mb-6">
          <p className="text-lg font-medium mb-2">No results yet</p>
          <p className="text-sm">
            Select a dataset and config above, then click <strong>Run Evaluation</strong>.
          </p>
        </div>
      ) : (
        <div className="bg-white rounded-2xl border border-gray-200 p-6 mb-6">
          <ResponsiveContainer width="100%" height={380}>
            <BarChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 50 }}>
              <XAxis
                dataKey="metric"
                tick={{ fontSize: 11 }}
                angle={-30}
                textAnchor="end"
                interval={0}
              />
              <YAxis
                domain={[0, 1]}
                tickFormatter={(v: number) => v.toFixed(2)}
                tick={{ fontSize: 11 }}
              />
              <Tooltip formatter={(v) => typeof v === "number" ? v.toFixed(4) : v} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {CONFIGS.map((cfg) => (
                <Bar key={cfg} dataKey={cfg} fill={CONFIG_COLORS[cfg]} radius={[4, 4, 0, 0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ── Legend ────────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-4 mb-6 text-xs text-gray-600">
        <div className="bg-gray-50 rounded-xl p-4">
          <p className="font-semibold mb-2 text-gray-800">Configurations</p>
          <ul className="space-y-1">
            {CONFIGS.map((cfg) => (
              <li key={cfg} className="flex items-center gap-1.5">
                <span className="inline-block w-2 h-2 rounded-full shrink-0"
                  style={{ backgroundColor: CONFIG_COLORS[cfg] }} />
                <span className="font-mono text-[11px]">{cfg}</span>
                <span className="text-gray-400">—</span>
                {CONFIG_LABELS[cfg]}
              </li>
            ))}
          </ul>
        </div>

        <div className="bg-gray-50 rounded-xl p-4">
          <p className="font-semibold mb-2 text-gray-800">Metrics by benchmark</p>
          <ul className="space-y-0.5">
            {Object.entries(METRIC_META).map(([key, { label, benchmarks }]) => (
              <li key={key}>
                <strong>{label}</strong>
                <span className="text-gray-400"> — {benchmarks}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* ── Run summary table — final metric per run, not per data point ───── */}
      {sortedRuns.length > 0 && (
        <div className="mb-8">
          <h2 className="text-sm font-semibold text-gray-800 mb-1">Run Summary</h2>
          <p className="text-xs text-gray-500 mb-2">
            One row per finished run — the mean score across that run's samples, not each individual data point.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left border-collapse">
              <thead>
                <tr className="text-gray-500 border-b border-gray-200">
                  {["Run", "Dataset", "Config", "Metric", "Final Score", "Samples", "Finished At"].map((h) => (
                    <th key={h} className="py-2 px-3 font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sortedRuns.map((r, i) => {
                  const newRunGroup = i === 0 || sortedRuns[i - 1].run_id !== r.run_id;
                  return (
                    <tr
                      key={r.id}
                      className={`border-b border-gray-100 hover:bg-gray-50 ${
                        newRunGroup ? "border-t-2 border-t-gray-200" : ""
                      }`}
                    >
                      <td className="py-2 px-3 text-gray-400 text-xs font-mono">
                        {newRunGroup ? `#${runNumber[r.run_id]}` : ""}
                      </td>
                      <td className="py-2 px-3 text-gray-700">{r.dataset}</td>
                      <td className="py-2 px-3">
                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                          r.config === "full_system"
                            ? "bg-emerald-100 text-emerald-800"
                            : "bg-gray-100 text-gray-700"
                        }`}>
                          {r.config}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-gray-700">
                        {METRIC_META[r.metric]?.label ?? r.metric}
                      </td>
                      <td className="py-2 px-3 font-mono font-semibold text-gray-900">
                        {r.score.toFixed(4)}
                      </td>
                      <td className="py-2 px-3 text-gray-500">{r.n_samples}</td>
                      <td className="py-2 px-3 text-gray-400 text-xs">
                        {new Date(r.run_at).toLocaleString()}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="text-xs text-gray-400 mt-2">{sortedRuns.length} rows across {runOrder.length} run(s)</p>
          </div>
        </div>
      )}

      {/* ── Per-data-point results table (latest 20 only) ──────────────────── */}
      {filteredRecent.length > 0 && (
        <div className="overflow-x-auto">
          <h2 className="text-sm font-semibold text-gray-800 mb-1">Latest Data Points</h2>
          <p className="text-xs text-gray-500 mb-2">Most recent individual sample scores — for spot-checking, not for reporting.</p>
          <table className="w-full text-sm text-left border-collapse">
            <thead>
              <tr className="text-gray-500 border-b border-gray-200">
                {["Dataset", "Config", "Metric", "Score", "Details", "Run At"].map((h) => (
                  <th key={h} className="py-2 px-3 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredRecent.map((r) => (
                <tr key={r.id} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="py-2 px-3 text-gray-700">{r.dataset}</td>
                  <td className="py-2 px-3">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                      r.config === "full_system"
                        ? "bg-emerald-100 text-emerald-800"
                        : "bg-gray-100 text-gray-700"
                    }`}>
                      {r.config}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-gray-700">
                    {METRIC_META[r.metric]?.label ?? r.metric}
                  </td>
                  <td className="py-2 px-3 font-mono font-medium text-gray-900">
                    {r.score.toFixed(4)}
                  </td>
                  <td className="py-2 px-3">
                    <NotesTag notes={r.notes} />
                  </td>
                  <td className="py-2 px-3 text-gray-400 text-xs">
                    {new Date(r.run_at).toLocaleDateString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-xs text-gray-400 mt-2">{filteredRecent.length} rows (latest 20 fetched)</p>
        </div>
      )}
    </div>
  );
}
