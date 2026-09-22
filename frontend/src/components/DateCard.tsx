import type { TemporalFacts } from "../types";

interface Props {
  facts: TemporalFacts;
}

/** Render an ISO-8601 date string as a friendlier locale date; fall back to the raw string if unparsable. */
function formatDate(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" });
}

/**
 * Dedicated date/temporal card, rendered separately from the answer text.
 * Summarises the parallel_circuit (earliest/latest + conflicts) and
 * series_circuit (computed deadline) outputs from the backend's temporal
 * reasoning sub-system, plus the raw explicit dates found.
 */
export function DateCard({ facts }: Props) {
  const parallel = facts?.parallel;
  const series = facts?.series;
  const explicitDates = facts?.explicit_dates ?? [];

  const hasEarliestOrLatest = Boolean(parallel?.earliest || parallel?.latest);
  const hasDeadline = Boolean(series?.deadline);
  const hasTrigger = Boolean(series?.trigger_event);
  const conflicts = parallel?.conflicts ?? [];
  const hasConflicts = conflicts.length > 0;
  const hasExplicitDates = explicitDates.length > 0;

  // Nothing substantive to show — don't render an empty shell.
  if (!hasEarliestOrLatest && !hasDeadline && !hasTrigger && !hasConflicts && !hasExplicitDates) {
    return null;
  }

  return (
    <div className="mt-2 w-full max-w-sm rounded-lg border border-teal-300 bg-teal-50 px-3 py-2 text-xs text-teal-900">
      <div className="mb-1 flex items-center gap-1 font-semibold text-teal-800">
        <span>📅</span>
        <span>Temporal analysis</span>
      </div>

      <div className="space-y-0.5">
        {parallel?.earliest && (
          <p>
            <span className="font-medium">Earliest date:</span> {formatDate(parallel.earliest)}
          </p>
        )}
        {parallel?.latest && (
          <p>
            <span className="font-medium">Latest date:</span> {formatDate(parallel.latest)}
          </p>
        )}
        {hasTrigger && (
          <p>
            <span className="font-medium">Trigger event:</span> {series!.trigger_event}
            {series?.trigger_date ? ` — ${formatDate(series.trigger_date)}` : ""}
          </p>
        )}
        {hasDeadline && (
          <p>
            <span className="font-medium">Computed deadline:</span> {formatDate(series!.deadline)}
            {series?.rule_applied && <span className="opacity-70"> ({series.rule_applied})</span>}
          </p>
        )}
      </div>

      {hasConflicts && (
        <div className="mt-1.5 rounded border border-red-300 bg-red-50 px-2 py-1 text-red-700">
          <p className="font-medium">⚠ Conflict detected</p>
          <ul className="list-inside list-disc">
            {conflicts.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        </div>
      )}

      {hasExplicitDates && (
        <div className="mt-1.5 border-t border-teal-200 pt-1.5">
          <p className="mb-0.5 font-medium">Dates found in context:</p>
          <ul className="space-y-0.5">
            {explicitDates.map((d, i) => (
              <li key={i}>
                <span className="font-mono opacity-80">{formatDate(d.date) ?? d.date}</span>
                {d.context && <span className="opacity-70"> — {d.context}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
