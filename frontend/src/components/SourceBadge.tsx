import { useState } from "react";
import type { Source } from "../types";

interface Props {
  source: Source;
}

export function SourceBadge({ source }: Props) {
  const [expanded, setExpanded] = useState(false);
  const isLegal = source.source_type === "legal";
  const color = isLegal
    ? "bg-blue-100 text-blue-800 border-blue-300"
    : "bg-green-100 text-green-800 border-green-300";
  const ring = isLegal ? "focus:ring-blue-400" : "focus:ring-green-400";
  const label = isLegal ? "Legal" : "App Docs";

  return (
    <div className="inline-block mr-1 mb-1 align-top">
      <button
        type="button"
        onClick={() => setExpanded((e) => !e)}
        aria-expanded={expanded}
        aria-label={`Source: ${source.filename}, page ${source.page}. ${
          expanded ? "Collapse" : "Expand"
        } details.`}
        title={`${source.filename} — page ${source.page}`}
        className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs rounded border ${color}
                   cursor-pointer transition-colors transition-transform
                   hover:brightness-95 active:scale-95
                   focus:outline-none focus:ring-2 focus:ring-offset-1 ${ring}
                   ${expanded ? "ring-2 ring-offset-1 " + ring : ""}`}
      >
        <span className="font-semibold">{label}</span>
        <span className="opacity-70 truncate max-w-[120px]">{source.filename}</span>
        <span className="opacity-50">p{source.page}</span>
        <span className="opacity-50 ml-0.5">{expanded ? "▲" : "▼"}</span>
      </button>

      {expanded && (
        <div
          className={`mt-1 max-w-xs rounded border p-2 text-xs shadow-sm ${color}`}
        >
          <dl className="space-y-0.5">
            <div>
              <dt className="inline font-semibold">Document: </dt>
              <dd className="inline break-words">{source.filename}</dd>
            </div>
            <div>
              <dt className="inline font-semibold">Page: </dt>
              <dd className="inline">{source.page}</dd>
            </div>
            <div>
              <dt className="inline font-semibold">Type: </dt>
              <dd className="inline">{label}</dd>
            </div>
            <div>
              <dt className="inline font-semibold">Chunk #: </dt>
              <dd className="inline">{source.chunk_index}</dd>
            </div>
            {typeof source.score === "number" && (
              <div>
                <dt className="inline font-semibold">Relevance score: </dt>
                <dd className="inline">{source.score.toFixed(3)}</dd>
              </div>
            )}
          </dl>

          {source.text ? (
            <p className="mt-1.5 whitespace-pre-wrap opacity-90 border-t border-current pt-1.5">
              {source.text}
            </p>
          ) : source.sac_summary ? (
            <p className="mt-1.5 whitespace-pre-wrap opacity-90 border-t border-current pt-1.5">
              <span className="font-semibold">Summary: </span>
              {source.sac_summary}
            </p>
          ) : null}
        </div>
      )}
    </div>
  );
}
