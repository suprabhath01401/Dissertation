import type { Constraint } from "../types";

interface Props {
  constraints: Constraint[];
  sessionId: string | null;
  onRemove: (id: string) => void;
}

export function ConstraintStrip({ constraints, sessionId, onRemove }: Props) {
  if (!constraints.length) return null;

  return (
    <div className="flex flex-wrap gap-1 px-4 py-2 bg-amber-50 border-t border-amber-100">
      {constraints.map((c) => (
        <span
          key={c.id}
          className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs rounded-full border ${
            c.is_permanent
              ? "bg-orange-100 text-orange-800 border-orange-300"
              : "bg-yellow-100 text-yellow-800 border-yellow-300"
          }`}
        >
          <span className="font-medium">{c.type}:</span>
          <span>{c.value}</span>
          {c.is_permanent && <span title="Permanent" className="opacity-60">★</span>}
          <button
            onClick={() => onRemove(c.id)}
            className="ml-0.5 opacity-50 hover:opacity-100 text-xs leading-none"
            title="Remove constraint"
          >
            ✕
          </button>
        </span>
      ))}
    </div>
  );
}
