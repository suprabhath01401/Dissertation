import type { Constraint } from "../types";

interface Props {
  constraints: Constraint[];
  onRemove: (id: string) => void;
}

export function MemoryPanel({ constraints, onRemove }: Props) {
  const permanent = constraints.filter((c) => c.is_permanent);

  if (!permanent.length) return null;

  return (
    <div className="mt-auto border-t border-gray-200 pt-3 pb-1 px-3">
      <p className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wide">
        Permanent Memory
      </p>
      {permanent.map((c) => (
        <div key={c.id} className="flex items-start gap-1 text-xs text-gray-700 mb-1">
          <span className="text-orange-500">★</span>
          <span className="flex-1">
            <span className="font-medium">{c.type}:</span> {c.value}
          </span>
          <button
            onClick={() => onRemove(c.id)}
            className="opacity-30 hover:opacity-70 text-gray-500 ml-1"
          >
            ✕
          </button>
        </div>
      ))}
    </div>
  );
}
