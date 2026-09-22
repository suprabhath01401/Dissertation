"""Debug: show exactly which chunks are retrieved for a query."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backend.retrieval.hybrid_retriever import retrieve

query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the legal basis for this amicus curiae submission?"

print(f"\nQuery: {query}\n{'='*70}")
results = retrieve(query)
for i, r in enumerate(results, 1):
    print(f"\n[{i}] {r['filename']}  page={r['page']}  score={r['score']:.4f}")
    print("-" * 60)
    print(r["text"][:600])
