interface Props {
  routePath: string;
  modelUsed?: string;
}

export function ModelBadge({ routePath, modelUsed }: Props) {
  const isCluster = routePath === "cluster";
  const isTemporal = routePath === "temporal";

  const color = isTemporal
    ? "bg-teal-100 text-teal-800 border-teal-300"
    : isCluster
    ? "bg-purple-100 text-purple-800 border-purple-300"
    : "bg-gray-100 text-gray-700 border-gray-300";

  const icon = isTemporal ? "📅" : isCluster ? "🖥" : "💻";

  const label = isTemporal ? "Temporal" : modelUsed ?? (isCluster ? "mixtral:8x7b" : "llama3.1:8b");

  const title = isTemporal
    ? modelUsed
      ? `Temporal reasoning path — answered via ${modelUsed}`
      : "Temporal reasoning path"
    : undefined;

  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs rounded border ${color}`}
      title={title}
    >
      <span>{icon}</span>
      <span>{label}</span>
    </span>
  );
}
