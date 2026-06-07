"use client";

import { useEffect, useState } from "react";
import { QueryLogEntry, SystemHealth, EvalHealth } from "@/lib/types";

export default function LogsPage() {
  const [logs, setLogs] = useState<QueryLogEntry[]>([]);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [evalHealth, setEvalHealth] = useState<EvalHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      const res = await fetch("/api/logs");
      const data = await res.json();
      setLogs(data.logs?.logs ?? []);
      setHealth(data.health ?? null);
      setEvalHealth(data.eval_health ?? null);
    } catch {
      // silently ignore polling errors
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="max-w-5xl mx-auto w-full px-4 py-6 flex flex-col gap-6">
      <div>
        <h1 className="text-xl font-semibold text-white">Query Observability</h1>
        <p className="text-gray-500 text-sm mt-1">Live query logs — refreshes every 5 seconds</p>
      </div>

      {/* System health metrics */}
      {health && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <MetricCard label="Total queries (24h)" value={health.total_queries.toString()} />
          <MetricCard
            label="Answer rate"
            value={`${health.answer_rate_pct.toFixed(1)}%`}
            color={health.answer_rate_pct >= 80 ? "green" : health.answer_rate_pct >= 50 ? "yellow" : "red"}
          />
          <MetricCard
            label="Avg latency"
            value={health.avg_latency_ms ? `${Math.round(health.avg_latency_ms)}ms` : "—"}
          />
          <MetricCard
            label="Avg confidence"
            value={health.avg_confidence ? `${(health.avg_confidence * 100).toFixed(0)}%` : "—"}
          />
        </div>
      )}

      {/* RAGAS evaluation metrics */}
      {evalHealth && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-600 uppercase tracking-wider">RAGAS Eval (24h)</span>
            <span className="text-xs text-gray-700">· {evalHealth.total_evaluated} queries evaluated</span>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <MetricCard
              label="Faithfulness — claims grounded in context"
              value={evalHealth.avg_faithfulness !== null ? evalHealth.avg_faithfulness.toFixed(3) : "—"}
              color={
                evalHealth.avg_faithfulness === null ? undefined :
                evalHealth.avg_faithfulness >= 0.9 ? "green" :
                evalHealth.avg_faithfulness >= 0.7 ? "yellow" : "red"
              }
            />
            <MetricCard
              label="Answer relevancy — answer addresses question"
              value={evalHealth.avg_answer_relevancy !== null ? evalHealth.avg_answer_relevancy.toFixed(3) : "—"}
              color={
                evalHealth.avg_answer_relevancy === null ? undefined :
                evalHealth.avg_answer_relevancy >= 0.9 ? "green" :
                evalHealth.avg_answer_relevancy >= 0.7 ? "yellow" : "red"
              }
            />
          </div>
        </div>
      )}

      {/* Logs table */}
      <div className="flex flex-col gap-2">
        {loading && (
          <div className="text-gray-600 text-sm">Loading...</div>
        )}
        {!loading && logs.length === 0 && (
          <div className="text-gray-600 text-sm">No queries yet. Ask something in the Chat tab.</div>
        )}
        {logs.map((log) => (
          <LogRow
            key={log.id}
            log={log}
            expanded={expanded === log.id}
            onToggle={() => setExpanded(expanded === log.id ? null : log.id)}
          />
        ))}
      </div>
    </div>
  );
}

function MetricCard({ label, value, color }: { label: string; value: string; color?: "green" | "yellow" | "red" }) {
  const colorClass =
    color === "green" ? "text-green-400" :
    color === "yellow" ? "text-yellow-400" :
    color === "red" ? "text-red-400" :
    "text-white";

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl px-4 py-3">
      <div className={`text-2xl font-semibold ${colorClass}`}>{value}</div>
      <div className="text-gray-500 text-xs mt-1">{label}</div>
    </div>
  );
}

function LogRow({ log, expanded, onToggle }: {
  log: QueryLogEntry;
  expanded: boolean;
  onToggle: () => void;
}) {
  const time = new Date(log.queried_at).toLocaleTimeString();

  return (
    <div className="border border-gray-800 rounded-xl bg-gray-900 overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full text-left px-4 py-3 flex items-center gap-3 hover:bg-gray-800/50 transition-colors"
      >
        {/* Status dot */}
        <span
          className={`w-2 h-2 rounded-full shrink-0 ${log.answered ? "bg-green-500" : "bg-yellow-500"}`}
        />

        {/* Question */}
        <span className="flex-1 text-sm text-gray-200 truncate">{log.query_text}</span>

        {/* Metadata */}
        <div className="flex items-center gap-4 text-xs text-gray-600 shrink-0">
          {log.confidence_score !== null && log.answered && (
            <span>{(log.confidence_score * 100).toFixed(0)}% conf</span>
          )}
          <span>{log.latency_total_ms}ms</span>
          <span>{time}</span>
          <span className="text-gray-700">{expanded ? "▲" : "▼"}</span>
        </div>
      </button>

      {expanded && (
        <div className="border-t border-gray-800 px-4 py-3 flex flex-col gap-3 text-sm">
          {log.answered ? (
            <>
              <div className="text-gray-300 leading-relaxed">{log.answer_text}</div>
              {log.citations.length > 0 && (
                <div className="flex flex-col gap-2">
                  <div className="text-xs text-gray-600 uppercase tracking-wider">Citations</div>
                  {log.citations.map((c, i) => (
                    <div key={i} className="bg-gray-800/50 rounded-lg px-3 py-2 text-xs text-gray-400">
                      <div className="text-gray-300 font-medium">{c.source_file}{c.page_number ? ` · p.${c.page_number}` : ""}</div>
                      {c.section_title && c.section_title !== "None" && <div className="text-gray-500">{c.section_title}</div>}
                      <div className="mt-1 italic">&ldquo;{c.excerpt}&rdquo;</div>
                    </div>
                  ))}
                </div>
              )}
              <div className="flex gap-4 text-xs text-gray-600">
                <span>RRF score: {log.top_rrf_score?.toFixed(4)}</span>
                <span>confidence: {log.confidence_score?.toFixed(2)}</span>
                <span>latency: {log.latency_total_ms}ms</span>
              </div>
            </>
          ) : (
            <div className="text-yellow-400/70 text-sm">
              Refused — no relevant content found (RRF score: {log.top_rrf_score?.toFixed(4) ?? "0"})
            </div>
          )}
        </div>
      )}
    </div>
  );
}
