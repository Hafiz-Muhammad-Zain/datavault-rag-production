"use client";

import { useEffect, useState } from "react";
import {
  LineChart, Line, BarChart, Bar, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
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

  // Build chart data from logs (chronological order)
  const chartData = [...logs]
    .reverse()
    .filter((l) => l.answered)
    .map((l, i) => ({
      index: i + 1,
      time: new Date(l.queried_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      latency: l.latency_total_ms,
      confidence: l.confidence_score !== null ? Math.round(l.confidence_score * 100) : null,
    }));

  const rasgasData = evalHealth
    ? [
        { metric: "Faithfulness", score: evalHealth.avg_faithfulness ?? 0 },
        { metric: "Answer Relevancy", score: evalHealth.avg_answer_relevancy ?? 0 },
      ]
    : [];

  return (
    <div className="max-w-5xl mx-auto w-full px-4 py-6 flex flex-col gap-8">
      <div>
        <h1 className="text-xl font-semibold text-white">Query Observability</h1>
        <p className="text-gray-500 text-sm mt-1">Live query logs — refreshes every 5 seconds</p>
      </div>

      {/* Top metric cards */}
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

      {/* Charts row */}
      {chartData.length >= 2 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {/* Latency trend */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl px-4 pt-4 pb-2">
            <div className="text-xs text-gray-500 uppercase tracking-wider mb-3">Response latency (ms)</div>
            <ResponsiveContainer width="100%" height={160}>
              <AreaChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="latencyGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="time" tick={{ fill: "#4b5563", fontSize: 10 }} />
                <YAxis tick={{ fill: "#4b5563", fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ background: "#111827", border: "1px solid #1f2937", borderRadius: 8 }}
                  labelStyle={{ color: "#9ca3af" }}
                  itemStyle={{ color: "#a5b4fc" }}
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  formatter={(v: any) => [`${v}ms`, "Latency"]}
                />
                <Area type="monotone" dataKey="latency" stroke="#6366f1" strokeWidth={2} fill="url(#latencyGrad)" dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Confidence trend */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl px-4 pt-4 pb-2">
            <div className="text-xs text-gray-500 uppercase tracking-wider mb-3">Answer confidence (%)</div>
            <ResponsiveContainer width="100%" height={160}>
              <LineChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="time" tick={{ fill: "#4b5563", fontSize: 10 }} />
                <YAxis domain={[0, 100]} tick={{ fill: "#4b5563", fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ background: "#111827", border: "1px solid #1f2937", borderRadius: 8 }}
                  labelStyle={{ color: "#9ca3af" }}
                  itemStyle={{ color: "#34d399" }}
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  formatter={(v: any) => [`${v}%`, "Confidence"]}
                />
                <Line type="monotone" dataKey="confidence" stroke="#34d399" strokeWidth={2} dot={{ fill: "#34d399", r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* RAGAS eval section */}
      {evalHealth && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl px-4 pt-4 pb-4">
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="text-xs text-gray-500 uppercase tracking-wider">RAGAS Evaluation (24h)</div>
              <div className="text-xs text-gray-600 mt-0.5">{evalHealth.total_evaluated} queries evaluated — faithfulness + answer relevancy</div>
            </div>
            <div className="flex gap-3">
              <ScorePill
                label="Faithfulness"
                value={evalHealth.avg_faithfulness}
                color={
                  evalHealth.avg_faithfulness === null ? "gray" :
                  evalHealth.avg_faithfulness >= 0.9 ? "green" :
                  evalHealth.avg_faithfulness >= 0.7 ? "yellow" : "red"
                }
              />
              <ScorePill
                label="Relevancy"
                value={evalHealth.avg_answer_relevancy}
                color={
                  evalHealth.avg_answer_relevancy === null ? "gray" :
                  evalHealth.avg_answer_relevancy >= 0.9 ? "green" :
                  evalHealth.avg_answer_relevancy >= 0.7 ? "yellow" : "red"
                }
              />
            </div>
          </div>

          {rasgasData.length > 0 && (
            <ResponsiveContainer width="100%" height={120}>
              <BarChart data={rasgasData} layout="vertical" margin={{ top: 0, right: 16, left: 8, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" horizontal={false} />
                <XAxis type="number" domain={[0, 1]} tick={{ fill: "#4b5563", fontSize: 10 }} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                <YAxis type="category" dataKey="metric" tick={{ fill: "#9ca3af", fontSize: 11 }} width={110} />
                <Tooltip
                  contentStyle={{ background: "#111827", border: "1px solid #1f2937", borderRadius: 8 }}
                  labelStyle={{ color: "#9ca3af" }}
                  itemStyle={{ color: "#a5b4fc" }}
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  formatter={(v: any) => [typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "—"]}
                />
                <Bar dataKey="score" fill="#6366f1" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}

          <div className="mt-3 flex gap-6 text-xs text-gray-600">
            <span><span className="text-gray-400">Faithfulness</span> — % of answer claims grounded in retrieved chunks</span>
            <span><span className="text-gray-400">Relevancy</span> — % of answer that addresses the question asked</span>
          </div>
        </div>
      )}

      {/* Logs table */}
      <div className="flex flex-col gap-2">
        <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">Query log</div>
        {loading && <div className="text-gray-600 text-sm">Loading...</div>}
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

function ScorePill({ label, value, color }: { label: string; value: number | null; color: "green" | "yellow" | "red" | "gray" }) {
  const colorClass =
    color === "green" ? "text-green-400 bg-green-400/10 border-green-400/20" :
    color === "yellow" ? "text-yellow-400 bg-yellow-400/10 border-yellow-400/20" :
    color === "red" ? "text-red-400 bg-red-400/10 border-red-400/20" :
    "text-gray-400 bg-gray-800 border-gray-700";

  return (
    <div className={`flex flex-col items-center px-3 py-1.5 rounded-lg border text-xs ${colorClass}`}>
      <div className="font-semibold text-base">{value !== null ? (value * 100).toFixed(1) + "%" : "—"}</div>
      <div className="opacity-70">{label}</div>
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
        <span className={`w-2 h-2 rounded-full shrink-0 ${log.answered ? "bg-green-500" : "bg-yellow-500"}`} />
        <span className="flex-1 text-sm text-gray-200 truncate">{log.query_text}</span>
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
