"use client";

import { useState, useRef, useEffect } from "react";
import { Citation, ChatMessage, QueryResponse } from "@/lib/types";

const EXAMPLE_QUESTIONS = [
  "What is the deadline for notifying the supervisory authority after a data breach?",
  "How long must DataVault retain employee payroll data?",
  "What rights do DataVault employees have regarding their personal data?",
  "Does DataVault transfer data outside the EU, and what safeguards apply?",
  "What happens if an employee uses an AI tool like ChatGPT at work?",
];

interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  confidence?: number;
  latency_ms?: number;
  answered?: boolean;
  refusal_reason?: string | null;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function sendMessage(question: string) {
    if (!question.trim() || loading) return;

    const userMsg: Message = { role: "user", content: question };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    // Only include answered messages in history — refused/empty answers confuse the LLM
    const chatHistory: ChatMessage[] = messages
      .filter((m) => m.role === "user" || (m.role === "assistant" && m.answered && m.content))
      .map((m) => ({ role: m.role, content: m.content }));

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, chat_history: chatHistory }),
      });

      const data: QueryResponse = await res.json();

      const assistantMsg: Message = {
        role: "assistant",
        content: data.answer_text ?? "",
        citations: data.citations,
        confidence: data.confidence_score,
        latency_ms: data.latency_total_ms,
        answered: data.answered,
        refusal_reason: data.refusal_reason,
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "",
          answered: false,
          refusal_reason: "Network error — could not reach the backend.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col h-full max-w-3xl mx-auto w-full px-4 py-6 gap-4" style={{ minHeight: "calc(100vh - 49px)" }}>
      {/* Header */}
      <div className="text-center">
        <h1 className="text-xl font-semibold text-white">DataVault Compliance Assistant</h1>
        <p className="text-gray-500 text-sm mt-1">
          GDPR · BDSG · DataVault internal policy — hybrid search + hallucination prevention
        </p>
      </div>

      {/* Chat area */}
      <div className="flex-1 flex flex-col gap-4 overflow-y-auto">
        {messages.length === 0 && (
          <div className="flex flex-col gap-3 mt-4">
            <p className="text-gray-500 text-sm text-center">Try an example question:</p>
            <div className="flex flex-col gap-2">
              {EXAMPLE_QUESTIONS.map((q) => (
                <button
                  key={q}
                  onClick={() => sendMessage(q)}
                  className="text-left text-sm px-4 py-3 rounded-lg border border-gray-800 bg-gray-900 hover:bg-gray-800 hover:border-gray-700 text-gray-300 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex flex-col gap-1 ${msg.role === "user" ? "items-end" : "items-start"}`}>
            {msg.role === "user" ? (
              <div className="bg-blue-600 text-white rounded-2xl rounded-tr-sm px-4 py-2.5 max-w-[85%] text-sm">
                {msg.content}
              </div>
            ) : (
              <div className="flex flex-col gap-2 max-w-[95%]">
                {msg.answered ? (
                  <>
                    <div className="bg-gray-900 border border-gray-800 rounded-2xl rounded-tl-sm px-4 py-3 text-sm text-gray-100 leading-relaxed">
                      {msg.content}
                    </div>
                    {msg.citations && msg.citations.length > 0 && (
                      <CitationsBlock citations={msg.citations} />
                    )}
                    <div className="flex gap-3 text-xs text-gray-600 px-1">
                      {msg.citations && msg.citations.length > 0 && (
                        <span>confidence {((msg.confidence ?? 0) * 100).toFixed(0)}%</span>
                      )}
                      <span>{msg.latency_ms}ms</span>
                    </div>
                  </>
                ) : (
                  <div className="bg-gray-900 border border-yellow-900/50 rounded-2xl rounded-tl-sm px-4 py-3 text-sm text-yellow-400/80">
                    {msg.refusal_reason ?? "No relevant information found in the knowledge base."}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex items-start">
            <div className="bg-gray-900 border border-gray-800 rounded-2xl rounded-tl-sm px-4 py-3">
              <div className="flex gap-1.5 items-center h-4">
                <span className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce [animation-delay:-0.3s]" />
                <span className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce [animation-delay:-0.15s]" />
                <span className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce" />
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          sendMessage(input);
        }}
        className="flex gap-2"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a compliance question..."
          disabled={loading}
          className="flex-1 bg-gray-900 border border-gray-800 rounded-xl px-4 py-2.5 text-sm text-gray-100 placeholder-gray-600 focus:outline-none focus:border-gray-600 disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium px-4 py-2.5 rounded-xl transition-colors"
        >
          Send
        </button>
      </form>
    </div>
  );
}

function CitationsBlock({ citations }: { citations: Citation[] }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="px-1">
      <button
        onClick={() => setOpen((v) => !v)}
        className="text-xs text-blue-400/70 hover:text-blue-400 transition-colors"
      >
        {open ? "Hide" : "Show"} {citations.length} citation{citations.length !== 1 ? "s" : ""}
      </button>
      {open && (
        <div className="flex flex-col gap-2 mt-2">
          {citations.map((c, i) => (
            <div
              key={i}
              className="bg-gray-900/50 border border-gray-800 rounded-lg px-3 py-2 text-xs text-gray-400"
            >
              <div className="flex items-center gap-2 mb-1">
                <span className="text-gray-500 font-mono">[{i + 1}]</span>
                <span className="text-gray-300 font-medium truncate">{c.source_file}</span>
                {c.page_number && <span className="text-gray-600">p.{c.page_number}</span>}
              </div>
              {c.section_title && c.section_title !== "None" && (
                <div className="text-gray-500 mb-1">{c.section_title}</div>
              )}
              <div className="text-gray-400 italic leading-relaxed">&ldquo;{c.excerpt}&rdquo;</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
