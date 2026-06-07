export interface Citation {
  chunk_id: string;
  source_file: string;
  source_url: string;
  page_number: number | null;
  section_title: string | null;
  excerpt: string;
}

export interface QueryResponse {
  answered: boolean;
  answer_text: string | null;
  citations: Citation[];
  confidence_score: number;
  top_rrf_score: number;
  latency_total_ms: number;
  refusal_reason: string | null;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface QueryLogEntry {
  id: string;
  queried_at: string;
  query_text: string;
  answered: boolean;
  answer_text: string | null;
  top_rrf_score: number | null;
  confidence_score: number | null;
  latency_total_ms: number;
  citations: Citation[];
  error_message: string | null;
}

export interface SystemHealth {
  total_queries: number;
  total_answered: number;
  total_refused: number;
  answer_rate_pct: number;
  avg_latency_ms: number | null;
  avg_confidence: number | null;
}

export interface EvalHealth {
  total_evaluated: number;
  avg_faithfulness: number | null;
  avg_answer_relevancy: number | null;
}
