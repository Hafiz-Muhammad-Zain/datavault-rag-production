import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function GET() {
  const [logsRes, healthRes, evalRes] = await Promise.all([
    fetch(`${BACKEND_URL}/logs?limit=50`, { cache: "no-store" }),
    fetch(`${BACKEND_URL}/logs/health`, { cache: "no-store" }),
    fetch(`${BACKEND_URL}/logs/eval`, { cache: "no-store" }),
  ]);

  const logs = await logsRes.json();
  const health = await healthRes.json();
  const eval_health = await evalRes.json();

  return NextResponse.json({ logs, health, eval_health });
}
