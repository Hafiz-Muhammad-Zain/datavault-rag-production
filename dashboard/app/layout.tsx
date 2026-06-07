import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "DataVault Compliance Assistant",
  description: "Production RAG system — GDPR + DataVault policy compliance chatbot with hybrid search and hallucination prevention.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full`}>
      <body className="min-h-full flex flex-col bg-gray-950 text-gray-100 antialiased">
        <nav className="border-b border-gray-800 px-6 py-3 flex items-center gap-6 text-sm">
          <span className="font-semibold text-white">DataVault Compliance RAG</span>
          <Link href="/" className="text-gray-400 hover:text-white transition-colors">Chat</Link>
          <Link href="/logs" className="text-gray-400 hover:text-white transition-colors">Observability</Link>
          <span className="ml-auto text-gray-600 text-xs">Portfolio project — zainsverse.de</span>
        </nav>
        <main className="flex-1 flex flex-col">{children}</main>
      </body>
    </html>
  );
}
