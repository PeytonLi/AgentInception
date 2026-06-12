import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AgentInception — Live Console",
  description:
    "Latent KV-bank injection for web agents. Live Memory Inception dashboard.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
