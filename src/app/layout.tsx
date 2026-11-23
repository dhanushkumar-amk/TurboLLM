import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TurboLLM — The engine behind the answer",
  description:
    "A ground-up study of LLM inference. Understand every operation, every byte, every tradeoff—from the first tensor to a streaming response.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
