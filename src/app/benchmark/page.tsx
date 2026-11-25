import BenchmarkDashboard from "@/components/benchmark-dashboard";
import fs from "node:fs/promises";
import path from "node:path";
import "./benchmark.css";

export const instant = false;

export default async function BenchmarkPage() {
  let report = null;
  try {
    const file = path.join(process.cwd(), "benchmarks", "results", "latest.json");
    report = JSON.parse(await fs.readFile(file, "utf8"));
  } catch { /* dashboard explains how to create the first run */ }
  return <BenchmarkDashboard report={report} />;
}
