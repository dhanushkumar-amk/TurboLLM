import fs from "node:fs/promises";
import path from "node:path";
import { Activity, ArrowDownRight, AudioLines, Cpu, Layers3, MemoryStick, Timer, Zap } from "lucide-react";
import HomeBenchmarkCharts from "@/components/home-benchmark-charts";
import "./home-v2.css";

export const instant = false;

type Run = { engine: string; concurrency: number; workload: string; summary: Record<string, any> };
type Report = { created_at: string; title?: string; hardware?: Record<string, any>; workload: Record<string, any>; runs: Run[]; quality?: Record<string, any> };

async function getReport(): Promise<Report | null> {
  try {
    const file = path.join(process.cwd(), "benchmarks", "results", "latest.json");
    return JSON.parse(await fs.readFile(file, "utf8")) as Report;
  } catch { return null; }
}

export default async function Home() {
  const report = await getReport();
  const runs = report?.runs?.filter(run => run.engine === "TurboLLM") ?? [];
  const ttft = median(runs.map(run => run.summary.ttft_ms?.p50));
  const tokens = median(runs.map(run => run.summary.output_tokens_per_sec));
  const matches = Object.values(report?.quality?.exact_output_match_percent_by_workload ?? {}).filter((v): v is number => typeof v === "number");
  const matchRate = matches.length ? Math.min(...matches) : null;

  return <main className="home-v2">
    <header className="home-nav"><a className="home-brand" href="#top"><AudioLines size={22}/> TurboLLM</a><nav><a href="#runtime">Runtime</a><a href="#benchmark">Benchmark</a><a href="#method">Method</a></nav><a className="home-nav-cta" href="#benchmark">View this run <ArrowDownRight size={15}/></a></header>

    <section className="home-hero" id="top"><div className="hero-copy"><p className="home-kicker"><i/> PYTORCH INFERENCE / MEASURED LOCALLY</p><h1>Inference,<br/>made <em>visible.</em></h1><p className="home-lede">A small PyTorch runtime for understanding model execution, memory, and serving performance?with measurements from your own machine.</p><div className="home-actions"><a className="home-primary" href="#benchmark">Explore benchmark <ArrowDownRight size={16}/></a><a className="home-secondary" href="#runtime">How the runtime works <span>↘</span></a></div><div className="home-runline"><span><i/> {report ? "LATEST LOCAL RUN" : "BENCHMARK READY"}</span><b>{report ? new Date(report.created_at).toLocaleDateString() : "Waiting for measurements"}</b></div></div>
      <aside className="hero-proof"><div className="proof-top"><span>RUNTIME SNAPSHOT</span><span className="proof-tag">TURBOLLM ONLY</span></div><div className="proof-number"><strong>{ttft == null ? "—" : `${ttft.toFixed(0)}`}</strong><span>ms median TTFT</span></div><div className="proof-divider"/><div className="proof-stats"><div><span>Output rate</span><b>{tokens == null ? "—" : `${tokens.toFixed(1)} tok/s`}</b></div><div><span>Greedy repeat match</span><b>{matchRate == null ? "—" : `${matchRate}%`}</b></div></div><div className="proof-foot"><Cpu size={15}/>{report?.hardware?.gpu ?? "RTX 3050 · 8 GB"}<span>·</span> FP16</div><div className="proof-orbit proof-orbit-a"/><div className="proof-orbit proof-orbit-b"/></aside>
    </section>

    <section className="home-strip" aria-label="Runtime characteristics"><div><Layers3/><span><b>Prefill + decode</b><small>Inspect the generation path</small></span></div><div><MemoryStick/><span><b>KV cache aware</b><small>Track sequence memory</small></span></div><div><Activity/><span><b>Serving experiments</b><small>Measure latency under load</small></span></div><div><Zap/><span><b>Open implementation</b><small>Built in PyTorch</small></span></div></section>

    <section className="home-section runtime-section" id="runtime"><div className="section-heading"><div><p className="home-kicker">01 / THE RUNTIME</p><h2>A readable path<br/>through inference.</h2></div><p>Follow the work a language model performs: process the prompt, retain attention state, then generate one token at a time.</p></div><div className="runtime-grid"><article><span>01</span><Timer/><h3>Prefill</h3><p>Process input tokens together and create the initial key/value state.</p><small>INPUT → CONTEXT</small></article><article><span>02</span><Layers3/><h3>Decode</h3><p>Reuse cached attention state to produce the next token at each step.</p><small>CONTEXT → TOKEN</small></article><article><span>03</span><Activity/><h3>Serve</h3><p>Measure real requests, queueing behavior, latency, and aggregate throughput.</p><small>TOKEN → RESPONSE</small></article></div></section>

    <section className="home-section benchmark-section" id="benchmark"><div className="section-heading benchmark-heading"><div><p className="home-kicker">02 / LATEST BENCHMARK</p><h2>Performance, in context.</h2></div><div className="benchmark-context"><span>{report?.workload?.workloads?.length ?? 0} prompt sizes</span><i/> <span>{report?.workload?.concurrency?.length ?? 0} load levels</span><i/> <span>{report?.workload?.requests_per_engine_per_level ?? 0} requests / level</span></div></div>
      {report && runs.length ? <><HomeBenchmarkCharts runs={runs}/><div className="benchmark-footnote"><span><i/> Measured TurboLLM endpoint · RTX 3050</span><p>Three measured requests per workload/load level after one warmup. p95/p99 are preliminary at this sample size. TPOT is estimated from streamed output duration. GPU, bandwidth, cache, and cost telemetry were not collected in this run.</p></div></> : <div className="benchmark-empty"><p>No TurboLLM run is recorded yet.</p><span>Start the local server, run <code>scripts/benchmark_serving.py</code>, and refresh this page to load its charts.</span></div>}
    </section>

    <section className="home-method" id="method"><div><p className="home-kicker">03 / HOW TO READ THIS</p><h2>Real measurements.<br/><em>Honest boundaries.</em></h2></div><div><p>Latency and throughput charts are populated from the latest local report. Missing GPU counters, cache hit rates, and cost are left out until the runtime exports those measurements.</p><a href="/benchmark">Open report details <span>↗</span></a></div></section>
    <footer className="home-footer"><a className="home-brand" href="#top"><AudioLines size={20}/> TurboLLM</a><span>SMALL RUNTIME · CLEAR MEASUREMENTS</span><a href="#top">Back to top ↑</a></footer>
  </main>;
}

function median(values: Array<number | null | undefined>) { const xs=values.filter((v):v is number=>typeof v==="number").sort((a,b)=>a-b); if(!xs.length)return null;const i=Math.floor(xs.length/2);return xs.length%2?xs[i]:(xs[i-1]+xs[i])/2; }
