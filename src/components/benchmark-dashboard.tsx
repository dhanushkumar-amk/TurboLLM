"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Bar, BarChart, Cell, Scatter, ScatterChart, ZAxis } from "recharts";

type Run = { engine: string; concurrency: number; summary: Record<string, any> };
type Report = { created_at: string; workload: Record<string, any>; runs: Run[]; notes: string[] };

const colors: Record<string, string> = { TurboLLM: "#7ce8d0", vLLM: "#caff63" };

function Chart({ title, unit, rows, metric, type = "line", lower = true }: { title: string; unit: string; rows: Run[]; metric: (r: Run) => number | null; type?: "line" | "bar"; lower?: boolean }) {
  const data = rows.map(r => ({ ...r, value: metric(r) })).filter(r => r.value !== null) as (Run & { value: number })[];
  const chartData = [...new Set(data.map(d => d.concurrency))].sort((a,b) => a-b).map(concurrency => {
    const point: Record<string, number> = { concurrency };
    for (const engine of ["TurboLLM", "vLLM"]) {
      const item = data.find(d => d.concurrency === concurrency && d.engine === engine);
      if (item) point[engine] = item.value;
    }
    return point;
  });
  const max = Math.max(1, ...data.map(d => d.value));
  return <article className="bench-chart"><div className="bench-chart-head"><h3>{title}</h3><span>{unit} · {lower ? "lower is better" : "higher is better"}</span></div>
    {!data.length ? <div className="bench-no-data">Run the benchmark to populate this chart.</div> : <div className="bench-chart-canvas">
      <ResponsiveContainer width="100%" height="100%">{type === "line" ? <LineChart data={chartData} margin={{top:12,right:18,bottom:4,left:0}}><CartesianGrid stroke="#ffffff12" vertical={false}/><XAxis dataKey="concurrency" tickFormatter={v=>`C=${v}`} stroke="#72807c" tickLine={false}/><YAxis stroke="#72807c" tickLine={false} axisLine={false} width={48}/><Tooltip contentStyle={{background:"#111819",border:"1px solid #34403e",borderRadius:8,color:"#eef5f2"}} labelFormatter={v=>`Concurrency ${v}`}/>{["TurboLLM","vLLM"].map(engine=><Line key={engine} type="monotone" dataKey={engine} name={engine} stroke={colors[engine]} strokeWidth={2.5} dot={{r:4,fill:colors[engine]}} connectNulls activeDot={{r:6}}/>)}</LineChart> : <BarChart data={data} margin={{top:12,right:12,bottom:8,left:0}}><CartesianGrid stroke="#ffffff12" vertical={false}/><XAxis dataKey="concurrency" tickFormatter={(v,i)=>`${data[i]?.engine === "TurboLLM" ? "T" : "v"}·${v}`} stroke="#72807c" tickLine={false}/><YAxis stroke="#72807c" tickLine={false} axisLine={false} width={48}/><Tooltip contentStyle={{background:"#111819",border:"1px solid #34403e",borderRadius:8,color:"#eef5f2"}} formatter={(v)=>[v,title]} labelFormatter={(v,i)=>`${data[i]?.engine} · C=${v}`}/><Bar dataKey="value" radius={[4,4,0,0]}>{data.map((d,i)=><Cell key={i} fill={colors[d.engine]}/>)}</Bar></BarChart>}</ResponsiveContainer>
    </div>}
  </article>;
}

function LatencyPlot({ title, rows, keyName }: { title: string; rows: Run[]; keyName: "ttft_ms" | "itl_mean_ms" | "e2e_ms" }) {
  const points = rows.flatMap(r => { const v=r.summary[keyName]; return ["p50","p95","p99"].map((q,i)=>({x:r.concurrency+(r.engine==="TurboLLM"?-.12:.12),y:v[q],engine:r.engine,quantile:q,concurrency:r.concurrency,rank:i+1})).filter(p=>p.y!==null); });
  return <article className="bench-chart"><div className="bench-chart-head"><h3>{title}</h3><span>p50 · p95 · p99 / ms · box range</span></div>{!points.length?<div className="bench-no-data">Run the benchmark to populate this plot.</div>:<div className="bench-chart-canvas"><ResponsiveContainer width="100%" height="100%"><ScatterChart margin={{top:14,right:18,bottom:4,left:0}}><CartesianGrid stroke="#ffffff12"/><XAxis type="number" dataKey="x" domain={[Math.min(...rows.map(r=>r.concurrency))-.5,Math.max(...rows.map(r=>r.concurrency))+.5]} ticks={[...new Set(rows.map(r=>r.concurrency))]} tickFormatter={v=>`C=${v}`} stroke="#72807c"/><YAxis type="number" dataKey="y" stroke="#72807c" tickLine={false} axisLine={false} width={48}/><ZAxis type="number" dataKey="rank" range={[55,55]}/><Tooltip content={({active,payload})=>active&&payload?.length?<div className="bench-tooltip"><b>{payload[0].payload.engine} · C={payload[0].payload.concurrency}</b><span>{payload[0].payload.quantile}: {Number(payload[0].payload.y).toFixed(1)} ms</span></div>:null}/>{["TurboLLM","vLLM"].map(engine=><Scatter key={engine} name={engine} data={points.filter(p=>p.engine===engine)} fill={colors[engine]} shape="circle"/> )}</ScatterChart></ResponsiveContainer></div>}</article>;
}

export default function BenchmarkDashboard({ report }: { report: Report | null }) {
  const [level, setLevel] = useState<number | null>(null);
  const runs = useMemo(() => (report?.runs ?? []).filter(r => level === null || r.concurrency === level), [report, level]);
  const concurrency = [...new Set(report?.runs.map(r => r.concurrency) ?? [])];
  return <main className="bench-page"><header className="bench-nav"><Link href="/">← TurboLLM</Link><span>INFERENCE PERFORMANCE / LAB</span><a href="/benchmark">Refresh data ↻</a></header>
    <section className="bench-hero"><p className="bench-eyebrow"><i/> MEASURED SERVING COMPARISON</p><h1>Performance,<br/><em>under load.</em></h1><p>Matched streaming requests to TurboLLM and vLLM. Read the latency tails, sustained output, and SLO goodput together.</p>
      <div className="bench-meta"><span>MODEL <b>{report?.workload?.model ?? "Configured endpoint model"}</b></span><span>CAPTURED <b>{report ? new Date(report.created_at).toLocaleString() : "No run yet"}</b></span><span>WORKLOAD <b>{report ? `${report.workload.requests_per_engine_per_level} requests / level` : "Waiting for measurements"}</b></span></div></section>
    {!report && <section className="bench-empty"><span>01 / FIRST RUN</span><h2>Collect the baseline first.</h2><p>Start the TurboLLM and vLLM servers with the same model revision, dtype, GPU, and serving settings, then run the command below. The charts will load from the generated report.</p><pre>python scripts/benchmark_serving.py --vllm-url http://127.0.0.1:8001 --vllm-model Qwen/Qwen2.5-0.5B-Instruct</pre><p className="bench-muted">The dashboard does not invent a result. No benchmark data is currently recorded.</p></section>}
    {report && <>
      <section className="bench-filter"><div><span>VIEW</span><b>{level === null ? "All concurrency levels" : `${level} simultaneous requests`}</b></div><div className="bench-pills"><button onClick={() => setLevel(null)} className={level === null ? "selected" : ""}>All</button>{concurrency.map(c => <button key={c} onClick={() => setLevel(c)} className={level === c ? "selected" : ""}>C={c}</button>)}</div></section>
      <section className="bench-grid"><LatencyPlot title="TTFT latency distribution" rows={runs} keyName="ttft_ms"/><LatencyPlot title="ITL latency distribution" rows={runs} keyName="itl_mean_ms"/><LatencyPlot title="E2E latency distribution" rows={runs} keyName="e2e_ms"/><Chart title="Output throughput" unit="tokens / sec" rows={runs} lower={false} metric={r => r.summary.output_tokens_per_sec}/><Chart title="SLO goodput" unit="requests / sec" rows={runs} lower={false} metric={r => r.summary.goodput_requests_per_sec}/><Chart title="Request throughput" unit="requests / sec" rows={runs} lower={false} metric={r => r.summary.requests_per_sec}/></section>
      <section className="bench-table-wrap"><div className="bench-section-title"><div><p className="bench-eyebrow">DISTRIBUTION DETAIL</p><h2>Every tail, in the open.</h2></div><span>SLO: TTFT &lt; {report.workload.ttft_slo_ms} ms · ITL &lt; {report.workload.itl_slo_ms} ms</span></div><div className="bench-table-scroll"><table><thead><tr><th>Engine</th><th>Concurrency</th><th>TTFT p50 / p95 / p99</th><th>ITL p50 / p95 / p99</th><th>E2E p50 / p95 / p99</th><th>Tokens/s</th><th>Goodput</th><th>Errors</th></tr></thead><tbody>{runs.map((r,i)=><tr key={i}><td><i style={{background:colors[r.engine]}}/>{r.engine}</td><td>{r.concurrency}</td><td>{triple(r.summary.ttft_ms)}</td><td>{triple(r.summary.itl_mean_ms)}</td><td>{triple(r.summary.e2e_ms)}</td><td>{r.summary.output_tokens_per_sec}</td><td>{r.summary.goodput_requests_per_sec} req/s</td><td>{r.summary.errors}</td></tr>)}</tbody></table></div></section>
      <section className="bench-limits"><h2>Efficiency &amp; quality signals</h2><div className="bench-limits-grid"><article><span>GPU / memory bandwidth / MFU</span><p>Not included in endpoint timing. Capture with DCGM or Nsight on the serving host, using the same request window.</p></article><article><span>KV cache use / prefix hit rate</span><p>Needs per-engine server metrics. Add the serving engine's cache counters to the same run before comparing.</p></article><article><span>Cost per million tokens</span><p>Join measured token counts with the actual hourly GPU and host cost for your deployment.</p></article><article><span>Quality / perplexity</span><p>Run a fixed held-out evaluation on both engines. Record model revision, tokenizer, precision, and score beside this report.</p></article></div></section>
      <details className="bench-notes"><summary>Method and measurement limits</summary><ul><li>Same prompt, requested output cap, and concurrency levels are sent sequentially to each endpoint.</li><li>TTFT and ITL use client-observed non-empty SSE text chunks. Chunk boundaries can differ by server; validate tokenizer-level token timing for strict ITL.</li><li>Endpoint latency includes queueing, tokenization, HTTP, and server scheduling.</li>{report.notes.map((n,i)=><li key={i}>{n}</li>)}</ul></details>
    </>}
    <footer className="bench-footer">TURBOLLM / BENCHMARK WORKSPACE <Link href="/">Return to the engine ↗</Link></footer>
  </main>;
}

function triple(v: Record<string, number | null>) { return [v.p50, v.p95, v.p99].map(n => n === null ? "—" : n.toFixed(1)).join(" / "); }
