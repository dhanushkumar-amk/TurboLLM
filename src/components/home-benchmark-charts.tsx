"use client";

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

type Run = { engine:string; concurrency:number; workload:string; summary:Record<string,any> };
const colors={p50:"#176b57",p95:"#bd7b35",p99:"#9d5c70",main:"#176b57"};
const label=(run:Run)=>`${run.workload.startsWith("short")?"Short":"Long"} · C${run.concurrency}`;

function LatencyChart({title,metric,rows,caption}:{title:string;metric:string;rows:Run[];caption:string}){
  const data=rows.map(run=>({label:label(run),p50:run.summary[metric]?.p50??null,p95:run.summary[metric]?.p95??null,p99:run.summary[metric]?.p99??null}));
  return <article className="home-chart-card"><div className="home-chart-heading"><div><h3>{title}</h3><p>{caption}</p></div><span>ms</span></div><div className="home-chart-plot"><ResponsiveContainer width="100%" height="100%"><LineChart data={data} margin={{top:12,right:14,bottom:3,left:0}}><CartesianGrid stroke="#e8eeea" vertical={false}/><XAxis dataKey="label" stroke="#83918a" tickLine={false} axisLine={false} tick={{fontSize:10}}/><YAxis stroke="#83918a" tickLine={false} axisLine={false} width={52} tick={{fontSize:10}}/><Tooltip contentStyle={{background:"#fff",border:"1px solid #dce5df",borderRadius:8,color:"#22362e",fontSize:12}}/><Line type="monotone" dataKey="p50" name="p50" stroke={colors.p50} strokeWidth={2.5} dot={{r:4}} connectNulls/><Line type="monotone" dataKey="p95" name="p95" stroke={colors.p95} strokeWidth={1.8} strokeDasharray="5 4" dot={{r:3}} connectNulls/><Line type="monotone" dataKey="p99" name="p99" stroke={colors.p99} strokeWidth={1.8} strokeDasharray="2 4" dot={{r:3}} connectNulls/></LineChart></ResponsiveContainer></div><div className="home-chart-legend"><span><i style={{background:colors.p50}}/>p50</span><span><i style={{background:colors.p95}}/>p95</span><span><i style={{background:colors.p99}}/>p99</span></div></article>;
}

function RateChart({title,metric,unit,rows,note}:{title:string;metric:string;unit:string;rows:Run[];note:string}){
  const data=rows.map(run=>({label:label(run),value:run.summary[metric]??null}));
  return <article className="home-chart-card"><div className="home-chart-heading"><div><h3>{title}</h3><p>{note}</p></div><span>{unit}</span></div><div className="home-chart-plot"><ResponsiveContainer width="100%" height="100%"><LineChart data={data} margin={{top:12,right:14,bottom:3,left:0}}><CartesianGrid stroke="#e8eeea" vertical={false}/><XAxis dataKey="label" stroke="#83918a" tickLine={false} axisLine={false} tick={{fontSize:10}}/><YAxis stroke="#83918a" tickLine={false} axisLine={false} width={52} tick={{fontSize:10}}/><Tooltip contentStyle={{background:"#fff",border:"1px solid #dce5df",borderRadius:8,color:"#22362e",fontSize:12}}/><Line type="monotone" dataKey="value" name={title} stroke={colors.main} strokeWidth={2.8} dot={{r:4,fill:colors.main}} activeDot={{r:6}} connectNulls/></LineChart></ResponsiveContainer></div><div className="home-rate-caption">TurboLLM · one measured series</div></article>;
}

export default function HomeBenchmarkCharts({runs}:{runs:Run[]}){
  const byMetric=(key:string)=>runs.some(r=>typeof r.summary[key]==="number");
  const charts=[
    byMetric("ttft_ms")&&<LatencyChart key="ttft" title="Time to first token" metric="ttft_ms" rows={runs} caption="Responsiveness · percentile tails"/>,
    byMetric("tpot_ms")&&<LatencyChart key="tpot" title="TPOT estimate" metric="tpot_ms" rows={runs} caption="Decode pace · streamed-duration estimate"/>,
    byMetric("e2e_ms")&&<LatencyChart key="e2e" title="End-to-end latency" metric="e2e_ms" rows={runs} caption="Request start through completion"/>,
    byMetric("output_tokens_per_sec")&&<RateChart key="tokens" title="Output throughput" metric="output_tokens_per_sec" unit="tok/s" rows={runs} note="System output tokens per second"/>,
    byMetric("requests_per_sec")&&<RateChart key="requests" title="Request rate" metric="requests_per_sec" unit="req/s" rows={runs} note="Completed requests per second"/>,
    byMetric("goodput_requests_per_sec")&&<RateChart key="goodput" title="SLO goodput" metric="goodput_requests_per_sec" unit="req/s" rows={runs} note="Requests meeting both latency targets"/>,
  ].filter(Boolean);
  return <div className="home-charts">{charts}</div>;
}
