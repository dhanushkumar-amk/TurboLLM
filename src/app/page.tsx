"use client";
import {useState} from "react";
import {motion, MotionConfig} from "motion/react";
import {Activity, AudioLines, Boxes, Cpu, Pause, Play} from "lucide-react";
import Sculpture from "@/components/sculpture";
import { Features } from "@/components/features";
import { DemoLab } from "@/components/demo-lab";
import { AgentTrace, type TraceSpan } from "@/components/ui/agent-trace";

const stages=[
  ["Prefill","Read the prompt","The model processes the prompt tokens together and stores their attention keys and values."],
  ["Decode","Generate a token","Each step uses the cached context to score the vocabulary and select the next token."],
  ["Schedule","Share the GPU","The scheduler groups requests and releases their cache blocks when generation ends."]
];
const milestones=["Match Hugging Face logits","Budget and page the KV cache","Batch concurrent requests","Profile and quantize","Serve and benchmark"];
const traceSpans: TraceSpan[] = [
  { id: "request", label: "generation_request", kind: "agent", start: 0, end: 10600, detail: "complete" },
  { id: "tokenizer", parentId: "request", label: "tokenizer.encode", kind: "io", start: 0, end: 380, detail: "24 token IDs" },
  { id: "prefill", parentId: "request", label: "model.prefill", kind: "model", start: 380, end: 2450, tokens: 24 },
  { id: "cache", parentId: "prefill", label: "kv_cache.reserve", kind: "tool", start: 920, end: 1560, detail: "3 blocks" },
  { id: "decode", parentId: "request", label: "model.decode", kind: "model", start: 2450, end: 10000, tokens: 32 },
  { id: "scheduler", parentId: "request", label: "scheduler.slot", kind: "agent", start: 2620, end: 7600, detail: "slot 02" },
  { id: "stream", parentId: "request", label: "stream.tokens", kind: "io", start: 2800, end: 9800, detail: "32 tokens" },
  { id: "cleanup", parentId: "request", label: "kv_cache.release", kind: "tool", start: 10000, end: 10600, detail: "3 blocks freed" },
];

export default function Home(){
  const [paused,setPaused]=useState(false),[stage,setStage]=useState(0);
  return <MotionConfig reducedMotion="user"><main id="top">
    <header><a className="brand" href="#top"><AudioLines/> TurboLLM</a><nav><a href="#architecture">Architecture</a><a href="#approach">Approach</a><a href="#memory-lab">Memory lab</a><a href="#trace-lab">Trace</a><a href="#roadmap">Roadmap</a></nav><a className="navButton" href="#memory-lab">Explore the lab</a></header>
    <section className="hero"><div className="intro"><p className="eyeline"><i/> Local inference lab</p><h1>Inside the<br/><em>inference</em><br/>engine.</h1><p className="lede">Building a PyTorch runtime to study how language models generate tokens within 8 GB of GPU memory.</p><a className="cta" href="#architecture">Explore the architecture</a></div>
    <figure><Sculpture paused={paused}/><figcaption><span>FIG. A &nbsp; / &nbsp; Tensor study</span><button onClick={()=>setPaused(!paused)} aria-label={paused?"Animate sculpture":"Pause sculpture"}>{paused?<Play size={16}/>:<Pause size={16}/>}</button></figcaption></figure></section>
    <div className="specs" id="lab"><span><Cpu/> RTX 3050</span><span><Boxes/> 8 GB VRAM</span><span><Activity/> PyTorch</span></div>
    <motion.section className="section" id="architecture" initial={false} whileInView={{y:[22,0],opacity:[.65,1]}} viewport={{once:true}}><p className="eyeline">Architecture study</p><h2>How a prompt becomes a token.</h2><div className="explorer"><div className="choices">{stages.map(([name],i)=><button key={name} aria-pressed={stage===i} onClick={()=>setStage(i)}>{name}</button>)}</div><div aria-live="polite"><motion.article key={stage} animate={{y:[8,0]}}><Layers/><h3>{stages[stage][1]}</h3><p>{stages[stage][2]}</p></motion.article></div></div></motion.section>
    <Features />
    <DemoLab />
    <section className="section trace-section" id="trace-lab">
      <div className="trace-heading"><div><p className="eyeline">Interactive trace / synthetic sample</p><h2>Replay the work behind a response.</h2></div><p>Scrub the timeline or select a span to inspect a sample request moving through tokenization, prefill, cache allocation, decode, and cleanup.</p></div>
      <div className="trace-theme"><AgentTrace spans={traceSpans} runId="demo_a17f" model="small-model · demo" duration={10600} autoPlay loop speed={0.8} holdMs={1400} showTokens labelWidth={205} /></div>
      <p className="trace-footnote">Synthetic timings and token counts for interface demonstration. TurboLLM does not produce these trace measurements yet.</p>
    </section>
    <section className="section" id="roadmap"><p className="eyeline">Development roadmap</p><h2>Building TurboLLM.</h2><ol>{milestones.map((name,i)=><motion.li key={name} whileInView={{x:[18,0],opacity:[.5,1]}} viewport={{once:true}}><span>Phase 0{i+1}</span>{name}</motion.li>)}</ol></section><footer><a className="brand" href="#top"><AudioLines/> TurboLLM</a><a href="#top">Back to top</a></footer>
  </main></MotionConfig>;
}
function Layers(){return <Boxes size={36} strokeWidth={1.4} aria-hidden="true"/>;}
