"use client";

import { useEffect, useRef, useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import {
  ArrowDown,
  ArrowUpRight,
  ChevronRight,
  Circle,
  Cpu,
  Layers3,
  MemoryStick,
  Menu,
  ScanLine,
  Terminal,
  X,
} from "lucide-react";

const phases = [
  { id: "01", title: "Understand the model", detail: "Weights, tokenizer, architecture", state: "in-progress" },
  { id: "02", title: "Build the forward pass", detail: "RMSNorm · RoPE · GQA · SwiGLU", state: "up-next" },
  { id: "03", title: "Make memory count", detail: "KV cache · paging · budgeting", state: "planned" },
  { id: "04", title: "Serve real requests", detail: "Scheduling · batching · streaming", state: "planned" },
  { id: "05", title: "Measure every tradeoff", detail: "Profile · compare · improve", state: "planned" },
];

const workloads = [
  { name: "Hugging Face", role: "Correctness reference", mark: "HF", className: "hf" },
  { name: "llama.cpp", role: "Local inference baseline", mark: "ll", className: "llama" },
  { name: "vLLM", role: "Serving baseline", mark: "v", className: "vllm" },
  { name: "TurboLLM", role: "The system being built", mark: "T", className: "turbo" },
];

function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <a className="brand" href="#top" aria-label="TurboLLM home">
      <span className="brand-mark"><span /><span /><span /><span /></span>
      {!compact && <span className="brand-name">turbo<span>llm</span></span>}
    </a>
  );
}

function CacheDiagram() {
  return (
    <div className="cache-visual" aria-label="Animated three-dimensional illustration of transformer inference layers">
      <div className="cache-label label-top"><span className="signal-dot" />TRANSFORMER / LAYER 07</div>
      <div className="scene-shadow" />
      <div className="scene-orbit orbit-one" />
      <div className="scene-orbit orbit-two" />
      <div className="scene-orbit orbit-three" />
      <div className="engine-stack" aria-hidden="true">
        <div className="stack-layer layer-back"><i /><i /><i /><i /><i /><i /><i /><i /><b>ATTENTION</b></div>
        <div className="stack-layer layer-mid"><i /><i /><i /><i /><i /><i /><i /><i /><b>KV CACHE</b></div>
        <div className="stack-layer layer-front"><i /><i /><i /><i /><i /><i /><i /><i /><b>FEED FORWARD</b></div>
        <div className="engine-cube">
          <span className="cube-face cube-front"><Layers3 size={30} strokeWidth={1.35} /><small>INFERENCE</small></span>
          <span className="cube-face cube-top" />
          <span className="cube-face cube-side" />
        </div>
      </div>
      <div className="float-chip chip-token"><span>TOKEN</span><b>128</b><i /></div>
      <div className="float-chip chip-memory"><span>KV BLOCK</span><b>07 <i>●</i></b></div>
      <div className="float-chip chip-latency"><span>STEP</span><b>DECODE</b></div>
      <div className="cache-coordinate">PREFILL <span>→</span> DECODE <span>→</span> NEXT TOKEN</div>
    </div>
  );
}

function Reveal({ children, className = "", delay = 0 }: { children: React.ReactNode; className?: string; delay?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const reduceMotion = useReducedMotion();

  useEffect(() => {
    if (!ref.current || reduceMotion) return;
    gsap.registerPlugin(ScrollTrigger);
    const context = gsap.context(() => {
      gsap.fromTo(ref.current, { autoAlpha: 0, y: 24 }, {
        autoAlpha: 1,
        y: 0,
        duration: 0.8,
        delay,
        ease: "power3.out",
        scrollTrigger: { trigger: ref.current, start: "top 88%", once: true },
      });
    }, ref);
    return () => context.revert();
  }, [delay, reduceMotion]);

  return <div ref={ref} className={className}>{children}</div>;
}

export default function Home() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <main id="top" className="site-shell">
      <div className="grain" aria-hidden="true" />
      <header className="topbar wrap">
        <Brand />
        <nav className={`nav-links ${menuOpen ? "nav-open" : ""}`} aria-label="Main navigation">
          <a href="#approach" onClick={() => setMenuOpen(false)}>Approach</a>
          <a href="#architecture" onClick={() => setMenuOpen(false)}>Inside the engine</a>
          <a href="#roadmap" onClick={() => setMenuOpen(false)}>Roadmap</a>
        </nav>
        <div className="nav-actions">
          <a className="github-link" href="#roadmap"><Terminal size={15} /> <span>Project plan</span></a>
          <a className="nav-cta" href="#roadmap">Follow the build <ArrowUpRight size={15} /></a>
        </div>
        <button className="menu-toggle" aria-label={menuOpen ? "Close menu" : "Open menu"} onClick={() => setMenuOpen(!menuOpen)}>
          {menuOpen ? <X size={21} /> : <Menu size={21} />}
        </button>
      </header>

      <section className="hero wrap">
        <div className="hero-copy">
          <motion.div className="eyebrow" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
            <span className="eyebrow-pulse" /> INDEPENDENT INFERENCE, BUILT IN PUBLIC
          </motion.div>
          <motion.h1 initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.65, delay: 0.08 }}>
            The engine<br />behind the <span className="hero-accent">answer.</span>
          </motion.h1>
          <motion.p className="hero-description" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.18 }}>
            A ground-up study of LLM inference. From the first tensor to a streaming response—understand every operation, every byte, every tradeoff.
          </motion.p>
          <motion.div className="hero-actions" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.28 }}>
            <a href="#approach" className="button-primary">Explore the build <ArrowDown size={16} /></a>
            <a href="#architecture" className="text-link">How inference works <ChevronRight size={15} /></a>
          </motion.div>
          <motion.div className="hero-meta" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.6, delay: 0.45 }}>
            <div className="meta-item"><span className="meta-icon"><Cpu size={15} /></span><span>BUILT ON<b>PyTorch</b></span></div>
            <span className="meta-divider" />
            <div className="meta-item"><span className="meta-icon"><MemoryStick size={15} /></span><span>CONSTRAINED BY<b>8 GB VRAM</b></span></div>
          </motion.div>
        </div>
        <motion.div className="hero-art" initial={{ opacity: 0, scale: 0.96, x: 12 }} animate={{ opacity: 1, scale: 1, x: 0 }} transition={{ duration: 0.9, delay: 0.15, ease: [0.2, 0.75, 0.25, 1] }}>
          <CacheDiagram />
        </motion.div>
        <div className="hero-foot"><span>01 / SYSTEMS, MADE VISIBLE</span><span>SCROLL TO EXPLORE <ArrowDown size={12} /></span></div>
      </section>

      <section className="manifesto section-pad" id="approach">
        <div className="wrap manifesto-grid">
          <Reveal className="section-kicker"><span className="kicker-index">01</span><span>WHY BUILD ANOTHER ENGINE?</span></Reveal>
          <Reveal className="manifesto-content">
            <h2>Don’t just call a model.<br /><span>Know what happens inside.</span></h2>
            <p className="manifesto-lead">Mature engines make inference fast. TurboLLM makes the systems behind inference understandable—and gives us a place to measure what actually matters on a small GPU.</p>
            <div className="manifesto-note"><span className="note-rule" /><p>Not a replacement for vLLM or llama.cpp. A transparent, testable engine built to learn from them—and find one real constraint worth improving.</p></div>
          </Reveal>
        </div>
      </section>

      <section className="architecture section-pad" id="architecture">
        <div className="wrap">
          <Reveal className="section-heading-row">
            <div><div className="section-kicker"><span className="kicker-index">02</span><span>THE INFERENCE PATH</span></div><h2>From prompt to token.</h2></div>
            <p>Each step earns its place. We build it, compare it, and keep the evidence.</p>
          </Reveal>
          <div className="flow-grid">
            {[
              { number: "01", icon: <Terminal size={19} />, title: "Read the model", text: "Load weights, tokenize the prompt, and shape the input the way the model expects." },
              { number: "02", icon: <ScanLine size={19} />, title: "Run the layers", text: "RMSNorm, RoPE, grouped-query attention, and SwiGLU—implemented and checked against a reference." },
              { number: "03", icon: <Layers3 size={19} />, title: "Remember context", text: "Keep key/value state between tokens. Budget memory explicitly, then explore paged allocation." },
              { number: "04", icon: <Circle size={19} />, title: "Generate & serve", text: "Sample the next token, schedule requests, and stream responses through an API." },
            ].map((step, index) => (
              <Reveal key={step.number} className="flow-card" delay={index * 0.07}>
                <div className="flow-card-top"><span>{step.number} / 04</span><span className="flow-icon">{step.icon}</span></div>
                <h3>{step.title}</h3><p>{step.text}</p>
                <div className="flow-card-bottom"><span className={index === 0 ? "status-active" : "status-planned"} />{index === 0 ? "FOUNDATION" : "PLANNED"}</div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <section className="constraint section-pad">
        <div className="wrap constraint-panel">
          <Reveal className="constraint-copy">
            <div className="section-kicker"><span className="kicker-index">03</span><span>THE DESIGN CONSTRAINT</span></div>
            <h2>8 GB is the<br /><span>whole point.</span></h2>
            <p>TurboLLM is being developed on a GeForce RTX 3050 with 8 GB of VRAM. That puts memory management in the foreground: not as a footnote, but as a design constraint we can measure against.</p>
            <a className="text-link" href="#roadmap">See how it shapes the roadmap <ChevronRight size={15} /></a>
          </Reveal>
          <Reveal className="spec-card" delay={0.12}>
            <div className="spec-card-header"><span>DEVELOPMENT MACHINE</span><span className="spec-live"><i /> LOCAL</span></div>
            <div className="spec-row"><span className="spec-icon"><Cpu size={17} /></span><span className="spec-label">PROCESSOR</span><b>AMD Ryzen 7 5800X</b><small>8 cores · 3.80 GHz</small></div>
            <div className="spec-row"><span className="spec-icon"><MemoryStick size={17} /></span><span className="spec-label">SYSTEM MEMORY</span><b>32 GB DDR4</b><small>3200 MHz</small></div>
            <div className="spec-row"><span className="spec-icon"><Layers3 size={17} /></span><span className="spec-label">GPU MEMORY</span><b>8 GB GDDR6</b><small>NVIDIA GeForce RTX 3050</small></div>
            <div className="spec-meter"><div><span>VRAM CAPACITY</span><b>8.0 GB TOTAL</b></div><span className="meter-track"><i /></span><small>Budget shared by model weights, KV cache, and runtime</small></div>
          </Reveal>
        </div>
      </section>

      <section className="roadmap section-pad" id="roadmap">
        <div className="wrap roadmap-layout">
          <Reveal className="roadmap-intro">
            <div className="section-kicker"><span className="kicker-index">04</span><span>THE BUILD LOGIC</span></div>
            <h2>Earn the<br /><span>optimization.</span></h2>
            <p>Correctness first. Then memory. Then concurrency. Only after we can measure each layer do we tune it.</p>
            <div className="roadmap-caption"><span className="caption-line" /> 50 SMALL STEPS. ONE WORKING ENGINE.</div>
          </Reveal>
          <div className="phase-list">
            {phases.map((phase, index) => (
              <Reveal key={phase.id} delay={index * 0.04}>
                <div className={`phase-row ${phase.state}`}>
                  <span className="phase-id">{phase.id}</span>
                  <div className="phase-main"><h3>{phase.title}</h3><p>{phase.detail}</p></div>
                  <span className="phase-status">{phase.state === "in-progress" ? <><i /> ACTIVE</> : phase.state === "up-next" ? "NEXT" : "QUEUED"}</span>
                </div>
              </Reveal>
            ))}
            <a className="all-phases" href="#architecture">EXPLORE THE INFERENCE SEQUENCE <ArrowUpRight size={14} /></a>
          </div>
        </div>
      </section>

      <section className="compare section-pad" id="comparisons">
        <div className="wrap">
          <Reveal className="compare-heading">
            <div><div className="section-kicker"><span className="kicker-index">05</span><span>LEARN FROM THE FIELD</span></div><h2>Different engines.<br /><span>Different tradeoffs.</span></h2></div>
            <p>We’ll use established tools as references, not strawmen. Same workloads where possible. Clear notes where they aren’t.</p>
          </Reveal>
          <div className="engine-grid">
            {workloads.map((engine, index) => (
              <Reveal key={engine.name} className={`engine-card ${engine.className}`} delay={index * 0.06}>
                <div className="engine-card-top"><div className="engine-mark">{engine.mark}</div><span>REFERENCE 0{index + 1}</span></div>
                <h3>{engine.name}</h3><p>{engine.role}</p><div className="engine-divider" /><div className="engine-bottom">{engine.name === "TurboLLM" ? "IMPLEMENTATION" : "COMPARISON"}<span className="engine-status">{engine.name === "TurboLLM" ? "BUILDING" : "PLANNED"}</span></div>
              </Reveal>
            ))}
          </div>
          <div className="metrics-strip"><span className="metrics-label">WHEN WE BENCHMARK</span><span>Time to first token</span><i /><span>Inter-token latency</span><i /><span>Tokens / second</span><i /><span>Peak memory</span><i /><span>Quality impact</span></div>
        </div>
      </section>

      <section className="closing section-pad">
        <div className="wrap closing-panel">
          <div className="closing-grid" aria-hidden="true" />
          <Reveal className="closing-content">
            <span className="closing-overline"><span className="eyebrow-pulse" /> THE BUILD IS JUST BEGINNING</span>
            <h2>Make inference<br />make sense.</h2>
            <p>Follow a system from its first tensor to its first honest benchmark.</p>
            <div className="closing-actions"><a href="#top" className="button-primary">Back to the beginning <ArrowUpRight size={15} /></a><a href="#roadmap" className="closing-secondary">Explore the roadmap</a></div>
          </Reveal>
          <div className="closing-mark"><Brand compact /><span>TURBOLLM<br />SYSTEMS PROJECT</span></div>
        </div>
      </section>

      <footer className="footer wrap">
        <Brand />
        <p>An independent inference engine, built to understand the machinery.</p>
        <div className="footer-right"><span>BUILT LOCALLY · MEASURED HONESTLY</span><a href="#top">BACK TO TOP ↑</a></div>
      </footer>
    </main>
  );
}
