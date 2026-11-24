"use client";

import { useRef, useState } from "react";
import { motion, useScroll, useSpring, useTransform } from "motion/react";
import { ArrowDownRight, Cpu, Database, Gauge, Layers3, MoveRight } from "lucide-react";

const contexts = [512, 1024, 2048, 4096];

export function DemoLab() {
  const [contextIndex, setContextIndex] = useState(2);
  const [requests, setRequests] = useState(4);
  const [tab, setTab] = useState<"memory" | "scheduler">("memory");
  const sectionRef = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({ target: sectionRef, offset: ["start end", "end start"] });
  const smoothProgress = useSpring(scrollYProgress, { stiffness: 80, damping: 22, mass: 0.3 });
  const leftY = useTransform(smoothProgress, [0, 1], [46, -46]);
  const rightY = useTransform(smoothProgress, [0, 1], [-30, 34]);

  const context = contexts[contextIndex];
  // Demo model: 24 layers × 2 KV heads × 64 dimensions × K/V × FP16.
  const kvBytesPerToken = 24 * 2 * 64 * 2 * 2;
  const cacheMiB = (kvBytesPerToken * context * requests) / (1024 * 1024);
  const cacheGiB = cacheMiB / 1024;
  const vramPercent = Math.min((cacheGiB / 8) * 100, 100);

  return (
    <>
      <section className="section demo-section" id="memory-lab" ref={sectionRef}>
        <div className="demo-heading">
          <div><p className="eyeline">Interactive demo / synthetic values</p><h2>Give the cache a budget.</h2></div>
          <p>Adjust context and active requests to see how KV-cache demand grows. This is a model-based estimate, not a benchmark result.</p>
        </div>

        <div className="lab-shell">
          <div className="lab-topbar"><span className="lab-led" /> TURBOLLM / RESOURCE LAB <span className="lab-topbar-right">DEMO CONFIG · FP16</span></div>
          <div className="lab-tabs" role="tablist" aria-label="Interactive demo views">
            <button type="button" role="tab" aria-selected={tab === "memory"} onClick={() => setTab("memory")}><Database size={15}/> Memory map</button>
            <button type="button" role="tab" aria-selected={tab === "scheduler"} onClick={() => setTab("scheduler")}><Gauge size={15}/> Request scheduler</button>
          </div>

          {tab === "memory" ? (
            <div className="lab-body" role="tabpanel">
              <div className="lab-controls">
                <label htmlFor="context-slider"><span>Context per request</span><b>{context.toLocaleString()} tokens</b></label>
                <input id="context-slider" type="range" min="0" max={contexts.length - 1} step="1" value={contextIndex} onChange={(event) => setContextIndex(Number(event.target.value))} />
                <div className="range-ticks"><span>512</span><span>1K</span><span>2K</span><span>4K</span></div>
                <label htmlFor="request-slider"><span>Active sequences</span><b>{requests}</b></label>
                <input id="request-slider" type="range" min="1" max="16" value={requests} onChange={(event) => setRequests(Number(event.target.value))} />
                <div className="range-ticks"><span>1 request</span><span>16 requests</span></div>
                <p className="lab-assumption">Estimate assumes 24 layers, 2 KV heads, 64 dimensions, and FP16 keys and values.</p>
              </div>
              <div className="memory-readout">
                <div className="readout-title"><span><Cpu size={15}/> GPU MEMORY / 8 GB</span><span>ESTIMATE</span></div>
                <div className="memory-meter" role="img" aria-label={`Estimated KV cache ${cacheMiB.toFixed(0)} megabytes of 8 gigabytes`}>
                  <div className="meter-fill meter-weights" style={{ width: "12.5%" }} />
                  <div className="meter-fill meter-kv" style={{ width: `${Math.max(vramPercent, 1.4)}%` }} />
                  <div className="meter-fill meter-open" />
                </div>
                <div className="meter-legend"><span><i className="legend-weights"/>Demo weights*</span><span><i className="legend-kv"/>KV cache</span><span><i className="legend-open"/>Remaining</span></div>
                <div className="cache-total"><strong>{cacheGiB < 1 ? `${cacheMiB.toFixed(0)} MiB` : `${cacheGiB.toFixed(2)} GiB`}</strong><span>estimated KV cache</span></div>
                <div className="memory-stats"><span><b>{(cacheMiB / requests).toFixed(0)} MiB</b> per sequence</span><span><b>{vramPercent.toFixed(1)}%</b> of VRAM*</span></div>
                <small>*Illustrative only. Weight portion is schematic; runtime workspace and allocator overhead are excluded.</small>
              </div>
            </div>
          ) : (
            <div className="scheduler-panel" role="tabpanel">
              <div className="scheduler-controls">
                <label htmlFor="scheduler-slider"><span>Requests in this demo batch</span><b>{requests}</b></label>
                <input id="scheduler-slider" type="range" min="1" max="16" value={requests} onChange={(event) => setRequests(Number(event.target.value))} />
                <p>Concept view: a scheduler can interleave active sequences while each request keeps its own cache state.</p>
                <div className="scheduler-stat"><span>Active lanes</span><b>{requests}</b></div>
                <div className="scheduler-stat"><span>Demo mode</span><b>Continuous batch</b></div>
                <small className="lane-note">Showing up to 8 animated lanes.</small>
              </div>
              <div className="lane-list" aria-label="Animated request lanes">
                {Array.from({ length: Math.min(requests, 8) }, (_, index) => (
                  <div className="request-lane" key={index}>
                    <span className="lane-id">R{String(index + 1).padStart(2, "0")}</span>
                    <span className="lane-track"><i style={{ animationDelay: `${index * -0.31}s` }} /></span>
                    <span className="lane-status">{index % 3 === 0 ? "decode" : index % 3 === 1 ? "prefill" : "waiting"}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </section>

      <section className="section scroll-story" aria-labelledby="story-title">
        <div className="story-heading"><p className="eyeline">Follow one request</p><h2 id="story-title">From text to token.</h2><p>Scroll to move through a simplified generation cycle. The motion is a visual explanation, not a live model run.</p></div>
        <div className="story-track">
          <motion.div className="story-card story-prompt" style={{ y: leftY }}>
            <span className="story-index">01 / INPUT</span><div className="prompt-window"><i/><i/><i/><p>Explain paged attention</p><span>tokens: 4</span></div><b>Prompt arrives</b><small>Tokenizer maps text to IDs</small>
          </motion.div>
          <div className="story-spine"><span/><ArrowDownRight size={18}/><span/><ArrowDownRight size={18}/><span/></div>
          <motion.div className="story-card story-cache" style={{ y: rightY }}>
            <span className="story-index">02 / CONTEXT</span><div className="block-grid"><i/><i/><i/><i/><i/><i/><i/><i/><i/><i/><i/><i/></div><b>KV state is stored</b><small>Attention reuses prior keys and values</small>
          </motion.div>
          <motion.div className="story-card story-token" style={{ y: leftY }}>
            <span className="story-index">03 / OUTPUT</span><div className="token-orbit"><Layers3/><span>next token</span><MoveRight/></div><b>One token is sampled</b><small>Decode repeats until the request ends</small>
          </motion.div>
        </div>
        <div className="story-progress" aria-hidden="true"><motion.i style={{ scaleX: smoothProgress, transformOrigin: "left" }}/></div>
      </section>
    </>
  );
}
