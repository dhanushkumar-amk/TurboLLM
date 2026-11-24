import { Activity, Blocks, Database, Shield, Users } from "lucide-react";
import { motion } from "motion/react";
import { Card, CardContent } from "@/components/ui/card";

const features = [
  {
    number: "01",
    title: "Understand every step",
    text: "Trace a prompt through prefill, attention, sampling, and decode in a small PyTorch runtime.",
    visual: "flow",
  },
  {
    number: "02",
    title: "Make memory visible",
    text: "Measure how model weights and the KV cache compete for the RTX 3050's 8 GB of VRAM.",
    visual: "memory",
  },
  {
    number: "03",
    title: "Prove correctness first",
    text: "Compare logits and generated tokens with a Hugging Face reference before tuning speed.",
    visual: "correct",
  },
  {
    number: "04",
    title: "Measure real tradeoffs",
    text: "Benchmark matching workloads and report latency, throughput, memory, and output quality.",
    visual: "chart",
    wide: "feature-wide",
  },
  {
    number: "05",
    title: "Study shared serving",
    text: "Build toward request scheduling, continuous batching, cancellation, and cache cleanup.",
    visual: "queue",
    wide: "feature-wide",
  },
];

export function Features() {
  return (
    <motion.section className="section features-section" id="approach" aria-labelledby="features-title" initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-70px" }} transition={{ duration: 0.65 }}>
      <p className="eyeline">What this project explores</p>
      <h2 id="features-title">A runtime you can inspect.</h2>
      <p className="features-intro">These are the engineering questions behind TurboLLM. Each capability will be implemented and measured as the roadmap progresses.</p>
      <div className="feature-grid">
        {features.map((feature, index) => (
          <motion.div key={feature.number} className={`feature-reveal ${feature.wide ?? ""}`} initial={{ opacity: 0, y: 28 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-45px" }} transition={{ duration: 0.5, delay: index * 0.07 }}>
          <Card className="feature-card">
            <CardContent className="feature-content">
              <div className={`feature-visual feature-visual-${feature.visual}`} aria-hidden="true">
                {feature.visual === "flow" && <>
                  <div className="flow-label">PROMPT</div><div className="flow-track"><span /><span /><span /><i /></div>
                  <div className="flow-steps"><b>prefill</b><b>KV cache</b><b>decode</b></div>
                </>}
                {feature.visual === "memory" && <>
                  <Database size={30} strokeWidth={1.3} /><div className="memory-stack"><span>weights</span><i style={{ width: "72%" }} /><span>KV cache</span><i style={{ width: "48%" }} /><small>8 GB budget</small></div>
                </>}
                {feature.visual === "correct" && <>
                  <Shield size={34} strokeWidth={1.2} /><div className="check-lines"><span>reference logits <b>↔</b></span><span>runtime logits <b>✓</b></span><i /></div>
                </>}
                {feature.visual === "chart" && <>
                  <div className="chart-top"><Activity size={16} /> illustrative benchmark</div><div className="chart-bars"><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /></div><div className="chart-axis"><span>prefill</span><span>decode</span><span>batch</span></div>
                </>}
                {feature.visual === "queue" && <>
                  <Users size={28} strokeWidth={1.3} /><div className="request-lanes"><span><b>A</b> prompt <i /></span><span><b>B</b> prompt <i /></span><span><b>C</b> prompt <i /></span></div><Blocks size={24} strokeWidth={1.3} />
                </>}
              </div>
              <div className="feature-copy"><span className="feature-number">{feature.number} / TURBOLLM</span><h3>{feature.title}</h3><p>{feature.text}</p></div>
            </CardContent>
          </Card>
          </motion.div>
        ))}
      </div>
    </motion.section>
  );
}
