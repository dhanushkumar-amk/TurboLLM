# TurboLLM — Interview Notes

## Project foundation and landing page

### What is TurboLLM?

TurboLLM is a planned PyTorch inference-engine project. The goal is to build up the path from model weights and transformer operations to KV caching, request scheduling, serving, and measured comparisons. The landing page communicates the project’s purpose, hardware constraints, and development sequence; it does not claim that the inference engine or benchmarks are already complete.

### Why build an inference engine when vLLM and llama.cpp already exist?

The project is not positioned as a replacement for mature inference runtimes. It is a transparent systems-learning project: implement the main mechanisms, compare them against established engines where workloads are comparable, and use measurements to identify a specific improvement worth pursuing.

### Why use a pretrained model instead of training one from scratch?

The core project is about inference systems, so pretrained weights provide a practical way to validate model loading and forward-pass correctness without making expensive model training a prerequisite. A small model can be compared against a Hugging Face reference. Training and fine-tuning experience from SQLForge can be discussed separately or used as an additional workload if its architecture and checkpoint format are compatible.

### Why Next.js, Tailwind, shadcn, Motion, and GSAP?

- **Next.js and TypeScript** provide a structured app framework and type checking for the project site.
- **Tailwind CSS and shadcn/ui** provide a consistent styling and component foundation.
- **Motion** handles entrance and interface transitions.
- **GSAP with ScrollTrigger** reveals explanatory sections as they enter view.
- **Reduced-motion support** keeps animation from being required to understand the page.

### What does the 8 GB GPU constraint contribute?

The RTX 3050’s 8 GB of VRAM makes memory budgeting a relevant design concern. The engine will need to account for model weights, KV cache, and runtime allocations. The constraint motivates experiments; it does not by itself prove an optimization or performance advantage.

### Why redesign the landing page with a 3D animation?

The inference concept uses CSS 3D animation rather than a static screenshot. Its moving cache cube suggests memory and transformer state. Motion is decorative and respects the user's reduced-motion preference; the underlying explanation remains readable without it.

### What visual reference informed the landing page?

The supplied recording showed a bright SaaS landing page with a centered hero, strong whitespace, a product UI preview, feature sections, and a darker developer resources section. TurboLLM borrows that presentation structure and the lime/lavender contrast, while using its own copy and a clearly labeled illustrative inference UI. The dashboard is a design concept, not a claim that those engine controls already exist.

### How will the project be evaluated?

First establish correctness against Hugging Face. Then evaluate cache correctness, resource cleanup, and serving behavior. For performance comparisons, use documented matching models and workloads and report TTFT, inter-token latency, tokens per second, latency percentiles, peak memory, and output quality where relevant.

### Current status

The PRD, 50-phase plan, and project landing page are in place. Core model execution, cache management, batching, serving, and benchmark results remain future implementation work.

### What is the development environment?

The target PC has a Ryzen 7 5800X, 32 GB system RAM, and an RTX 3050 with 8 GB VRAM. The NVIDIA driver sees the GPU, but PyTorch is not installed in the active Python environment yet. The machine currently reports Python 3.15.2, so the next setup step is to check PyTorch's supported Python versions and create a compatible project virtual environment. See [environment.md](environment.md) for the captured versions and the Windows product-name discrepancy.

### Why start with Qwen2.5-0.5B-Instruct?

It is small enough to make the first end-to-end reference practical on an 8 GB GPU, while exposing the architecture components the project needs to implement: RMSNorm, RoPE, SwiGLU, and grouped-query attention. The Hugging Face model card lists 0.49B parameters, 24 layers, and an Apache-2.0 license. I will pin a checkpoint revision and validate outputs against Hugging Face before using it for performance claims.

### How does the current landing page render its sculpture?

The hero draws a rotating tube around a mathematical knot. The browser projects its mesh onto a canvas and paints the distant faces first. A pause control and reduced-motion support keep the illustration usable at rest. The architecture explorer explains prefill, decode, and scheduling.

### What does the feature-card section explain?

The section organizes TurboLLM's engineering questions into five themes: tracing inference, budgeting GPU memory, checking logits against Hugging Face, measuring performance tradeoffs, and studying shared request serving. Its diagrams are illustrative. The section describes work planned for the project and does not imply those engine capabilities are already implemented.

### How do the interactive landing-page demos work?

The memory lab estimates KV-cache size from a demo configuration of 24 layers, 2 KV heads, 64 dimensions, and FP16 key/value tensors. Sliders change context length and active request count. The request scheduler animates synthetic request lanes, and the scroll story uses gentle parallax to explain prompt input, cached context, and token output. These are explanatory frontend demos, not measurements from the inference engine; the displayed memory estimate excludes runtime workspaces and allocator overhead.

### What does the interactive trace show?

The trace replays a synthetic request across tokenization, prefill, KV-cache allocation, decode, token streaming, and cache release. Its timeline can be scrubbed, played, paused, and inspected by selecting spans. The timing and token counts are demo values, clearly labeled as such; they are not captured from a running TurboLLM engine.
