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

### How does the current landing page render its sculpture?

The hero draws a rotating tube around a mathematical knot. The browser projects its mesh onto a canvas and paints the distant faces first. A pause control and reduced-motion support keep the illustration usable at rest. The architecture explorer explains prefill, decode, and scheduling.
