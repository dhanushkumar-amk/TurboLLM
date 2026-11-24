# TurboLLM

TurboLLM is a learning project for building an LLM inference engine in PyTorch. The project is planned in stages: understand model execution, implement a correct forward pass, add KV-cache memory management and scheduling, then profile, serve, and benchmark the engine.

The first milestone is the project landing page and roadmap. Inference features and benchmark results are not implemented yet; the PRD and 50-phase plan describe the intended work.

## Project documents

- [Product requirements](PRD.md)
- [50-phase implementation plan](plans.md)
- [Interview notes](interview.md)
- [Development hardware and software baseline](environment.md)

## Landing page

The landing page uses Next.js, TypeScript, Tailwind CSS, shadcn/ui, Motion, GSAP, and Lucide icons.

```bash
npm install
npm run dev
```

Open `http://localhost:3000` in your browser.

To check the project:

```bash
npm run lint
npm run build
```

## Current development machine

- AMD Ryzen 7 5800X
- 32 GB RAM
- NVIDIA GeForce RTX 3050 with 8 GB VRAM

These are development constraints, not benchmark claims. Performance targets will be set after a model and workload baseline are selected.

## Initial model

The first planned reference model is [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). Its architecture exercises the RMSNorm, RoPE, SwiGLU, and GQA components in the roadmap. Record a pinned model revision before downloading weights or publishing benchmark results.

## PyTorch reference run

The ML dependencies are kept separate from the Next.js app dependencies. In PowerShell, create or activate `.venv`, then install and run:

```powershell
python -m pip install -r requirements-ml.txt
python scripts/reference_run.py
```

The runner pins model revision `7ae557604adf67be50417f59c2c2f167def9a775`, uses FP16 on CUDA, and asks for at most 48 new tokens. Hugging Face downloads the weights to its local cache. The printed load time, generation time, generated-token count, and peak allocated VRAM are a smoke-test baseline; repeat runs and control other GPU workloads before treating timing as a benchmark.
