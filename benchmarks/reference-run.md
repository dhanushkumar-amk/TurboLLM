# Reference Generation Record

The first Transformers smoke run used Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`, FP16, CUDA on the RTX 3050, and a 48-token cap.

- Prompt: `Explain what a key-value cache does in one short sentence.`
- Generated text: `A key-value cache is a data structure that stores and retrieves values based on keys, allowing for fast access to frequently accessed data without the need for full database lookups.`
- Generated tokens: 35
- First-run load and download: 76.41 seconds
- Generation: 1.62 seconds
- Peak allocated VRAM: 961 MiB

This is a smoke-test record from the first model download, not a repeatable performance result. See `local-baseline.md` for later controlled-script runs.
