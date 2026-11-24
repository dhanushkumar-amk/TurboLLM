# TurboLLM Development Environment

Captured 2026-11-24 from the development PC. The operating-system values are recorded as reported by the screenshot and the local registry because they disagree.

## Hardware

| Component | Detected configuration |
|---|---|
| Processor | AMD Ryzen 7 5800X, 8 cores / 16 threads, about 3.8 GHz |
| System memory | 32 GB DDR4, 3200 MHz per the System > About screenshot |
| GPU | NVIDIA GeForce RTX 3050, 8 GB VRAM |
| GPU driver | 610.88 (NVIDIA-SMI) |
| GPU mode | WDDM |
| Storage | 1.14 TB total per the System > About screenshot |

At capture, NVIDIA-SMI reported 626 MiB of 8192 MiB GPU memory in use. This is a momentary reading and will vary while applications are running.

## Software

| Software | Detected version / state |
|---|---|
| Windows | Screenshot says Windows 11 Pro 26H2. Registry reports Windows 10 Pro 26H2, build 26300. |
| Python | 3.15.2 |
| PyTorch | Not installed in the active Python environment (`import torch` fails). |
| CUDA toolkit compiler | `nvcc` not found on PATH. |
| NVIDIA driver CUDA UMD | 13.3, reported by NVIDIA-SMI. This is not evidence that the CUDA Toolkit is installed. |
| Node.js | v24.20.0 |
| npm | 11.19.0 |
| Git | 2.55.0.windows.5 |

## Notes for the first model run

- The NVIDIA driver detects the RTX 3050, but the active Python environment is not ready for PyTorch inference yet.
- Choose a PyTorch version and CUDA wheel that officially supports the installed Python version. If Python 3.15 is unsupported by the intended PyTorch release, use a supported Python version in a project-local virtual environment.
- The CUDA runtime bundled with a PyTorch wheel and the separately installed CUDA Toolkit are different. The Toolkit is only needed for workflows that compile CUDA extensions.
- Close GPU-heavy applications before measuring available VRAM. The operating system and desktop apps already use part of the 8 GB.
- Resolve the Windows product-name mismatch before documenting a fixed OS requirement. The current build is 26300, while the registry product name differs from the screenshot.
