# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Live2Diff generates Live2D-like character animation videos using point trajectory control and diffusion models. The system tracks key points on a character across video frames, builds a connectivity graph between frames, and uses a WAN video diffusion model with optical flow adapters to generate smooth transitions between character poses.

## Setup & Installation

```bash
conda create -n live2diff python==3.10.11 -y
conda activate live2diff

pip install -e .
pip install -r requirements.txt

cd co-tracker && pip install -e . && cd ..
```

## Running the Applications

**Desktop GUI (main interactive app):**
```bash
cd app/Live2D
python app_qt5.py
```

**Web backends:**
```bash
cd app/Live2D
python main.py          # FastAPI server for point annotation
python main_chat.py     # With OpenAI/DeepSeek chat integration
```

**Batch inference:**
```bash
python examples/f2lbatch.py         # Frame-to-Live2D batch processing
python examples/validatebatch.py    # Validation
```

**Data processing utilities:**
```bash
python utils/video2images.py   # Extract frames from video
python utils/images2video.py   # Assemble frames into video
```

**Training the adjacency model:**
```bash
cd app/Live2D
python train.py      # Train AdjacencyCNN
python gen_graph.py  # Build frame connectivity graph
```

**WAN model inference examples** are in `examples/wanvideo/model_inference/` — each file is a self-contained script for a specific model variant (T2V, I2V, camera control, etc.).

## Architecture

### Core Pipeline

```
Video → Frames → CoTracker (point tracks) → Graph (frame connectivity)
  → User selects start/end poses → BFS path search → WAN diffusion generation → Animated video
```

### Module Map

**`diffsynth/`** — Core diffusion synthesis library (installable as a package)
- `models/wan_video_*.py` — WAN model components: DiT (`wan_video_dit.py`), VAE, image/text encoders, motion controller, flow-line adapter (`wan_video_flow_line_adapter.py` is the key adapter for Live2D control)
- `pipelines/wan_video.py` — Main `WanVideoPipeline`; handles T2V, I2V, and flow-controlled generation modes
- `diffusion/` — Flow-matching scheduler, base pipeline, training runner, loss functions
- `core/` — VRAM management, device compatibility (NPU/GPU/CPU), attention mechanisms, model loading

**`app/Live2D/`** — Main application code
- `app_qt5.py` — PyQt5 desktop GUI; point annotation, track visualization, drag-mode pose search
- `models.py` — CoTracker3 model loading, BFS graph traversal, node position caching
- `gen_graph.py` — Pipeline that runs CoTracker on all images, trains/uses AdjacencyCNN to detect adjacent frame pairs, and writes `graph.txt`
- `train.py` — `AdjacencyCNN` model + `TrackGridDataset`; predicts whether two frames are adjacent
- `utils.py` — `gen_tracks()` for CoTracker inference, `v2i()` for video→images, track visualization

**`co-tracker/`** — Facebook's CoTracker3 as a git submodule; provides dense point tracking across video frames

**`app/Live2D/nuero/`** — Runtime data directory
- `graph.txt` — Pre-built frame connectivity graph (node pairs)
- `best_model.pth` — Pre-trained AdjacencyCNN weights
- `videos/`, `images/`, `track/`, `queries/` — Character videos, extracted frames, and CoTracker outputs

### Key Design Patterns

- **Graph-based pose navigation**: Frames are nodes; CoTracker-detected adjacency edges connect them. BFS finds the shortest path between two user-selected poses to determine the generation sequence.
- **Flow adapter control**: `WanVideoPipeline` accepts optical flow tensors via the flow-line adapter to guide frame-to-frame motion instead of relying solely on text prompts.
- **VRAM management**: `diffsynth/core/vram/` implements CPU offloading and tiling strategies; controlled via `offload_model` and `vram_management` flags in pipeline calls.
- **Dual frontend**: PyQt5 desktop GUI for local interactive use; FastAPI + HTML/JS templates for web-based annotation.
