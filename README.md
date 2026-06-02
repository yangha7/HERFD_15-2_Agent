# 🔬 HERFD Agent — SSRL BL 15-2

An AI-powered chat agent for analyzing **HERFD-XAS** (High Energy Resolution Fluorescence Detected X-ray Absorption Spectroscopy) data collected at **SSRL beamline 15-2**.

Talk to your data in natural language — plot spectra, compare samples, normalize XANES, identify elements, and more.

## Features

- **Scan Classification** — Automatically identifies XAS scans vs. alignment, emission, and aborted scans
- **Smart Averaging** — Averages multiple XAS scans with heterogeneous energy grids
- **XANES Normalization** — Athena-style pre-edge subtraction and post-edge normalization with flattening
- **Element Identification** — Guesses the element/edge from metadata and energy range
- **Interactive Chat** — Natural language interface powered by LLM (supports CBORG, OpenAI, Gemini, Claude)
- **File Explorer** — Left sidebar with data directory tree; double-click to paste filenames
- **Command History** — Up/down arrow keys to recall previous commands
- **Plotting** — Inline plots with zoom (x and y axis), auto-scaling, offset stacking, and custom styling
- **Batch Processing** — Process all samples at once (average + normalize + export)

## Supported Data

- **Format**: SPEC-format `.dat` files (standard at SSRL)
- **Energy Range**: 4.5–37 keV (K-edges for 3d/4d metals, L-edges for 5d metals, lanthanides, actinides)
- **Signal**: `vortDT / I0` (deadtime-corrected vortex detector / ion chamber)
- **Organization**: Sample directories containing multiple scan files

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your LLM API key

Copy `.env.example` to `.env` and set your API key:

```bash
cp .env.example .env
# Edit .env and set one of:
#   CBORG_API_KEY=your-key    (for LBL users)
#   OPENAI_API_KEY=your-key   (for OpenAI)
#   GEMINI_API_KEY=your-key   (for Google Gemini)
#   ANTHROPIC_API_KEY=your-key (for Anthropic Claude)
```

### 3. Add your data

Place your HERFD data in a subdirectory (e.g., `2026-05_Yano/`). The expected structure is:

```
2026-05_Yano/
├── 20260519_RuO2-TiO2_100_normal_dir/
│   ├── 20260519_RuO2-TiO2_100_normal_001.dat
│   ├── 20260519_RuO2-TiO2_100_normal_002.dat
│   └── ...
├── 20260521_RuO2_Ref_dir/
│   └── ...
└── readme.txt
```

Set the data directory in `.env` if different from the default:
```
HERFD_DATA_DIR=2026-05_Yano
```

### 4. Launch the agent

```bash
python run_agent.py
```

This starts the server and opens your browser at `http://localhost:5051`.

## Usage

### Chat Interface

Type natural language commands in the chat:

- **"List samples"** — Show all sample directories
- **"Plot 110_normal"** — Average and plot a sample's HERFD spectrum
- **"Compare 100_normal and 110_normal"** — Overlay spectra
- **"Normalize RuO2_Ref"** — XANES normalization with flattening
- **"What element is this?"** — Identify element from energy range
- **"Show all scans for 100_grazing"** — Check individual scan quality
- **"Process all samples"** — Batch average + normalize + export
- **"Zoom in from 22110 to 22150 eV"** — Energy range zoom
- **"Set y axis from 0 to 1.5"** — Y-axis range control

### Command-Line Interface

```bash
# Scan summaries
python herfd_utils.py --summary

# Batch process all samples
python herfd_utils.py --process-all

# Process a specific sample
python herfd_utils.py --process 2026-05_Yano/20260519_RuO2-TiO2_110_normal_dir

# Average specific scans
python herfd_utils.py --average 2026-05_Yano/20260519_RuO2-TiO2_110_normal_dir --scans 6 7 8 9 10
```

### Python API

```python
import herfd_utils as hu

# Average a sample
energy, mu, files = hu.average_sample_dir("2026-05_Yano/20260519_RuO2-TiO2_110_normal_dir")

# Normalize
norm = hu.normalize_xanes(energy, mu)
print(f"E0 = {norm['e0']:.1f} eV")

# Use flattened spectrum for display
mu_flat = norm["flat"]  # post-edge ≈ 1.0

# Derivatives
deriv1 = hu.smooth_derivative(energy, mu, order=1)

# Identify element
candidates = hu.identify_element_from_energy((energy.min(), energy.max()), "RuO2-TiO2")
```

## File Structure

```
├── chat_app.py          # Flask chat web app (15 LLM tools)
├── herfd_utils.py       # Core data processing utilities
├── run_agent.py         # One-click launcher
├── requirements.txt     # Python dependencies
├── .env.example         # LLM API key template
├── .gitignore           # Git exclusions
└── README.md            # This file
```

## Normalization Details

The XANES normalization follows the **Athena-style** approach:

1. **E0 detection** — Maximum of smoothed 1st derivative
2. **Pre-edge fit** — Linear fit to E0−50 to E0−15 eV
3. **Post-edge fit** — Quadratic fit to E0+30 to end of data
4. **Edge step** — post_edge(E0) − pre_edge(E0)
5. **Normalized μ** — (μ − pre_edge) / edge_step
6. **Flattening** — Removes post-edge slope so it oscillates around 1.0

The **flattened** spectrum (`mu_flat`) is used by default, which is important for hard X-ray data with wide energy ranges where the raw normalized spectrum shows natural post-edge decay.

## License

MIT
