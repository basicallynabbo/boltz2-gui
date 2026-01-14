# Boltz-2 GUI

A user-friendly graphical interface for running Boltz-2 predictions locally.

## 📦 Installation & Setup

Follow these steps to set up Boltz-2 and the GUI from scratch.

### 1. Clone the Repository

```bash
git clone https://github.com/basicallynabbo/boltz2-gui.git
cd boltz2-gui
```

### 2. Create Conda Environment

Boltz requires Python 3.10, 3.11, or 3.12 (Python 3.13+ is not supported).

```bash
conda create -n boltz python=3.12
conda activate boltz
```

### 3. Install Boltz-2

Install the package in editable mode:

```bash
pip install -e .
```

### 4. Install GUI Requirements

Install Gradio for the user interface:

```bash
pip install gradio
```

### 5. Run the GUI

Launch the application:

```bash
python run_gui.py
```

The GUI will automatically open in your browser at `http://localhost:7860`.

---

## Features

| Tab | Description |
|-----|-------------|
| 🚀 **Quick Start** | Upload file → Select preset → Click Run! |
| 🔧 **Input Builder** | Create YAML files visually without coding |
| ⚙️ **Advanced Settings** | Access all 25+ prediction parameters |
| 📊 **Results** | Analyze confidence scores and affinity from any folder |
| 📚 **Help** | Documentation, examples, troubleshooting |

## Tips for Beginners

1. Start with the **Quick Start** tab
2. Enable **"Use MSA Server"** for automatic sequence alignments
3. Enable **"Use Potentials"** for better structure quality
4. Use the **"Balanced"** preset for most cases

## Technical Details: Quality Presets

The GUI provides four carefully tuned presets for different use cases:

| Preset | Description | Recycling Loop | Diffusion Samples | Sampling Steps | Step Scale |
|--------|-------------|----------------|-------------------|----------------|------------|
| **⚡ Fast** | Quick sanity checks | 1 | 1 | 50 | 1.638 |
| **⚖️ Balanced** | **Boltz Recommended** Defaults | 3 | 1 | 200 | 1.638 |
| **🎯 High Quality** | Recommended settings with **5x more samples** | 3 | 5 | 200 | 1.638 |
| **🔬 AlphaFold3-like** | Heavy sampling (mimics AF3 params) | 10 | 25 | 200 | 1.638 |

**Notes:**

- **Step Scale 1.638**: The officially recommended temperature for diffusion sampling.
- **Balanced**: Uses the exact default parameters recommended by the Boltz authors.
- **High Quality**: Increases robustness by generating 5 distinct diffusion samples and picking the best one (by confidence), without changing the underlying model parameters.

## Stopping the GUI

Press `Ctrl+C` in the terminal to stop the server.
