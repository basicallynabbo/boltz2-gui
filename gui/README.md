# Boltz-2 GUI

A user-friendly graphical interface for running Boltz-2 predictions locally.

## 📦 Installation & Setup

**Already have Boltz installed?** You're 99% there!

1. **Activate your Boltz environment:**

   ```bash
   conda activate boltz
   ```

2. **Install the GUI dependency:**

   ```bash
   pip install gradio
   ```

3. **Launch the App:**

   ```bash
   python run_gui.py
   ```

The GUI will automatically open in your browser at `http://localhost:7860`.

---

## Alternative Launch Methods

## Features

| Tab | Description |
|-----|-------------|
| 🚀 **Quick Start** | Upload file → Select preset → Click Run! |
| 🔧 **Input Builder** | Create YAML files visually without coding |
| ⚙️ **Advanced Settings** | Access all 25+ prediction parameters |
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
