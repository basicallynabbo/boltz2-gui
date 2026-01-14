# Boltz-2 GUI

A user-friendly graphical interface for running Boltz-2 predictions locally.

## 🚀 Quick Launch

```bash
# Make sure you're in the boltz directory and conda environment is active
conda activate boltz

# Launch the GUI
python run_gui.py
```

The GUI will automatically open in your browser at `http://localhost:7860`

## Alternative Launch Methods

```bash
# From the boltz root directory:
python -m gui.app

# Or directly:
python gui/app.py
```

## Requirements

- Python 3.10-3.12
- Gradio (installed automatically with boltz)

If Gradio is not installed:

```bash
pip install gradio
```

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

## Stopping the GUI

Press `Ctrl+C` in the terminal to stop the server.
