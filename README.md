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

### 3. Install PyTorch (with GPU Support)

Essential for fast predictions.

```bash
# Install PyTorch with CUDA 12.4 support
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install Boltz-2

Install the package in editable mode:

```bash
pip install -e .
```

### 5. Install GUI Requirements

Install Gradio for the user interface:

```bash
pip install gradio
```

### 6. Run the GUI

Launch the application:

```bash
conda activate boltz
python run_gui.py
```

The GUI will automatically open in your browser at `http://localhost:7860`.

---

## 🚀 Quick Start - Job Queue

The GUI uses a **simple job queue system**:

1. **Upload** your YAML or FASTA file(s)
2. **Name** your job (e.g., `protein_binding_study`)
3. **Add to Queue** - jobs run automatically one-by-one

### Why a Queue?

- **Run multiple experiments** without waiting
- **Each job gets its own folder** at `~/boltz-predictions/{job_name}/`
- **Jobs run sequentially** (GPU handles one at a time)
- **Queue your next jobs** while one is running

### Default Settings

The GUI uses **Boltz recommended settings** automatically:

- ✅ MSA Server enabled (auto-generates alignments)
- ✅ Potentials enabled (better structure quality)
- ✅ Balanced preset (3 recycling, 200 sampling steps)

---

## Features

| Tab | Description |
|-----|-------------|
| 🚀 **Quick Start** | Upload → Name → Add to Queue |
| 🔧 **Input Builder** | Create YAML files visually without coding |
| ⚙️ **Advanced Settings** | Access all 25+ prediction parameters |
| 📊 **Results** | Analyze scores + **Export to CSV** for pandas/matplotlib |
| 📚 **Help** | Documentation, examples, troubleshooting |

---

## 📊 Results & CSV Export

Analyze your predictions and export data for further analysis:

1. Go to the **Results** tab
2. Select a prediction folder
3. Click **"🔍 Analyze Results"** to see scores
4. Click **"📥 Export to CSV"** to download data

### CSV Columns

| Column | Description |
|--------|-------------|
| `prediction_name` | Name of the prediction |
| `model_id` | Model number |
| `confidence_score` | Overall confidence |
| `plddt` | Per-residue confidence |
| `ptm` | Fold accuracy |
| `iptm` | Interface accuracy |
| `ligand_iptm` | Protein-ligand interface |
| `binding_probability` | Affinity probability |
| `affinity_pIC50` | Binding strength |

### Example with Pandas

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("boltz_results_xxx.csv")
df.plot(x="prediction_name", y="confidence_score", kind="bar")
plt.show()
```

---

## 🔥 Batch Processing

You can run predictions on multiple structures at once:

1. Drag & Drop **multiple YAML or FASTA files** into the upload box.
2. Give the batch a **job name**.
3. Click **Add to Queue**.

The GUI will automatically process all files as a single batch job.

### 🛠️ Batch Input Generation

Need to run 100 predictions? You don't need to create YAMLs manually.

1. Go to **Input Builder** > **Batch Converter**.
2. Upload a **Multi-FASTA** file (e.g., one file with 100 sequences).
3. Click **Convert to YAML Batch**.
4. Download the **ZIP file**, extract it, and use the contents in **Quick Start**.

### 🧬 Combinatorial Peptide-Receptor Docking

Generate all combinations of peptide-receptor pairs for bulk docking experiments:

1. Go to **Input Builder** > **Combinatorial Peptide-Receptor Docking**.
2. Upload **`peptide.fasta`** containing your de novo peptide sequences.
3. Upload **`receptor.fasta`** containing your receptor protein sequences.
4. Click **Generate Combinatorial YAMLs**.
5. Download the **ZIP file** containing N×M YAML files (N peptides × M receptors).

**Example:** 10 peptides × 5 receptors = 50 YAML files, each with:

- Chain A = Receptor
- Chain B = Peptide

## Stopping the GUI

Press `Ctrl+C` in the terminal to stop the server.
