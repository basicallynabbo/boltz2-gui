#!/usr/bin/env python3
"""
Boltz-2 GUI Application

A user-friendly graphical interface for running Boltz-2 predictions locally.
Designed with progressive disclosure: simple for beginners, powerful for experts.

Usage:
    python -m gui.app
    # or
    python gui/app.py
"""

import gradio as gr
import subprocess
import os
import sys
import tempfile
import threading
import queue
from pathlib import Path
from datetime import datetime

from .config import DEFAULTS, PRESETS, HELP_TEXT, EXAMPLE_YAML
from .yaml_builder import (
    create_protein_entry,
    create_dna_entry,
    create_rna_entry,
    create_ligand_entry,
    create_affinity_property,
    build_yaml,
    validate_protein_sequence,
    validate_dna_sequence,
    validate_rna_sequence,
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_boltz_command(
    input_path: str,
    output_dir: str,
    settings: dict,
) -> list[str]:
    """Build the boltz predict command from settings."""
    cmd = ["boltz", "predict", input_path]
    
    # Output directory
    cmd.extend(["--out_dir", output_dir])
    
    # Model selection
    if settings.get("model"):
        cmd.extend(["--model", settings["model"]])
    
    # Inference settings
    if settings.get("recycling_steps"):
        cmd.extend(["--recycling_steps", str(settings["recycling_steps"])])
    
    if settings.get("sampling_steps"):
        cmd.extend(["--sampling_steps", str(settings["sampling_steps"])])
    
    if settings.get("diffusion_samples"):
        cmd.extend(["--diffusion_samples", str(settings["diffusion_samples"])])
    
    if settings.get("step_scale"):
        cmd.extend(["--step_scale", str(settings["step_scale"])])
    
    if settings.get("max_parallel_samples"):
        cmd.extend(["--max_parallel_samples", str(settings["max_parallel_samples"])])
    
    # Hardware settings
    if settings.get("accelerator"):
        cmd.extend(["--accelerator", settings["accelerator"]])
    
    if settings.get("devices"):
        cmd.extend(["--devices", str(settings["devices"])])
    
    if settings.get("num_workers"):
        cmd.extend(["--num_workers", str(settings["num_workers"])])
    
    if settings.get("preprocessing_threads"):
        cmd.extend(["--preprocessing-threads", str(settings["preprocessing_threads"])])
    
    # MSA settings
    if settings.get("use_msa_server"):
        cmd.append("--use_msa_server")
        
        if settings.get("msa_server_url") and settings["msa_server_url"] != DEFAULTS["msa_server_url"]:
            cmd.extend(["--msa_server_url", settings["msa_server_url"]])
        
        if settings.get("msa_pairing_strategy"):
            cmd.extend(["--msa_pairing_strategy", settings["msa_pairing_strategy"]])
    
    if settings.get("max_msa_seqs"):
        cmd.extend(["--max_msa_seqs", str(settings["max_msa_seqs"])])
    
    if settings.get("subsample_msa"):
        cmd.append("--subsample_msa")
        if settings.get("num_subsampled_msa"):
            cmd.extend(["--num_subsampled_msa", str(settings["num_subsampled_msa"])])
    
    # Affinity settings
    if settings.get("affinity_mw_correction"):
        cmd.append("--affinity_mw_correction")
    
    if settings.get("sampling_steps_affinity"):
        cmd.extend(["--sampling_steps_affinity", str(settings["sampling_steps_affinity"])])
    
    if settings.get("diffusion_samples_affinity"):
        cmd.extend(["--diffusion_samples_affinity", str(settings["diffusion_samples_affinity"])])
    
    # Output settings
    if settings.get("output_format"):
        cmd.extend(["--output_format", settings["output_format"]])
    
    if settings.get("write_full_pae"):
        cmd.append("--write_full_pae")
    
    if settings.get("write_full_pde"):
        cmd.append("--write_full_pde")
    
    if settings.get("write_embeddings"):
        cmd.append("--write_embeddings")
    
    # Advanced options
    if settings.get("use_potentials"):
        cmd.append("--use_potentials")
    
    if settings.get("override"):
        cmd.append("--override")
    
    if settings.get("no_kernels"):
        cmd.append("--no_kernels")
    
    if settings.get("seed") is not None and settings["seed"] != "":
        cmd.extend(["--seed", str(settings["seed"])])
    
    return cmd


class PredictionRunner:
    """Manages running predictions in a background thread."""
    
    def __init__(self):
        self.process = None
        self.output_queue = queue.Queue()
        self.is_running = False
        self.thread = None
    
    def run(self, cmd: list[str], env: dict = None):
        """Start a prediction in a background thread."""
        self.is_running = True
        self.thread = threading.Thread(target=self._run_process, args=(cmd, env))
        self.thread.start()
    
    def _run_process(self, cmd: list[str], env: dict = None):
        """Run the prediction process."""
        try:
            process_env = os.environ.copy()
            if env:
                process_env.update(env)
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=process_env,
            )
            
            for line in iter(self.process.stdout.readline, ''):
                self.output_queue.put(line)
            
            self.process.wait()
            
            if self.process.returncode == 0:
                self.output_queue.put("\n✅ Prediction completed successfully!\n")
            else:
                self.output_queue.put(f"\n❌ Prediction failed with exit code {self.process.returncode}\n")
        
        except Exception as e:
            self.output_queue.put(f"\n❌ Error: {str(e)}\n")
        
        finally:
            self.is_running = False
    
    def get_output(self) -> str:
        """Get all available output from the queue."""
        output = []
        while not self.output_queue.empty():
            try:
                output.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return ''.join(output)
    
    def stop(self):
        """Stop the running prediction."""
        if self.process and self.is_running:
            self.process.terminate()
            self.is_running = False


# Global runner instance
runner = PredictionRunner()


# ============================================================================
# TAB 1: QUICK START
# ============================================================================

def create_quick_start_tab():
    """Create the Quick Start tab for beginners."""
    
    with gr.Column():
        gr.Markdown("""
        ## 🚀 Quick Start
        
        Get started in 3 easy steps:
        1. **Upload** your input file (YAML or FASTA)
        2. **Choose** your output directory  
        3. **Click Run!**
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                input_file = gr.File(
                    label="📁 Input File(s) - Drag multiple for Batch Mode",
                    file_types=[".yaml", ".yml", ".fasta", ".fa"],
                    type="filepath",
                    file_count="multiple",
                )
                
                output_dir = gr.Textbox(
                    label="📂 Output Directory",
                    value="./predictions",
                    placeholder="Path to save predictions",
                )
                
                with gr.Row():
                    use_msa = gr.Checkbox(
                        label="🌐 Use MSA Server",
                        value=True,
                        info="Auto-generate sequence alignments (recommended)",
                    )
                    use_potentials = gr.Checkbox(
                        label="⚡ Use Potentials",
                        value=True,
                        info="Improve physical quality of structures (recommended)",
                    )
                
                preset_dropdown = gr.Dropdown(
                    choices=[
                        ("⚡ Fast - Quick test (1 rec, 50 samp, 1 diff)", "fast"),
                        ("⚖️ Balanced - Boltz Recommended (3 rec, 200 samp, 1 diff)", "balanced"),
                        ("🎯 High Quality - Recommended + Robust (3 rec, 200 samp, 5 diff)", "high_quality"),
                        ("🔬 AlphaFold3-like - Deep sampling (10 rec, 200 samp, 25 diff)", "alphafold3_like"),
                    ],
                    value="balanced",
                    label="🎚️ Quality Preset",
                    info="Choose between speed and accuracy",
                )
                
                run_button = gr.Button(
                    "🚀 Run Prediction",
                    variant="primary",
                    size="lg",
                )
                
                stop_button = gr.Button(
                    "⏹️ Stop",
                    variant="stop",
                    visible=False,
                )
            
            with gr.Column(scale=1):
                gr.Markdown("""
                ### 💡 Tips for Beginners
                
                - **Upload multiple files** to run a batch job
                - **YAML format** is recommended over FASTA
                - Enable **MSA Server** for automatic alignments
                - **Potentials** improve structure quality
                - Start with **Balanced** preset
                - Check **Help** tab for examples
                """)
        
        status_text = gr.Markdown("**Status:** Ready")
        
        output_log = gr.Textbox(
            label="📋 Output Log",
            lines=15,
            max_lines=30,
            interactive=False,
        )
        
        # Add timer for updating output
        timer = gr.Timer(value=1, active=False)
    
    def run_prediction(input_files, output_dir, use_msa, use_potentials, preset):
        """Run a prediction with Quick Start settings."""
        if not input_files:
            return "**Status:** ❌ Please upload input file(s)", "", gr.update(active=False)
        
        # Handle Batch Processing
        import os
        import shutil
        import tempfile
        
        target_input = ""
        is_batch = False
        batch_msg = ""
        
        # Check if input_files is a list (Batch Mode) or single string
        if isinstance(input_files, list):
            if len(input_files) == 1:
                target_input = input_files[0]
            else:
                is_batch = True
                # Create a temporary directory for the batch
                batch_dir = tempfile.mkdtemp(prefix="boltz_batch_")
                for fpath in input_files:
                    try:
                        shutil.copy(fpath, batch_dir)
                    except Exception as e:
                        print(f"Error copying {fpath}: {e}")
                
                target_input = batch_dir
                batch_msg = f" (Batch of {len(input_files)} inputs)"
        else:
            target_input = input_files
        
        # Get preset settings
        preset_settings = PRESETS.get(preset, PRESETS["balanced"])["settings"].copy()
        
        # Override with user choices
        preset_settings["use_msa_server"] = use_msa
        preset_settings["use_potentials"] = use_potentials
        
        # Add defaults for other settings
        for key, value in DEFAULTS.items():
            if key not in preset_settings:
                preset_settings[key] = value
        
        # Build command
        cmd = get_boltz_command(target_input, output_dir, preset_settings)
        
        # Start prediction
        runner.run(cmd)
        
        cmd_str = ' '.join(cmd)
        return (
            f"**Status:** 🔄 Running{batch_msg}...",
            f"$ {cmd_str}\n\n",
            gr.update(active=True),
        )
    
    def update_output(current_log):
        """Update the output log with new content."""
        new_output = runner.get_output()
        if new_output:
            current_log = current_log + new_output
        
        if not runner.is_running:
            return current_log, "**Status:** ✅ Complete", gr.update(active=False)
        
        return current_log, "**Status:** 🔄 Running...", gr.update(active=True)
    
    def stop_prediction():
        """Stop the running prediction."""
        runner.stop()
        return "**Status:** ⏹️ Stopped", gr.update(active=False)
    
    run_button.click(
        fn=run_prediction,
        inputs=[input_file, output_dir, use_msa, use_potentials, preset_dropdown],
        outputs=[status_text, output_log, timer],
    )
    
    timer.tick(
        fn=update_output,
        inputs=[output_log],
        outputs=[output_log, status_text, timer],
    )
    
    stop_button.click(
        fn=stop_prediction,
        outputs=[status_text, timer],
    )
    
    return input_file, output_dir


# ============================================================================
# TAB 2: INPUT BUILDER
# ============================================================================

def create_input_builder_tab():
    """Create the Input Builder tab for visually creating YAML files."""
    
    with gr.Column():
        gr.Markdown("""
        ## 🔧 Input Builder
        
        Design complex experiments including **Proteins**, **DNA/RNA**, **Ligands**, and **Affinity Predictions**.
        
        **Supported Workflows:**
        * **🧬 Complex Modeling**: Add multiple Chains (Proteins, DNA, Ligands) to model their interaction.
        * **🎯 Affinity Prediction**: Predict binding affinity (pIC50) between a Binder and Target.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Add Sequences")
                
                entity_type = gr.Dropdown(
                    choices=["protein", "dna", "rna", "ligand"],
                    value="protein",
                    label="Entity Type",
                )
                
                chain_id = gr.Textbox(
                    label="Chain ID",
                    value="A",
                    placeholder="e.g., A, B, C",
                )
                
                sequence_input = gr.Textbox(
                    label="Sequence",
                    placeholder="Amino acid or nucleotide sequence...",
                    lines=3,
                )
                
                smiles_input = gr.Textbox(
                    label="SMILES (for ligands)",
                    placeholder="e.g., CC1=CC=CC=C1",
                    visible=False,
                )
                
                ccd_input = gr.Textbox(
                    label="CCD Code (for ligands)",
                    placeholder="e.g., ATP, SAH",
                    visible=False,
                )
                
                add_button = gr.Button("➕ Add to YAML", variant="primary")
                
                validation_msg = gr.Markdown("")
                
                gr.Markdown("---")
                
                enable_affinity = gr.Checkbox(
                    label="Enable Affinity / Binding Strength Prediction",
                    value=False,
                )
                
                affinity_binder = gr.Textbox(
                    label="Affinity Binder Chain ID",
                    placeholder="Chain ID of ligand",
                    visible=False,
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### Generated YAML")
                
                yaml_output = gr.Code(
                    label="YAML Preview",
                    language="yaml",
                    lines=20,
                    value="version: 1\nsequences: []",
                )
                
                with gr.Row():
                    clear_button = gr.Button("🗑️ Clear All")
                    export_button = gr.Button("💾 Export YAML", variant="primary")
                
                exported_file = gr.File(label="Download YAML", visible=False)
        
        with gr.Accordion("📋 Load Example", open=False):
            with gr.Row():
                example_protein = gr.Button("Protein Only")
                example_protein_ligand = gr.Button("Protein + Ligand")
                example_dimer = gr.Button("Protein Dimer")
        
        with gr.Accordion("📂 Batch Converter (Multi-FASTA)", open=False):
            gr.Markdown("""
            **Convert a Multi-FASTA file into individual YAML files.**
            Perfect for preparing batch jobs.
            """)
            with gr.Row():
                batch_fasta = gr.File(label="Upload Multi-FASTA", file_types=[".fasta", ".fa"])
                batch_type = gr.Dropdown(
                    label="Entity Type for All Sequences",
                    choices=["protein", "dna", "rna"],
                    value="protein"
                )
            
            convert_btn = gr.Button("🔄 Convert to YAML Batch", variant="secondary")
            batch_output = gr.File(label="Download ZIP", visible=False)

    # State to track added sequences
    sequences_state = gr.State([])
    properties_state = gr.State([])
    
    def convert_batch_fasta(fasta_file, entity_type):
        """Convert multi-FASTA to batch of YAMLs."""
        if not fasta_file:
            return None
            
        import tempfile
        import zipfile
        import os
        from pathlib import Path
        
        # Create temp dir
        tmp_dir = Path(tempfile.mkdtemp())
        yaml_dir = tmp_dir / "yamls"
        yaml_dir.mkdir()
        
        # Simple FASTA parser
        content = Path(fasta_file).read_text()
        entries = []
        current_header = None
        current_seq = []
        
        def save_entry(header, seq):
            if not header or not seq:
                return
            
            clean_id = header.split()[0][:20] # Short ID
            clean_seq = "".join(seq)
            
            # Create YAML content
            entry = {}
            if entity_type == "protein":
                entry = create_protein_entry(clean_id, clean_seq)
            elif entity_type == "dna":
                entry = create_dna_entry(clean_id, clean_seq)
            elif entity_type == "rna":
                entry = create_rna_entry(clean_id, clean_seq)
                
            yaml_content = build_yaml([entry])
            
            # Save file
            out_path = yaml_dir / f"{clean_id}.yaml"
            out_path.write_text(yaml_content)
            
        for line in content.splitlines():
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if current_header:
                    save_entry(current_header, current_seq)
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        
        # Save last entry
        if current_header:
            save_entry(current_header, current_seq)
            
        # Zip it
        zip_path = tmp_dir / "batch_yamls.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for yf in yaml_dir.glob("*.yaml"):
                zf.write(yf, yf.name)
                
        return str(zip_path)

    convert_btn.click(
        fn=convert_batch_fasta,
        inputs=[batch_fasta, batch_type],
        outputs=[batch_output],
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[batch_output]
    )
    
    def toggle_ligand_fields(entity_type):
        """Show/hide ligand-specific fields."""
        is_ligand = entity_type == "ligand"
        return (
            gr.update(visible=not is_ligand),  # sequence_input
            gr.update(visible=is_ligand),       # smiles_input
            gr.update(visible=is_ligand),       # ccd_input
        )
    
    entity_type.change(
        fn=toggle_ligand_fields,
        inputs=[entity_type],
        outputs=[sequence_input, smiles_input, ccd_input],
    )
    
    def toggle_affinity(enable):
        """Show/hide affinity binder field."""
        return gr.update(visible=enable)
    
    enable_affinity.change(
        fn=toggle_affinity,
        inputs=[enable_affinity],
        outputs=[affinity_binder],
    )
    
    def add_sequence(entity_type, chain_id, sequence, smiles, ccd, sequences, enable_affinity, affinity_binder, properties):
        """Add a sequence to the YAML."""
        try:
            if entity_type == "protein":
                is_valid, error = validate_protein_sequence(sequence)
                if not is_valid:
                    return sequences, properties, gr.update(), f"❌ {error}"
                entry = create_protein_entry(chain_id, sequence)
            
            elif entity_type == "dna":
                is_valid, error = validate_dna_sequence(sequence)
                if not is_valid:
                    return sequences, properties, gr.update(), f"❌ {error}"
                entry = create_dna_entry(chain_id, sequence)
            
            elif entity_type == "rna":
                is_valid, error = validate_rna_sequence(sequence)
                if not is_valid:
                    return sequences, properties, gr.update(), f"❌ {error}"
                entry = create_rna_entry(chain_id, sequence)
            
            elif entity_type == "ligand":
                if not smiles and not ccd:
                    return sequences, properties, gr.update(), "❌ Provide SMILES or CCD code"
                if smiles and ccd:
                    return sequences, properties, gr.update(), "❌ Provide only SMILES or CCD, not both"
                entry = create_ligand_entry(chain_id, smiles=smiles if smiles else None, ccd=ccd if ccd else None)
            
            else:
                return sequences, properties, gr.update(), "❌ Invalid entity type"
            
            sequences = sequences + [entry]
            
            # Handle affinity
            if enable_affinity and affinity_binder:
                properties = [create_affinity_property(affinity_binder)]
            
            yaml_str = build_yaml(sequences, properties=properties if properties else None)
            
            return sequences, properties, yaml_str, f"✅ Added {entity_type} chain {chain_id}"
        
        except Exception as e:
            return sequences, properties, gr.update(), f"❌ Error: {str(e)}"
    
    add_button.click(
        fn=add_sequence,
        inputs=[entity_type, chain_id, sequence_input, smiles_input, ccd_input, sequences_state, enable_affinity, affinity_binder, properties_state],
        outputs=[sequences_state, properties_state, yaml_output, validation_msg],
    )
    
    def clear_all():
        """Clear all sequences."""
        return [], [], "version: 1\nsequences: []", ""
    
    clear_button.click(
        fn=clear_all,
        outputs=[sequences_state, properties_state, yaml_output, validation_msg],
    )
    
    def export_yaml(yaml_content):
        """Export YAML to a file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            return gr.update(value=f.name, visible=True)
    
    export_button.click(
        fn=export_yaml,
        inputs=[yaml_output],
        outputs=[exported_file],
    )
    
    def load_example(example_key):
        """Load an example YAML."""
        return EXAMPLE_YAML[example_key], "✅ Example loaded"
    
    example_protein.click(
        fn=lambda: load_example("protein_only"),
        outputs=[yaml_output, validation_msg],
    )
    
    example_protein_ligand.click(
        fn=lambda: load_example("protein_ligand"),
        outputs=[yaml_output, validation_msg],
    )
    
    example_dimer.click(
        fn=lambda: load_example("protein_dimer"),
        outputs=[yaml_output, validation_msg],
    )
    
    return yaml_output


# ============================================================================
# TAB 3: ADVANCED SETTINGS
# ============================================================================

def create_advanced_settings_tab():
    """Create the Advanced Settings tab with all options."""
    
    with gr.Column():
        gr.Markdown("""
        ## ⚙️ Advanced Settings
        
        Fine-tune all prediction parameters. Expand sections below to modify settings.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # File inputs
                with gr.Group():
                    gr.Markdown("### 📁 Input/Output")
                    adv_input_file = gr.File(
                        label="Input File",
                        file_types=[".yaml", ".yml", ".fasta", ".fa"],
                        type="filepath",
                    )
                    adv_output_dir = gr.Textbox(
                        label="Output Directory",
                        value="./predictions",
                    )
                
                # Inference Settings
                with gr.Accordion("🔬 Inference Settings", open=True):
                    adv_model = gr.Dropdown(
                        choices=["boltz1", "boltz2"],
                        value=DEFAULTS["model"],
                        label="Model",
                        info=HELP_TEXT["model"],
                    )
                    
                    adv_recycling = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=DEFAULTS["recycling_steps"],
                        step=1,
                        label="Recycling Steps",
                        info=HELP_TEXT["recycling_steps"],
                    )
                    
                    adv_sampling = gr.Slider(
                        minimum=50,
                        maximum=500,
                        value=DEFAULTS["sampling_steps"],
                        step=10,
                        label="Sampling Steps",
                        info=HELP_TEXT["sampling_steps"],
                    )
                    
                    adv_diffusion = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=DEFAULTS["diffusion_samples"],
                        step=1,
                        label="Diffusion Samples",
                        info=HELP_TEXT["diffusion_samples"],
                    )
                    
                    adv_step_scale = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=DEFAULTS["step_scale"],
                        step=0.1,
                        label="Step Scale",
                        info=HELP_TEXT["step_scale"],
                    )
                    
                    adv_max_parallel = gr.Number(
                        value=DEFAULTS["max_parallel_samples"],
                        label="Max Parallel Samples",
                        info=HELP_TEXT["max_parallel_samples"],
                    )
                
                # Hardware Settings
                with gr.Accordion("💻 Hardware Settings", open=False):
                    adv_accelerator = gr.Dropdown(
                        choices=["gpu", "cpu", "tpu"],
                        value=DEFAULTS["accelerator"],
                        label="Accelerator",
                        info=HELP_TEXT["accelerator"],
                    )
                    
                    adv_devices = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=DEFAULTS["devices"],
                        step=1,
                        label="Number of Devices",
                        info=HELP_TEXT["devices"],
                    )
                    
                    adv_workers = gr.Number(
                        value=DEFAULTS["num_workers"],
                        label="Data Workers",
                        info=HELP_TEXT["num_workers"],
                    )
                    
                    adv_threads = gr.Number(
                        value=DEFAULTS["preprocessing_threads"],
                        label="Preprocessing Threads",
                        info=HELP_TEXT["preprocessing_threads"],
                    )
                
                # MSA Settings
                with gr.Accordion("🌐 MSA Settings", open=False):
                    adv_use_msa = gr.Checkbox(
                        value=DEFAULTS["use_msa_server"],
                        label="Use MSA Server",
                        info=HELP_TEXT["use_msa_server"],
                    )
                    
                    adv_msa_url = gr.Textbox(
                        value=DEFAULTS["msa_server_url"],
                        label="MSA Server URL",
                        info=HELP_TEXT["msa_server_url"],
                    )
                    
                    adv_msa_strategy = gr.Dropdown(
                        choices=["greedy", "complete"],
                        value=DEFAULTS["msa_pairing_strategy"],
                        label="Pairing Strategy",
                        info=HELP_TEXT["msa_pairing_strategy"],
                    )
                    
                    adv_max_msa = gr.Number(
                        value=DEFAULTS["max_msa_seqs"],
                        label="Max MSA Sequences",
                        info=HELP_TEXT["max_msa_seqs"],
                    )
                    
                    adv_subsample = gr.Checkbox(
                        value=DEFAULTS["subsample_msa"],
                        label="Subsample MSA",
                        info=HELP_TEXT["subsample_msa"],
                    )
                    
                    adv_num_subsample = gr.Number(
                        value=DEFAULTS["num_subsampled_msa"],
                        label="Number of Subsampled MSA",
                        info=HELP_TEXT["num_subsampled_msa"],
                    )
                
                # Affinity Settings
                with gr.Accordion("🎯 Affinity Settings", open=False):
                    adv_mw_correction = gr.Checkbox(
                        value=DEFAULTS["affinity_mw_correction"],
                        label="MW Correction",
                        info=HELP_TEXT["affinity_mw_correction"],
                    )
                    
                    adv_affinity_sampling = gr.Number(
                        value=DEFAULTS["sampling_steps_affinity"],
                        label="Affinity Sampling Steps",
                        info=HELP_TEXT["sampling_steps_affinity"],
                    )
                    
                    adv_affinity_samples = gr.Number(
                        value=DEFAULTS["diffusion_samples_affinity"],
                        label="Affinity Diffusion Samples",
                        info=HELP_TEXT["diffusion_samples_affinity"],
                    )
                
                # Output Settings
                with gr.Accordion("📤 Output Settings", open=False):
                    adv_format = gr.Dropdown(
                        choices=["mmcif", "pdb"],
                        value=DEFAULTS["output_format"],
                        label="Output Format",
                        info=HELP_TEXT["output_format"],
                    )
                    
                    adv_write_pae = gr.Checkbox(
                        value=DEFAULTS["write_full_pae"],
                        label="Write Full PAE",
                        info=HELP_TEXT["write_full_pae"],
                    )
                    
                    adv_write_pde = gr.Checkbox(
                        value=DEFAULTS["write_full_pde"],
                        label="Write Full PDE",
                        info=HELP_TEXT["write_full_pde"],
                    )
                    
                    adv_write_embed = gr.Checkbox(
                        value=DEFAULTS["write_embeddings"],
                        label="Write Embeddings",
                        info=HELP_TEXT["write_embeddings"],
                    )
                
                # Advanced Options
                with gr.Accordion("🔧 Advanced Options", open=False):
                    adv_potentials = gr.Checkbox(
                        value=DEFAULTS["use_potentials"],
                        label="Use Potentials",
                        info=HELP_TEXT["use_potentials"],
                    )
                    
                    adv_override = gr.Checkbox(
                        value=DEFAULTS["override"],
                        label="Override Existing",
                        info=HELP_TEXT["override"],
                    )
                    
                    adv_no_kernels = gr.Checkbox(
                        value=DEFAULTS["no_kernels"],
                        label="No Kernels (for older GPUs)",
                        info=HELP_TEXT["no_kernels"],
                    )
                    
                    adv_seed = gr.Textbox(
                        value="",
                        label="Random Seed",
                        placeholder="Leave empty for random",
                        info=HELP_TEXT["seed"],
                    )
                
                adv_run_button = gr.Button(
                    "🚀 Run with Advanced Settings",
                    variant="primary",
                    size="lg",
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Command Preview")
                
                command_preview = gr.Code(
                    label="Generated Command",
                    language="shell",
                    lines=10,
                )
                
                adv_status = gr.Markdown("**Status:** Ready")
                
                adv_output_log = gr.Textbox(
                    label="Output Log",
                    lines=15,
                    interactive=False,
                )
                
                adv_timer = gr.Timer(value=1, active=False)
    
    # Collect all settings
    all_settings = [
        adv_model, adv_recycling, adv_sampling, adv_diffusion, adv_step_scale,
        adv_max_parallel, adv_accelerator, adv_devices, adv_workers, adv_threads,
        adv_use_msa, adv_msa_url, adv_msa_strategy, adv_max_msa, adv_subsample,
        adv_num_subsample, adv_mw_correction, adv_affinity_sampling, adv_affinity_samples,
        adv_format, adv_write_pae, adv_write_pde, adv_write_embed, adv_potentials,
        adv_override, adv_no_kernels, adv_seed,
    ]
    
    def update_command_preview(input_file, output_dir, *settings):
        """Update the command preview."""
        if not input_file:
            return "# Upload an input file to see the command"
        
        settings_dict = {
            "model": settings[0],
            "recycling_steps": settings[1],
            "sampling_steps": settings[2],
            "diffusion_samples": settings[3],
            "step_scale": settings[4],
            "max_parallel_samples": settings[5],
            "accelerator": settings[6],
            "devices": settings[7],
            "num_workers": settings[8],
            "preprocessing_threads": settings[9],
            "use_msa_server": settings[10],
            "msa_server_url": settings[11],
            "msa_pairing_strategy": settings[12],
            "max_msa_seqs": settings[13],
            "subsample_msa": settings[14],
            "num_subsampled_msa": settings[15],
            "affinity_mw_correction": settings[16],
            "sampling_steps_affinity": settings[17],
            "diffusion_samples_affinity": settings[18],
            "output_format": settings[19],
            "write_full_pae": settings[20],
            "write_full_pde": settings[21],
            "write_embeddings": settings[22],
            "use_potentials": settings[23],
            "override": settings[24],
            "no_kernels": settings[25],
            "seed": settings[26],
        }
        
        cmd = get_boltz_command(input_file, output_dir, settings_dict)
        return ' \\\n  '.join(cmd)
    
    # Update preview when any setting changes
    for setting in [adv_input_file, adv_output_dir] + all_settings:
        setting.change(
            fn=update_command_preview,
            inputs=[adv_input_file, adv_output_dir] + all_settings,
            outputs=[command_preview],
        )
    
    def run_advanced(input_file, output_dir, *settings):
        """Run prediction with advanced settings."""
        if not input_file:
            return "**Status:** ❌ Please upload an input file", "", gr.update(active=False)
        
        settings_dict = {
            "model": settings[0],
            "recycling_steps": settings[1],
            "sampling_steps": settings[2],
            "diffusion_samples": settings[3],
            "step_scale": settings[4],
            "max_parallel_samples": settings[5],
            "accelerator": settings[6],
            "devices": settings[7],
            "num_workers": settings[8],
            "preprocessing_threads": settings[9],
            "use_msa_server": settings[10],
            "msa_server_url": settings[11],
            "msa_pairing_strategy": settings[12],
            "max_msa_seqs": settings[13],
            "subsample_msa": settings[14],
            "num_subsampled_msa": settings[15],
            "affinity_mw_correction": settings[16],
            "sampling_steps_affinity": settings[17],
            "diffusion_samples_affinity": settings[18],
            "output_format": settings[19],
            "write_full_pae": settings[20],
            "write_full_pde": settings[21],
            "write_embeddings": settings[22],
            "use_potentials": settings[23],
            "override": settings[24],
            "no_kernels": settings[25],
            "seed": settings[26],
        }
        
        cmd = get_boltz_command(input_file, output_dir, settings_dict)
        runner.run(cmd)
        
        return "**Status:** 🔄 Running...", f"$ {' '.join(cmd)}\n\n", gr.update(active=True)
    
    def update_adv_output(current_log):
        """Update the advanced output log."""
        new_output = runner.get_output()
        if new_output:
            current_log = current_log + new_output
        
        if not runner.is_running:
            return current_log, "**Status:** ✅ Complete", gr.update(active=False)
        
        return current_log, "**Status:** 🔄 Running...", gr.update(active=True)
    
    adv_run_button.click(
        fn=run_advanced,
        inputs=[adv_input_file, adv_output_dir] + all_settings,
        outputs=[adv_status, adv_output_log, adv_timer],
    )
    
    adv_timer.tick(
        fn=update_adv_output,
        inputs=[adv_output_log],
        outputs=[adv_output_log, adv_status, adv_timer],
    )
    
    return adv_input_file


# ============================================================================
# TAB 4: RESULTS
# ============================================================================

def create_results_tab():
    """Create a simple Results tab for analyzing confidence and affinity scores."""
    
    with gr.Column():
        gr.Markdown("""
        ## 📊 Results Analyzer
        
        Paste your output folder path to see confidence scores and affinity predictions.
        """)
        
        results_dir = gr.Textbox(
            label="📁 Output Folder Path",
            placeholder="/path/to/your/results_directory",
            value="",
            info="Enter the path to the folder containing your Boltz predictions (e.g. ./predictions)",
        )
        
        analyze_btn = gr.Button("🔍 Analyze Results", variant="primary", size="lg")
        
        results_output = gr.Markdown("*Paste your output folder path and click Analyze*")
    
    def analyze_results(folder_path):
        """Analyze prediction results from a folder."""
        import os
        import json
        import glob
        
        if not folder_path:
            return "❌ Please enter a folder path"
        
        # Expand ~ to home directory
        folder_path = os.path.expanduser(folder_path)
        
        if not os.path.exists(folder_path):
            return f"❌ Folder not found: `{folder_path}`"
        
        # Look for predictions folder
        predictions_path = os.path.join(folder_path, "predictions")
        if os.path.exists(predictions_path):
            search_path = predictions_path
        else:
            search_path = folder_path
        
        # Find all prediction subdirectories
        results_md = ""
        prediction_count = 0
        
        # Check subdirectories
        dirs_to_check = []
        if os.path.isdir(search_path):
            for item in os.listdir(search_path):
                item_path = os.path.join(search_path, item)
                if os.path.isdir(item_path):
                    dirs_to_check.append((item, item_path))
        
        # If no subdirs, check the path itself
        if not dirs_to_check:
            dirs_to_check = [(os.path.basename(search_path), search_path)]
        
        for pred_name, pred_path in dirs_to_check:
            # Find confidence files
            conf_files = glob.glob(os.path.join(pred_path, "confidence_*.json"))
            aff_files = glob.glob(os.path.join(pred_path, "affinity_*.json"))
            structure_files = glob.glob(os.path.join(pred_path, "*_model_*.cif")) + glob.glob(os.path.join(pred_path, "*_model_*.pdb"))
            
            if not conf_files and not structure_files:
                continue
            
            prediction_count += 1
            results_md += f"\n---\n\n## 🧬 Prediction: `{pred_name}`\n\n"
            
            # Structure files info
            if structure_files:
                results_md += f"**📁 Structure files:** {len(structure_files)} model(s)\n\n"
            
            # Parse confidence scores
            for conf_file in conf_files:
                try:
                    # Extract model name
                    fname = os.path.basename(conf_file)
                    model_label = fname.replace("confidence", "").replace(".json", "").replace("_", " ").strip().title()
                    if not model_label: model_label = "Model 0"
                    
                    with open(conf_file, 'r') as f:
                        conf = json.load(f)
                    
                    score = conf.get("confidence_score", 0)
                    if score > 0.9:
                        quality = "🟢 Excellent"
                    elif score > 0.7:
                        quality = "🟡 Good"  
                    elif score > 0.5:
                        quality = "🟠 Moderate"
                    else:
                        quality = "🔴 Low"
                    
                    results_md += f"""### 📈 Confidence Scores ({model_label})

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Overall Confidence** | {score:.3f} | {quality} |
| **pLDDT** | {conf.get('complex_plddt', 0):.3f} | Per-residue confidence |
| **pTM** | {conf.get('ptm', 0):.3f} | Fold accuracy |
| **ipTM** | {conf.get('iptm', 0):.3f} | Interface accuracy |
| **Ligand ipTM** | {conf.get('ligand_iptm', 0):.3f} | Protein-ligand interface |

"""
                except Exception as e:
                    results_md += f"⚠️ Error reading confidence: {str(e)}\n\n"
            
            # Parse affinity scores
            for aff_file in aff_files:
                try:
                    # Extract model name
                    fname = os.path.basename(aff_file)
                    model_label = fname.replace("affinity", "").replace(".json", "").replace("_", " ").strip().title()
                    if not model_label: model_label = "Model 0"
                    
                    with open(aff_file, 'r') as f:
                        aff = json.load(f)
                    
                    prob = aff.get("affinity_probability_binary", 0)
                    binding = "✅ Likely binder" if prob > 0.5 else "❌ Likely non-binder"
                    
                    results_md += f"""### 🎯 Affinity Prediction ({model_label})

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Binding Probability** | {prob:.3f} | {binding} |
| **Affinity (pIC50)** | {aff.get('affinity_pred_value', 0):.3f} | Higher = stronger binding |

"""
                except Exception as e:
                    results_md += f"⚠️ Error reading affinity: {str(e)}\n\n"
        
        if prediction_count == 0:
            return f"❌ No predictions found in `{folder_path}`\n\nMake sure the folder contains a `predictions/` subdirectory."
        
        return f"# ✅ Found {prediction_count} prediction(s)\n" + results_md
    
    analyze_btn.click(
        fn=analyze_results,
        inputs=[results_dir],
        outputs=[results_output],
    )
    
    return results_dir


# ============================================================================
# TAB 5: HELP
# ============================================================================

def create_help_tab():
    """Create the Help tab with documentation and examples."""
    
    with gr.Column():
        gr.Markdown("""
        ## 📚 Help & Documentation
        
        ### Getting Started
        
        Boltz-2 predicts 3D structures of biomolecular complexes including:
        - **Proteins** (amino acid sequences)
        - **DNA/RNA** (nucleotide sequences)  
        - **Small molecules/Ligands** (SMILES or CCD codes)
        
        It can also predict **binding affinity** between proteins and small molecules.
        
        ---
        
        ### Quick Start Guide
        
        1. **Prepare your input file** in YAML format (see examples below)
        2. Go to the **Quick Start** tab
        3. Upload your file and click **Run Prediction**
        4. Results will be saved to the output directory
        
        ---
        
        ### Input File Format (YAML)
        
        The basic structure is:
        
        ```yaml
        version: 1
        sequences:
          - protein:
              id: A  # Chain identifier
              sequence: MVTPEGN...  # Amino acid sequence
          - ligand:
              id: B
              smiles: 'CC1=CC=CC=C1'  # Or use ccd: ATP
        properties:  # Optional: for affinity prediction
          - affinity:
              binder: B  # Chain ID of the ligand
        ```
        
        ---
        
        ### Understanding the Output
        
        After prediction, you'll find:
        
        | File | Description |
        |------|-------------|
        | `*_model_0.cif` | Predicted structure (best model) |
        | `confidence_*.json` | Confidence scores (pLDDT, pTM, ipTM) |
        | `affinity_*.json` | Binding affinity predictions (if requested) |
        | `pae_*.npz` | Predicted Aligned Error matrix |
        
        **Confidence Scores:**
        - **pLDDT** (0-1): Per-residue confidence. >0.7 is good, >0.9 is excellent
        - **pTM** (0-1): Predicted TM-score for overall fold accuracy
        - **ipTM** (0-1): Interface pTM for complex modeling accuracy
        
        ---
        
        ### Tips for Best Results
        
        1. **Use MSA Server**: Improves accuracy significantly
        2. **Enable Potentials**: Better physical quality of structures
        3. **More Samples**: Increase `diffusion_samples` for important predictions
        4. **Check Confidence**: Low confidence regions may be disordered
        
        ---
        
        ### Troubleshooting
        
        **"CUDA out of memory"**
        - Reduce `max_parallel_samples`
        - Use smaller `diffusion_samples`
        
        **"cuequivariance error"**
        - Enable "No Kernels" option for older GPUs
        
        **Slow predictions**
        - Use "Fast" preset for testing
        - Ensure GPU is being used (check `accelerator` setting)
        
        ---
        
        ### Links
        
        - [Boltz-2 Paper](https://doi.org/10.1101/2025.06.14.659707)
        - [Boltz-1 Paper](https://doi.org/10.1101/2024.11.19.624167)
        - [GitHub Repository](https://github.com/jwohlwend/boltz)
        - [Slack Community](https://boltz.bio/join-slack)
        """)
        
        with gr.Accordion("📋 Example: Protein Only", open=False):
            gr.Code(
                value=EXAMPLE_YAML["protein_only"],
                language="yaml",
                label="protein_only.yaml",
            )
        
        with gr.Accordion("📋 Example: Protein + Ligand with Affinity", open=False):
            gr.Code(
                value=EXAMPLE_YAML["protein_ligand"],
                language="yaml",
                label="protein_ligand.yaml",
            )
        
        with gr.Accordion("📋 Example: Protein Dimer", open=False):
            gr.Code(
                value=EXAMPLE_YAML["protein_dimer"],
                language="yaml",
                label="protein_dimer.yaml",
            )


# ============================================================================
# MAIN APPLICATION
# ============================================================================

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def create_app():
    """Create the main Gradio application."""
    
    # Custom CSS for that "YC Startup" look
    yc_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    body {
        font-family: 'Inter', sans-serif !important;
        background-color: #f8fafc;
    }
    
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.025em;
    }
    
    /* Elegant Cards */
    .block {
        border: 1px solid #e2e8f0;
        border-radius: 12px !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.01), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        background: white;
        overflow: hidden;
    }
    
    /* Clean Buttons */
    button.primary {
        background: #0f172a !important; /* Slate 900 */
        color: white !important;
        border-radius: 8px !important;
        font-weight: 500;
        transition: all 0.2s;
        border: none !important;
    }
    button.primary:hover {
        background: #1e293b !important; /* Slate 800 */
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    button.secondary {
        background: white !important;
        border: 1px solid #e2e8f0 !important;
        color: #475569 !important; /* Slate 600 */
        border-radius: 8px !important;
        font-weight: 500;
        transition: all 0.2s;
    }
    button.secondary:hover {
        background: #f8fafc !important;
        border-color: #cbd5e1 !important;
        color: #0f172a !important;
    }
    
    /* Inputs */
    input, textarea, select {
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
    }
    input:focus, textarea:focus, select:focus {
        border-color: #0f172a !important;
        ring: 2px solid #0f172a !important;
    }
    
    /* Tabs */
    .tabs {
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 24px;
        background: transparent;
    }
    .tab-nav {
        border: none !important;
        background: transparent !important;
    }
    .tab-nav button {
        font-weight: 500;
        color: #64748b;
    }
    .tab-nav button.selected {
        color: #0f172a;
        font-weight: 600;
        border-bottom: 2px solid #0f172a !important;
        background: transparent !important;
    }
    
    /* Header */
    .header-logo {
        font-size: 24px;
        font-weight: 700;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
    .header-subtitle {
        color: #64748b;
        font-size: 16px;
        font-weight: 400;
    }
    
    /* Footer */
    footer {
        display: none !important;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #e2e8f0;
        color: #94a3b8;
        font-size: 0.875rem;
    }
    """
    
    with gr.Blocks(
        title="Boltz-2",
        # css and theme moved to launch() for Gradio 6.0 compatibility
    ) as app:
        
        with gr.Row(elem_classes="header-container"):
            with gr.Column(scale=1):
                gr.HTML("""
                <div style="padding: 24px 0;">
                    <div class="header-logo">
                        <span style="background: #0f172a; color: white; width: 32px; height: 32px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 18px;">⚡</span>
                        Boltz-2
                    </div>
                    <div class="header-subtitle">
                        Biomolecular structure prediction, simplified.
                    </div>
                </div>
                """)
        
        with gr.Tabs():
            with gr.Tab("🚀 Quick Start"):
                create_quick_start_tab()
            
            with gr.Tab("🔧 Input Builder"):
                create_input_builder_tab()
            
            with gr.Tab("⚙️ Advanced Settings"):
                create_advanced_settings_tab()
            
            with gr.Tab("📊 Results"):
                create_results_tab()
            
            with gr.Tab("📚 Help"):
                create_help_tab()
        
        gr.HTML("""
        <div style="text-align: center; margin-top: 40px; padding: 20px; color: #94a3b8; border-top: 1px solid #e2e8f0;">
            <p>Boltz-2 GUI v0.1.0 &middot; <a href="https://github.com/jwohlwend/boltz" target="_blank" style="color: #64748b; text-decoration: none;">GitHub</a> &middot; MIT License</p>
        </div>
        """)
    
    return app


def main():
    """Launch the Boltz-2 GUI application."""
    
    # Custom CSS for that "YC Startup" look
    yc_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    body {
        font-family: 'Inter', sans-serif !important;
        background-color: #f8fafc;
    }
    
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.025em;
    }
    
    /* Elegant Cards */
    .block {
        border: 1px solid #e2e8f0;
        border-radius: 12px !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.01), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        background: white;
        overflow: hidden;
    }
    
    /* Clean Buttons */
    button.primary {
        background: #0f172a !important; /* Slate 900 */
        color: white !important;
        border-radius: 8px !important;
        font-weight: 500;
        transition: all 0.2s;
        border: none !important;
    }
    button.primary:hover {
        background: #1e293b !important; /* Slate 800 */
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    button.secondary {
        background: white !important;
        border: 1px solid #e2e8f0 !important;
        color: #475569 !important; /* Slate 600 */
        border-radius: 8px !important;
        font-weight: 500;
        transition: all 0.2s;
    }
    button.secondary:hover {
        background: #f8fafc !important;
        border-color: #cbd5e1 !important;
        color: #0f172a !important;
    }
    
    /* Inputs */
    input, textarea, select {
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
    }
    input:focus, textarea:focus, select:focus {
        border-color: #0f172a !important;
        ring: 2px solid #0f172a !important;
    }
    
    /* Tabs */
    .tabs {
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 24px;
        background: transparent;
    }
    .tab-nav {
        border: none !important;
        background: transparent !important;
    }
    .tab-nav button {
        font-weight: 500;
        color: #64748b;
    }
    .tab-nav button.selected {
        color: #0f172a;
        font-weight: 600;
        border-bottom: 2px solid #0f172a !important;
        background: transparent !important;
    }
    
    /* Header */
    .header-logo {
        font-size: 24px;
        font-weight: 700;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
    .header-subtitle {
        color: #64748b;
        font-size: 16px;
        font-weight: 400;
    }
    
    /* Footer */
    footer {
        display: none !important;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #e2e8f0;
        color: #94a3b8;
        font-size: 0.875rem;
    }
    """
    
    # Using GoogleFont for Inter
    try:
        font = gr.themes.GoogleFont("Inter")
    except Exception:
        font = "Inter"  # Fallback
        
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        share=False,
        inbrowser=True,
        allowed_paths=["/"],
        css=yc_css,
        theme=gr.themes.Soft(
            primary_hue="slate",
            secondary_hue="slate",
            text_size="sm",
            spacing_size="sm",
            radius_size="md",
            font=[font, 'ui-sans-serif', 'system-ui', 'sans-serif'],
        ),
    )


if __name__ == "__main__":
    main()
