"""Configuration and default values for Boltz-2 GUI."""

# Default parameter values
DEFAULTS = {
    # Inference Settings
    "model": "boltz2",
    "recycling_steps": 3,
    "sampling_steps": 200,
    "diffusion_samples": 1,
    "step_scale": 1.5,
    "max_parallel_samples": 5,
    
    # Hardware Settings
    "accelerator": "gpu",
    "devices": 1,
    "num_workers": 2,
    "preprocessing_threads": 1,
    
    # MSA Settings
    "use_msa_server": True,
    "msa_server_url": "https://api.colabfold.com",
    "msa_pairing_strategy": "greedy",
    "max_msa_seqs": 8192,
    "subsample_msa": True,
    "num_subsampled_msa": 1024,
    
    # Affinity Settings
    "affinity_mw_correction": False,
    "sampling_steps_affinity": 200,
    "diffusion_samples_affinity": 5,
    
    # Output Settings
    "output_format": "mmcif",
    "write_full_pae": True,
    "write_full_pde": False,
    "write_embeddings": False,
    
    # Advanced Options
    "use_potentials": True,
    "override": False,
    "no_kernels": False,
    "seed": None,
}

# Preset configurations
PRESETS = {
    "fast": {
        "name": "⚡ Fast",
        "description": "Quick test (1 rec, 50 samp, 1 diff)",
        "settings": {
            "recycling_steps": 1,
            "sampling_steps": 50,
            "diffusion_samples": 1,
            "step_scale": 1.638,
            "use_potentials": False,
        }
    },
    "balanced": {
        "name": "⚖️ Balanced",
        "description": "Boltz Recommended (3 rec, 200 samp, 1 diff)",
        "settings": {
            "recycling_steps": 3,
            "sampling_steps": 200,
            "diffusion_samples": 1,
            "step_scale": 1.638,
            "use_potentials": True,
        }
    },
    "high_quality": {
        "name": "🎯 High Quality",
        "description": "Recommended + Robust (3 rec, 200 samp, 5 diff)",
        "settings": {
            "recycling_steps": 3,
            "sampling_steps": 200,
            "diffusion_samples": 5,
            "step_scale": 1.638,
            "use_potentials": True,
        }
    },
    "alphafold3_like": {
        "name": "🔬 AlphaFold3-like",
        "description": "Deep sampling (10 rec, 200 samp, 25 diff)",
        "settings": {
            "recycling_steps": 10,
            "sampling_steps": 200,
            "diffusion_samples": 25,
            "step_scale": 1.638,
            "use_potentials": True,
        }
    },
}

# Help text for each option
HELP_TEXT = {
    "model": "Choose between Boltz-1 and Boltz-2 models. Boltz-2 includes affinity prediction.",
    
    "recycling_steps": "Number of recycling iterations. More steps = better quality but slower. Default: 3, Max quality: 10",
    
    "sampling_steps": "Number of diffusion sampling steps. Higher = better structure quality. Default: 200",
    
    "diffusion_samples": "Number of structure samples to generate. More samples = better chance of getting accurate structure. Default: 1",
    
    "step_scale": "Controls diversity in diffusion sampling. Lower = more diverse structures (range: 1.0-2.0). Default: 1.5",
    
    "max_parallel_samples": "Maximum samples to predict simultaneously. Increase if you have more GPU memory.",
    
    "accelerator": "Hardware to use: GPU (fastest), CPU (slowest), or TPU.",
    
    "devices": "Number of GPUs/devices to use. More = faster for large batches.",
    
    "num_workers": "Number of data loading workers. More can speed up preprocessing.",
    
    "preprocessing_threads": "Threads for preprocessing. More = faster input processing.",
    
    "use_msa_server": "Automatically generate MSA (Multiple Sequence Alignment) using the ColabFold server. Recommended for most users.",
    
    "msa_server_url": "URL of the MSA server. Default is ColabFold public server.",
    
    "msa_pairing_strategy": "Strategy for pairing MSA sequences: 'greedy' (faster) or 'complete' (more thorough).",
    
    "max_msa_seqs": "Maximum number of MSA sequences to use. More = potentially better accuracy but slower.",
    
    "subsample_msa": "Whether to randomly sample a subset of MSA sequences. Helps with memory and speed.",
    
    "num_subsampled_msa": "How many MSA sequences to sample if subsampling is enabled.",
    
    "affinity_mw_correction": "Apply molecular weight correction to affinity predictions. May improve accuracy for some cases.",
    
    "sampling_steps_affinity": "Diffusion steps specifically for affinity prediction.",
    
    "diffusion_samples_affinity": "Number of samples for affinity prediction. More = more robust estimate.",
    
    "output_format": "Output structure file format: mmCIF (modern, recommended) or PDB (legacy).",
    
    "write_full_pae": "Save the full Predicted Aligned Error matrix. Useful for confidence analysis.",
    
    "write_full_pde": "Save the full Predicted Distance Error matrix.",
    
    "write_embeddings": "Save internal model embeddings. Useful for downstream ML tasks.",
    
    "use_potentials": "Apply inference-time potentials to improve physical plausibility of structures. Recommended.",
    
    "override": "Overwrite existing predictions if found in output directory.",
    
    "no_kernels": "Disable cuEquivariance kernels. Use this for older GPUs that don't support them.",
    
    "seed": "Random seed for reproducibility. Leave empty for random.",
}

# Example YAML templates
EXAMPLE_YAML = {
    "protein_only": """version: 1
sequences:
  - protein:
      id: A
      sequence: MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ
""",
    
    "protein_ligand": """version: 1
sequences:
  - protein:
      id: A
      sequence: MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAE
  - ligand:
      id: B
      smiles: 'CC1=CC=CC=C1'
properties:
  - affinity:
      binder: B
""",
    
    "protein_dimer": """version: 1
sequences:
  - protein:
      id: [A, B]
      sequence: MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAE
""",
}
