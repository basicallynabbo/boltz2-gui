"""YAML input file builder utilities for Boltz-2 GUI."""

import yaml
from typing import Optional


def create_protein_entry(
    chain_id: str | list[str],
    sequence: str,
    msa_path: Optional[str] = None,
    modifications: Optional[list[dict]] = None,
    cyclic: bool = False,
) -> dict:
    """Create a protein entry for the YAML file.
    
    Args:
        chain_id: Unique chain identifier(s). Use list for multiple identical chains.
        sequence: Amino acid sequence.
        msa_path: Optional path to pre-computed MSA (.a3m file).
        modifications: Optional list of modifications [{position: int, ccd: str}].
        cyclic: Whether the protein is cyclic.
    
    Returns:
        Dictionary representing the protein entry.
    """
    entry = {
        "protein": {
            "id": chain_id,
            "sequence": sequence.upper().strip(),
        }
    }
    
    if msa_path:
        entry["protein"]["msa"] = msa_path
    
    if modifications:
        entry["protein"]["modifications"] = modifications
    
    if cyclic:
        entry["protein"]["cyclic"] = True
    
    return entry


def create_dna_entry(
    chain_id: str | list[str],
    sequence: str,
    modifications: Optional[list[dict]] = None,
    cyclic: bool = False,
) -> dict:
    """Create a DNA entry for the YAML file.
    
    Args:
        chain_id: Unique chain identifier(s).
        sequence: Nucleotide sequence (ATCG).
        modifications: Optional list of modifications.
        cyclic: Whether the DNA is cyclic.
    
    Returns:
        Dictionary representing the DNA entry.
    """
    entry = {
        "dna": {
            "id": chain_id,
            "sequence": sequence.upper().strip(),
        }
    }
    
    if modifications:
        entry["dna"]["modifications"] = modifications
    
    if cyclic:
        entry["dna"]["cyclic"] = True
    
    return entry


def create_rna_entry(
    chain_id: str | list[str],
    sequence: str,
    modifications: Optional[list[dict]] = None,
    cyclic: bool = False,
) -> dict:
    """Create an RNA entry for the YAML file.
    
    Args:
        chain_id: Unique chain identifier(s).
        sequence: Nucleotide sequence (AUCG).
        modifications: Optional list of modifications.
        cyclic: Whether the RNA is cyclic.
    
    Returns:
        Dictionary representing the RNA entry.
    """
    entry = {
        "rna": {
            "id": chain_id,
            "sequence": sequence.upper().strip(),
        }
    }
    
    if modifications:
        entry["rna"]["modifications"] = modifications
    
    if cyclic:
        entry["rna"]["cyclic"] = True
    
    return entry


def create_ligand_entry(
    chain_id: str | list[str],
    smiles: Optional[str] = None,
    ccd: Optional[str] = None,
) -> dict:
    """Create a ligand entry for the YAML file.
    
    Args:
        chain_id: Unique chain identifier(s).
        smiles: SMILES string (mutually exclusive with ccd).
        ccd: CCD code (mutually exclusive with smiles).
    
    Returns:
        Dictionary representing the ligand entry.
    
    Raises:
        ValueError: If both or neither smiles and ccd are provided.
    """
    if smiles and ccd:
        raise ValueError("Provide either smiles or ccd, not both.")
    if not smiles and not ccd:
        raise ValueError("Must provide either smiles or ccd.")
    
    entry = {
        "ligand": {
            "id": chain_id,
        }
    }
    
    if smiles:
        entry["ligand"]["smiles"] = smiles
    else:
        entry["ligand"]["ccd"] = ccd
    
    return entry


def create_bond_constraint(
    atom1_chain: str,
    atom1_residue: int,
    atom1_name: str,
    atom2_chain: str,
    atom2_residue: int,
    atom2_name: str,
) -> dict:
    """Create a covalent bond constraint.
    
    Args:
        atom1_chain: Chain ID of first atom.
        atom1_residue: Residue index (1-indexed) of first atom.
        atom1_name: Atom name of first atom.
        atom2_chain: Chain ID of second atom.
        atom2_residue: Residue index (1-indexed) of second atom.
        atom2_name: Atom name of second atom.
    
    Returns:
        Dictionary representing the bond constraint.
    """
    return {
        "bond": {
            "atom1": [atom1_chain, atom1_residue, atom1_name],
            "atom2": [atom2_chain, atom2_residue, atom2_name],
        }
    }


def create_pocket_constraint(
    binder_chain: str,
    contacts: list[tuple[str, int | str]],
    max_distance: float = 6.0,
    force: bool = False,
) -> dict:
    """Create a pocket (binding site) constraint.
    
    Args:
        binder_chain: Chain ID of the binder molecule.
        contacts: List of (chain_id, residue_index or atom_name) tuples.
        max_distance: Maximum distance in Angstroms (4-20, default 6).
        force: Whether to enforce with a potential.
    
    Returns:
        Dictionary representing the pocket constraint.
    """
    constraint = {
        "pocket": {
            "binder": binder_chain,
            "contacts": [list(c) for c in contacts],
        }
    }
    
    if max_distance != 6.0:
        constraint["pocket"]["max_distance"] = max_distance
    
    if force:
        constraint["pocket"]["force"] = True
    
    return constraint


def create_contact_constraint(
    token1_chain: str,
    token1_residue: int | str,
    token2_chain: str,
    token2_residue: int | str,
    max_distance: float = 6.0,
    force: bool = False,
) -> dict:
    """Create a contact constraint between two residues/atoms.
    
    Args:
        token1_chain: Chain ID of first token.
        token1_residue: Residue index or atom name of first token.
        token2_chain: Chain ID of second token.
        token2_residue: Residue index or atom name of second token.
        max_distance: Maximum distance in Angstroms (4-20, default 6).
        force: Whether to enforce with a potential.
    
    Returns:
        Dictionary representing the contact constraint.
    """
    constraint = {
        "contact": {
            "token1": [token1_chain, token1_residue],
            "token2": [token2_chain, token2_residue],
        }
    }
    
    if max_distance != 6.0:
        constraint["contact"]["max_distance"] = max_distance
    
    if force:
        constraint["contact"]["force"] = True
    
    return constraint


def create_affinity_property(binder_chain: str) -> dict:
    """Create an affinity property specification.
    
    Args:
        binder_chain: Chain ID of the ligand for affinity calculation.
    
    Returns:
        Dictionary representing the affinity property.
    """
    return {
        "affinity": {
            "binder": binder_chain,
        }
    }


def build_yaml(
    sequences: list[dict],
    constraints: Optional[list[dict]] = None,
    templates: Optional[list[dict]] = None,
    properties: Optional[list[dict]] = None,
    version: int = 1,
) -> str:
    """Build a complete YAML input file.
    
    Args:
        sequences: List of sequence entries (protein, dna, rna, ligand).
        constraints: Optional list of constraints (bond, pocket, contact).
        templates: Optional list of template specifications.
        properties: Optional list of property specifications (e.g., affinity).
        version: YAML format version (default 1).
    
    Returns:
        YAML string ready to be saved to a file.
    """
    data = {
        "version": version,
        "sequences": sequences,
    }
    
    if constraints:
        data["constraints"] = constraints
    
    if templates:
        data["templates"] = templates
    
    if properties:
        data["properties"] = properties
    
    return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)


def validate_protein_sequence(sequence: str) -> tuple[bool, str]:
    """Validate a protein sequence.
    
    Args:
        sequence: Amino acid sequence to validate.
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    sequence = sequence.upper().strip()
    
    if not sequence:
        return False, "Sequence cannot be empty."
    
    invalid_chars = set(sequence) - valid_aa
    if invalid_chars:
        return False, f"Invalid amino acids: {', '.join(sorted(invalid_chars))}"
    
    return True, ""


def validate_dna_sequence(sequence: str) -> tuple[bool, str]:
    """Validate a DNA sequence.
    
    Args:
        sequence: Nucleotide sequence to validate.
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    valid_bases = set("ATCG")
    sequence = sequence.upper().strip()
    
    if not sequence:
        return False, "Sequence cannot be empty."
    
    invalid_chars = set(sequence) - valid_bases
    if invalid_chars:
        return False, f"Invalid nucleotides: {', '.join(sorted(invalid_chars))}"
    
    return True, ""


def validate_rna_sequence(sequence: str) -> tuple[bool, str]:
    """Validate an RNA sequence.
    
    Args:
        sequence: Nucleotide sequence to validate.
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    valid_bases = set("AUCG")
    sequence = sequence.upper().strip()
    
    if not sequence:
        return False, "Sequence cannot be empty."
    
    invalid_chars = set(sequence) - valid_bases
    if invalid_chars:
        return False, f"Invalid nucleotides: {', '.join(sorted(invalid_chars))}"
    
    return True, ""


def validate_chain_id(chain_id: str, existing_ids: set[str]) -> tuple[bool, str]:
    """Validate a chain ID.
    
    Args:
        chain_id: Chain identifier to validate.
        existing_ids: Set of already used chain IDs.
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    if not chain_id:
        return False, "Chain ID cannot be empty."
    
    if chain_id in existing_ids:
        return False, f"Chain ID '{chain_id}' is already in use."
    
    return True, ""
