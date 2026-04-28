"""Pure RDKit helpers for hashing/dedup and CXSMILES parsing.

All public functions are pure (no I/O, no global state) and return ``None``
on parse failure rather than raising.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import RegistrationHash


def tautomer_hash(smiles: str) -> str | None:
    """Return the tautomer-insensitive registration hash of ``smiles``.

    Returns ``None`` if the SMILES cannot be parsed or is empty.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    layers = RegistrationHash.GetMolLayers(mol)
    return layers[RegistrationHash.HashLayer.TAUTOMER_HASH]


def canonical_smiles(smiles: str) -> str | None:
    """Return the canonical SMILES for ``smiles`` or ``None`` on parse failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol)


def parse_cxsmiles(line: str) -> tuple[str, str] | None:
    """Parse a single ``"<cxsmiles>\\t<id>"`` (or whitespace-separated) line.

    Returns ``(canonical_smiles, mol_id)`` or ``None`` if the line cannot be
    parsed (missing id, unparseable SMILES, etc.).
    """
    stripped = line.rstrip("\n").rstrip("\r")
    if not stripped:
        return None
    parts = stripped.split(None, 1)
    if len(parts) < 2:
        return None
    cxsmi, mol_id = parts[0], parts[1].strip()
    if not mol_id:
        return None
    canon = canonical_smiles(cxsmi)
    if canon is None:
        return None
    return canon, mol_id
