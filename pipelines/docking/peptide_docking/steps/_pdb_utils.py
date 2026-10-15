"""Shared PDB helper for the robustness pipeline steps."""


def filter_chain(src_pdb: str, dst_pdb: str, keep_chain: str) -> None:
    """Keep only ATOM records for `keep_chain` (plus TER/END).

    LightDock's peptide-rebuild step occasionally misdetects a fragment of
    a large side chain (e.g. a fatty-acid conjugate) as its own residue and
    writes it as a spurious second PDB chain (typically 'B'). HADDOCK3's
    CNS backend then sees a topology mismatch against run.toml's 2-molecule
    `molecules=[...]` list and crashes with "Symbol not found:
    $mol_fix_origin_$maxid" (data.ncomponents=2 vs. 3 real segments in the
    PSF). PyRosetta silently mis-numbers residues on the same phantom chain
    if it isn't filtered out first. Filtering to the one real peptide chain
    before either tool sees the file fixes both -- confirmed against 6/6
    real HADDOCK3 failures found in this project.
    """
    filter_chains(src_pdb, dst_pdb, {keep_chain})


def filter_chains(src_pdb: str, dst_pdb: str, keep_chains) -> None:
    """Same as filter_chain(), keeping any chain letter in `keep_chains`."""
    keep_chains = set(keep_chains)
    with open(src_pdb) as fin, open(dst_pdb, "w") as fout:
        for line in fin:
            if line.startswith("ATOM") and line[21] not in keep_chains:
                continue
            fout.write(line)


def chain_letters(pdb_path: str) -> set:
    """Return the set of distinct ATOM chain letters actually present.

    Used to tell the diagnosed "phantom residue" bug (exactly one small
    spurious extra chain, see filter_chain()) apart from a peptide whose
    build_pdb() output has more than one REAL chain on purpose (a lariat/
    bicyclic topology, see PeptideProcessor.order_chain -- chains > 1 there
    is a legitimate multi-segment peptide, not an artifact). Filtering the
    latter down to chain A alone would silently discard real peptide atoms
    instead of the intended few-atom artifact, so callers should treat more
    than 2 chains as a case this extraction method can't handle yet, not
    apply the same blanket fix.
    """
    chains = set()
    with open(pdb_path, errors="ignore") as fh:
        for line in fh:
            if line.startswith("ATOM"):
                chains.add(line[21])
    return chains
