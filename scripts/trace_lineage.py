#!/usr/bin/env python3
"""
SpaceHASTEN Lineage Tracer

Traces the "family tree" of a hit compound back to its original seed
by walking through the similarity search cycles.

Supports both the new SpaceHASTEN workspace layout and the legacy layout.

Usage:
    python3 trace_lineage.py <database.dbsh> [--smiles SMILES | --id SPACEHASTENID | --smilesid SMILESID]

Example:
    python3 trace_lineage.py myproject.dbsh --smiles "CCO"
    python3 trace_lineage.py myproject.dbsh --id 12345
    python3 trace_lineage.py myproject.dbsh --smilesid "Z1234567890"
    python3 trace_lineage.py myproject.dbsh --id 12345 --layout legacy
"""

import argparse
import csv
import glob
import json
import os
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
from rdkit import Chem


def canonicalize(smiles):
    """Return canonical SMILES, or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def find_compound(dbname, smiles=None, spacehastenid=None, smilesid=None):
    """Look up a compound in the database and return its record."""
    conn = sqlite3.connect(dbname)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    if spacehastenid is not None:
        row = c.execute(
            "SELECT spacehastenid, smiles, smilesid, simsearch_cycle, query, dock_score, pred_score "
            "FROM data WHERE spacehastenid = ?",
            [spacehastenid],
        ).fetchone()
    elif smilesid is not None:
        row = c.execute(
            "SELECT spacehastenid, smiles, smilesid, simsearch_cycle, query, dock_score, pred_score "
            "FROM data WHERE smilesid = ?",
            [smilesid],
        ).fetchone()
    elif smiles is not None:
        # Try exact match first, then canonical match
        row = c.execute(
            "SELECT spacehastenid, smiles, smilesid, simsearch_cycle, query, dock_score, pred_score "
            "FROM data WHERE smiles = ?",
            [smiles],
        ).fetchone()
        if row is None:
            can_smi = canonicalize(smiles)
            if can_smi:
                rows = c.execute(
                    "SELECT spacehastenid, smiles, smilesid, simsearch_cycle, query, dock_score, pred_score "
                    "FROM data"
                ).fetchall()
                for r in rows:
                    if canonicalize(r["smiles"]) == can_smi:
                        row = r
                        break
    else:
        conn.close()
        return None

    conn.close()
    if row is None:
        return None
    return dict(row)


def _get_cycle_dir(layout, spacehasten_dir, project_name, cycle_number):
    """Return the cycle directory path based on layout mode."""
    if layout == "legacy":
        return os.path.join(
            spacehasten_dir, f"SIMSEARCH_{project_name}_cycle{cycle_number}"
        )
    else:
        return os.path.join(spacehasten_dir, "simsearch", f"cycle{cycle_number}")


def _get_result_glob_pattern(layout, cycle_dir, method, project_name):
    """Return the glob pattern for result CSV files based on layout mode."""
    if layout == "legacy":
        # Legacy: {cycle_dir}/{method}result_{name}_{taskID}_1.csv
        return os.path.join(cycle_dir, f"{method}result_{project_name}_*_1.csv")
    else:
        # New: {cycle_dir}/results/{method}result_{taskID}.csv
        return os.path.join(cycle_dir, "results", f"{method}result_*.csv")


def _extract_task_id(layout, basename, method, project_name):
    """Extract the task ID from a result filename based on layout mode.

    Returns the integer task ID, or None if parsing fails.
    """
    if layout == "legacy":
        # Format: {method}result_{name}_{taskID}_1.csv
        prefix = f"{method}result_{project_name}_"
        suffix = "_1.csv"
        if not basename.startswith(prefix) or not basename.endswith(suffix):
            return None
        task_id_str = basename[len(prefix):-len(suffix)]
    else:
        # Format: {method}result_{taskID}.csv
        # Also handle paginated: {method}result_{taskID}_{page}.csv
        prefix = f"{method}result_"
        if not basename.startswith(prefix):
            return None
        rest = basename[len(prefix):]
        # Remove .csv extension
        if rest.endswith(".csv"):
            rest = rest[:-4]
        # If there's an underscore, take the first part as task_id
        parts = rest.split("_")
        task_id_str = parts[0]

    try:
        return int(task_id_str)
    except ValueError:
        return None


def _scan_single_file(resfile, method, layout, project_name, target_canonical, sim_field):
    """Scan a single result CSV for a matching compound (worker function).

    Returns (task_id, similarity) or None if no match found.
    """
    basename = os.path.basename(resfile)
    task_id = _extract_task_id(layout, basename, method, project_name)
    if task_id is None:
        return None

    try:
        resdata = pd.read_csv(resfile)
    except Exception:
        return None

    if "#result-smiles" not in resdata.columns:
        return None

    best_sim = -1.0
    found = False
    for _, row in resdata.iterrows():
        result_smiles = str(row["#result-smiles"])
        result_canonical = canonicalize(result_smiles)
        if result_canonical == target_canonical:
            similarity = float(row.get(sim_field, 0.0))
            if similarity > best_sim:
                best_sim = similarity
                found = True

    if found:
        return (task_id, method, best_sim)
    return None


def find_query_for_compound(compound_smiles, cycle_number, project_name, spacehasten_dir, layout="new", workers=1):
    """
    Search result CSV files in the cycle directory to find which query
    produced a given compound.

    Returns (query_line_number, query_smiles, query_spacehastenid, method, similarity)
    or None if not found.
    """
    cycle_dir = _get_cycle_dir(layout, spacehasten_dir, project_name, cycle_number)

    if not os.path.isdir(cycle_dir):
        print(f"  WARNING: Cycle directory not found: {cycle_dir}")
        return None

    queries_file = os.path.join(cycle_dir, f"queries_{project_name}.smi")
    if not os.path.isfile(queries_file):
        print(f"  WARNING: Queries file not found: {queries_file}")
        return None

    # Read queries file (1-indexed lines correspond to SLURM array task IDs)
    with open(queries_file, "rt") as f:
        query_lines = f.readlines()

    # Canonicalize target for matching
    target_canonical = canonicalize(compound_smiles)
    if target_canonical is None:
        print(f"  WARNING: Could not canonicalize target SMILES: {compound_smiles}")
        return None

    # Collect all (file, method, sim_field) jobs
    jobs = []
    for method in ["spacelight", "ftrees"]:
        pattern = _get_result_glob_pattern(layout, cycle_dir, method, project_name)
        result_files = glob.glob(pattern)
        sim_field = "fingerprint-similarity" if method == "spacelight" else "pharmacophore-similarity"
        for resfile in result_files:
            jobs.append((resfile, method, sim_field))

    if not jobs:
        return None

    # Scan files in parallel
    best_match = None
    best_similarity = -1.0

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _scan_single_file, resfile, method, layout, project_name, target_canonical, sim_field
                ): (resfile, method)
                for resfile, method, sim_field in jobs
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    task_id, method, similarity = result
                    if similarity > best_similarity:
                        best_similarity = similarity
                        if task_id <= len(query_lines):
                            qline = query_lines[task_id - 1].strip()
                            parts = qline.split()
                            q_smiles = parts[0] if len(parts) >= 1 else ""
                            q_id = int(parts[1]) if len(parts) >= 2 else None
                            best_match = (task_id, q_smiles, q_id, method, similarity)
    else:
        for resfile, method, sim_field in jobs:
            result = _scan_single_file(resfile, method, layout, project_name, target_canonical, sim_field)
            if result is not None:
                task_id, method, similarity = result
                if similarity > best_similarity:
                    best_similarity = similarity
                    if task_id <= len(query_lines):
                        qline = query_lines[task_id - 1].strip()
                        parts = qline.split()
                        q_smiles = parts[0] if len(parts) >= 1 else ""
                        q_id = int(parts[1]) if len(parts) >= 2 else None
                        best_match = (task_id, q_smiles, q_id, method, similarity)

    return best_match


def trace_lineage(dbname, compound, spacehasten_dir=None, layout="new", workers=1):
    """
    Trace the full lineage of a compound back to its seed.

    Returns a list of dicts representing each step in the chain,
    from the hit back to the original seed.
    """
    # Determine project name from dbname
    project_name = os.path.basename(dbname).replace(".dbsh", "")

    # Default SPACEHASTEN directory
    if spacehasten_dir is None:
        spacehasten_dir = os.path.join(os.getenv("HOME"), "SPACEHASTEN")

    chain = [compound]
    current = compound

    visited = set()

    while True:
        shid = current["spacehastenid"]
        if shid in visited:
            print("  WARNING: Cycle detected in lineage! Stopping.")
            break
        visited.add(shid)

        cycle = current.get("simsearch_cycle")

        if cycle is None or cycle == 0:
            # This is a seed compound
            current["role"] = "SEED"
            break

        # Find which query produced this compound
        result = find_query_for_compound(
            current["smiles"], cycle, project_name, spacehasten_dir, layout, workers
        )

        if result is None:
            print(
                f"  WARNING: Could not find query for compound {shid} "
                f"(cycle {cycle}). Trail ends here."
            )
            break

        task_id, q_smiles, q_id, method, similarity = result
        current["found_by_method"] = method
        current["similarity_to_parent"] = similarity

        # Look up the parent query in the database
        parent = find_compound(dbname, spacehastenid=q_id)
        if parent is None:
            print(
                f"  WARNING: Query compound (spacehastenid={q_id}) "
                f"not found in database. Trail ends here."
            )
            # Still record what we know
            chain.append(
                {
                    "spacehastenid": q_id,
                    "smiles": q_smiles,
                    "smilesid": "?",
                    "simsearch_cycle": None,
                    "role": "SEED (not in DB)",
                }
            )
            break

        chain.append(parent)
        current = parent

    # Mark roles
    if len(chain) > 0 and "role" not in chain[0]:
        chain[0]["role"] = "HIT"
    for i in range(1, len(chain) - 1):
        if "role" not in chain[i]:
            chain[i]["role"] = "QUERY"

    return chain


def print_lineage(chain):
    """Pretty-print the lineage chain."""
    print("\n" + "=" * 70)
    print("COMPOUND LINEAGE TRACE")
    print("=" * 70)

    for i, step in enumerate(chain):
        indent = "  " * i
        role = step.get("role", "?")
        shid = step.get("spacehastenid", "?")
        smiles = step.get("smiles", "?")
        smilesid = step.get("smilesid", "?")
        cycle = step.get("simsearch_cycle")
        dock = step.get("dock_score")
        pred = step.get("pred_score")

        print(f"\n{indent}[{role}] spacehastenid={shid}")
        print(f"{indent}  SMILES: {smiles}")
        print(f"{indent}  Title:  {smilesid}")
        if cycle is not None:
            print(f"{indent}  Found in cycle: {cycle}")
        if dock is not None:
            print(f"{indent}  Dock score:  {dock}")
        if pred is not None:
            print(f"{indent}  Pred score:  {pred}")

        # Print link info
        if "found_by_method" in step:
            method = step["found_by_method"]
            sim = step.get("similarity_to_parent", "?")
            print(f"{indent}  Found via: {method} (similarity={sim:.3f})")

        if i < len(chain) - 1:
            print(f"{indent}  {'|'}")
            print(f"{indent}  {'v'}")

    print("\n" + "=" * 70)
    print(f"Chain length: {len(chain)} compounds "
          f"({len(chain) - 1} steps from hit to seed)")
    print("=" * 70 + "\n")


def write_lineage_json(chain, dbname, outpath):
    """Write a lineage chain to a JSON file."""
    output = {
        "metadata": {
            "database": os.path.abspath(dbname),
            "timestamp": datetime.now().isoformat(),
            "chain_length": len(chain),
            "hit_spacehastenid": chain[0]["spacehastenid"] if chain else None,
            "seed_spacehastenid": chain[-1]["spacehastenid"] if chain else None,
        },
        "lineage": chain,
    }
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)


def write_lineage_csv(chain, dbname, outpath):
    """Write a lineage chain to a CSV file.

    Each row represents one compound in the chain. Columns include all
    fields present in any compound record plus metadata columns.
    """
    if not chain:
        return

    # Collect all possible keys across all compounds in the chain
    all_keys = set()
    for step in chain:
        all_keys.update(step.keys())

    # Define a sensible column order: metadata first, then compound fields
    metadata_cols = ["step", "database", "timestamp", "chain_length",
                     "hit_spacehastenid", "seed_spacehastenid"]
    # Preferred order for compound fields
    preferred_order = [
        "role", "spacehastenid", "smiles", "smilesid",
        "simsearch_cycle", "query", "dock_score", "pred_score",
        "found_by_method", "similarity_to_parent",
    ]
    # Any remaining keys not in preferred_order
    remaining = sorted(all_keys - set(preferred_order))
    compound_cols = [k for k in preferred_order if k in all_keys] + remaining

    fieldnames = metadata_cols + compound_cols

    # Build metadata values
    timestamp = datetime.now().isoformat()
    db_abs = os.path.abspath(dbname)
    chain_length = len(chain)
    hit_id = chain[0]["spacehastenid"] if chain else None
    seed_id = chain[-1]["spacehastenid"] if chain else None

    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for i, step in enumerate(chain):
            row = dict(step)
            row["step"] = i
            row["database"] = db_abs
            row["timestamp"] = timestamp
            row["chain_length"] = chain_length
            row["hit_spacehastenid"] = hit_id
            row["seed_spacehastenid"] = seed_id
            writer.writerow(row)


def _write_output(chain, dbname, outpath, fmt):
    """Write lineage output in the specified format(s).

    fmt can be 'json', 'csv', or 'both'.
    """
    base, ext = os.path.splitext(outpath)

    if fmt == "json":
        json_path = outpath if ext == ".json" else base + ".json"
        write_lineage_json(chain, dbname, json_path)
        print(f"Lineage written to {json_path}")
    elif fmt == "csv":
        csv_path = outpath if ext == ".csv" else base + ".csv"
        write_lineage_csv(chain, dbname, csv_path)
        print(f"Lineage written to {csv_path}")
    elif fmt == "both":
        json_path = base + ".json" if ext not in (".json", ".csv") else base + ".json"
        csv_path = base + ".csv"
        write_lineage_json(chain, dbname, json_path)
        write_lineage_csv(chain, dbname, csv_path)
        print(f"Lineage written to {json_path} and {csv_path}")


def run_single(args):
    """Trace lineage for a single compound."""
    print(f"Looking up compound in {args.database}...")
    compound = find_compound(
        args.database,
        smiles=args.smiles,
        spacehastenid=args.id,
        smilesid=args.smilesid,
    )

    if compound is None:
        sys.exit("Error: Compound not found in database!")

    print(f"Found: spacehastenid={compound['spacehastenid']}, "
          f"smiles={compound['smiles']}")

    print("\nTracing lineage...")
    chain = trace_lineage(args.database, compound, args.spacehasten_dir, args.layout, args.workers)

    print_lineage(chain)

    if args.output:
        _write_output(chain, args.database, args.output, args.format)


def run_batch(args):
    """Trace lineage for every smilesid listed in a file."""
    if not os.path.exists(args.batch):
        sys.exit(f"Error: Batch file not found: {args.batch}")

    with open(args.batch, "rt") as f:
        smilesids = [line.strip() for line in f if line.strip()]

    if not smilesids:
        sys.exit("Error: Batch file is empty!")

    outdir = args.output if args.output else "."
    os.makedirs(outdir, exist_ok=True)

    print(f"Batch mode: {len(smilesids)} compounds to trace")
    print(f"Output directory: {os.path.abspath(outdir)}\n")

    found = 0
    not_found = 0
    for i, sid in enumerate(smilesids, 1):
        print(f"[{i}/{len(smilesids)}] {sid}")
        compound = find_compound(args.database, smilesid=sid)
        if compound is None:
            print(f"  NOT FOUND in database, skipping.\n")
            not_found += 1
            continue

        chain = trace_lineage(args.database, compound, args.spacehasten_dir, args.layout, args.workers)
        print_lineage(chain)

        safe_name = sid.replace("/", "_").replace(" ", "_")
        outpath = os.path.join(outdir, f"lineage_{safe_name}")
        _write_output(chain, args.database, outpath, args.format)
        print()
        found += 1

    print("=" * 70)
    print(f"Batch complete: {found} traced, {not_found} not found")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Trace the lineage of a SpaceHASTEN hit compound back to its seed."
    )
    parser.add_argument("database", help="Path to the .dbsh database file")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smiles", help="SMILES of the hit compound")
    group.add_argument("--id", type=int, help="spacehastenid of the hit compound")
    group.add_argument("--smilesid", help="smilesid (title) of the hit compound")
    group.add_argument(
        "--batch",
        help="File with one smilesid per line (saves one JSON per compound)",
    )

    parser.add_argument(
        "--spacehasten-dir",
        default=None,
        help="Path to the SpaceHASTEN shared workspace directory "
             "(default: ~/SPACEHASTEN)",
    )
    parser.add_argument(
        "--layout",
        choices=["new", "legacy"],
        default="new",
        help="Workspace layout version. 'new' uses <dir>/simsearch/cycle<N>/results/; "
             "'legacy' uses <dir>/SIMSEARCH_<name>_cycle<N>/ (default: new)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Write lineage to file (default: print to terminal only). "
             "For --batch mode, this is the output directory (default: current dir).",
    )
    parser.add_argument(
        "-f", "--format",
        choices=["json", "csv", "both"],
        default="json",
        help="Output format: 'json' (default), 'csv', or 'both'.",
    )
    parser.add_argument(
        "-j", "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for scanning result files. "
             "Default: all available CPU cores. Use 1 to disable parallelism.",
    )

    args = parser.parse_args()

    if args.workers is None:
        args.workers = os.cpu_count() or 4

    if not os.path.exists(args.database):
        sys.exit(f"Error: Database file not found: {args.database}")

    if args.batch:
        run_batch(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
