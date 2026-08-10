#!/usr/bin/env python3
"""
Convert SpaceHASTEN lineage JSON file(s) to CSV.

Reads one or more lineage JSON files produced by trace_lineage.py and writes
a corresponding CSV file for each, preserving all data fields as columns.

Usage:
    python3 convert_lineage_json_to_csv.py lineage_file.json
    python3 convert_lineage_json_to_csv.py lineage_*.json --outdir results/
    python3 convert_lineage_json_to_csv.py lineage_*.json --merge merged.csv
"""

import argparse
import csv
import json
import os
import sys


def json_to_rows(json_path):
    """Load a lineage JSON file and return (metadata_dict, list_of_row_dicts)."""
    with open(json_path, "r") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    chain = data.get("lineage", [])

    rows = []
    for i, step in enumerate(chain):
        row = dict(step)
        row["step"] = i
        row["database"] = metadata.get("database", "")
        row["timestamp"] = metadata.get("timestamp", "")
        row["chain_length"] = metadata.get("chain_length", len(chain))
        row["hit_spacehastenid"] = metadata.get("hit_spacehastenid")
        row["seed_spacehastenid"] = metadata.get("seed_spacehastenid")
        rows.append(row)

    return rows


def collect_fieldnames(all_rows):
    """Determine column order from all rows."""
    all_keys = set()
    for row in all_rows:
        all_keys.update(row.keys())

    # Preferred column order
    metadata_cols = ["step", "database", "timestamp", "chain_length",
                     "hit_spacehastenid", "seed_spacehastenid"]
    preferred_order = [
        "role", "spacehastenid", "smiles", "smilesid",
        "simsearch_cycle", "query", "dock_score", "pred_score",
        "found_by_method", "similarity_to_parent",
    ]
    remaining = sorted(all_keys - set(metadata_cols) - set(preferred_order))

    fieldnames = (
        [c for c in metadata_cols if c in all_keys]
        + [c for c in preferred_order if c in all_keys]
        + remaining
    )
    return fieldnames


def write_csv(rows, fieldnames, outpath):
    """Write rows to a CSV file."""
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Convert SpaceHASTEN lineage JSON files to CSV."
    )
    parser.add_argument(
        "input", nargs="+",
        help="One or more lineage JSON files to convert.",
    )
    parser.add_argument(
        "--outdir", default=None,
        help="Output directory for CSV files (default: same directory as input).",
    )
    parser.add_argument(
        "--merge", default=None,
        help="Merge all inputs into a single CSV file at the given path.",
    )

    args = parser.parse_args()

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    all_rows = []
    per_file_rows = []

    for json_path in args.input:
        if not os.path.exists(json_path):
            print(f"WARNING: File not found, skipping: {json_path}", file=sys.stderr)
            continue
        rows = json_to_rows(json_path)
        per_file_rows.append((json_path, rows))
        all_rows.extend(rows)

    if not all_rows:
        sys.exit("Error: No data found in input files.")

    if args.merge:
        # Merge all into one CSV
        fieldnames = collect_fieldnames(all_rows)
        # Add a source_file column to distinguish origins
        fieldnames = ["source_file"] + fieldnames
        for json_path, rows in per_file_rows:
            for row in rows:
                row["source_file"] = os.path.basename(json_path)
        merged_rows = []
        for _, rows in per_file_rows:
            merged_rows.extend(rows)
        write_csv(merged_rows, fieldnames, args.merge)
        print(f"Merged {len(per_file_rows)} files ({len(merged_rows)} rows) -> {args.merge}")
    else:
        # One CSV per input JSON
        for json_path, rows in per_file_rows:
            fieldnames = collect_fieldnames(rows)
            base = os.path.splitext(os.path.basename(json_path))[0]
            if args.outdir:
                csv_path = os.path.join(args.outdir, base + ".csv")
            else:
                csv_path = os.path.join(os.path.dirname(json_path), base + ".csv")
            write_csv(rows, fieldnames, csv_path)
            print(f"{json_path} -> {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
