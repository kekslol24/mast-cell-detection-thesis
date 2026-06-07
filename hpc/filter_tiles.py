"""
Filter tiles in a patient directory to keep only those listed in the Excel file.
Shows a preview and asks for confirmation before deleting anything.

Usage:
    python filter_tiles.py /path/to/new/P10         # single patient
    python filter_tiles.py /path/to/new/ --all      # all P* subdirs
"""

import argparse
import glob
import os
import sys

import openpyxl

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def load_keep_set(xlsx_path: str) -> set[str]:
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    keep = set()
    for row in ws.iter_rows(values_only=True):
        val = row[0]
        if val:
            keep.add(str(val).strip())
    wb.close()
    return keep


def analyze_dir(patient_dir: str) -> dict | None:
    xlsx_files = glob.glob(os.path.join(patient_dir, "*.xlsx"))
    if not xlsx_files:
        return None

    xlsx_path = xlsx_files[0]
    excel_set = load_keep_set(xlsx_path)

    tile_dir  = os.path.join(patient_dir, "images", "Train")
    label_dir = os.path.join(patient_dir, "labels", "Train")
    if not os.path.isdir(tile_dir):
        return None

    all_images = [
        f for f in os.listdir(tile_dir)
        if os.path.isfile(os.path.join(tile_dir, f))
        and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]

    # Keep if in Excel (FP negatives) OR has a matching label file (annotated)
    has_label = lambda f: os.path.isfile(
        os.path.join(label_dir, os.path.splitext(f)[0] + ".txt")
    )
    keep_set  = {f for f in all_images if f in excel_set or has_label(f)}
    to_delete = [f for f in all_images if f not in keep_set]
    missing   = excel_set - set(all_images)

    labeled_count = sum(1 for f in all_images if has_label(f))

    return {
        "dir":           patient_dir,
        "tile_dir":      tile_dir,
        "xlsx":          os.path.basename(xlsx_path),
        "excel_set":     excel_set,
        "keep_set":      keep_set,
        "all_images":    all_images,
        "labeled_count": labeled_count,
        "to_delete":     to_delete,
        "missing":       missing,
    }


def print_summary(info: dict, name: str = None):
    label = name or os.path.basename(info["dir"])
    print(f"\n  {label}/")
    print(f"    Excel ({info['xlsx']}):  {len(info['excel_set'])} FP-negative tiles")
    print(f"    Labeled tiles:         {info['labeled_count']} (have .txt in labels/Train/)")
    print(f"    Images on disk:        {len(info['all_images'])}")
    print(f"    Would keep:            {len(info['keep_set'])}  (Excel + labeled)")
    print(f"    Would delete:          {len(info['to_delete'])}")
    if info["missing"]:
        print(f"    Excel not on disk:     {len(info['missing'])}  (already gone or name mismatch)")


def do_delete(info: dict):
    for f in info["to_delete"]:
        os.remove(os.path.join(info["tile_dir"], f))
    print(f"  Deleted {len(info['to_delete'])} files from {os.path.basename(info['dir'])}/")


def ask(prompt: str) -> bool:
    while True:
        ans = input(prompt).strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no", ""):
            return False
        print("  Please enter y or n.")


def main():
    parser = argparse.ArgumentParser(description="Filter patient tile dirs by Excel allowlist.")
    parser.add_argument("path", help="Patient dir (e.g. /data/new/P10) or parent dir with --all")
    parser.add_argument("--all", action="store_true", help="Process all P* subdirs under <path>")
    args = parser.parse_args()

    if args.all:
        subdirs = sorted(d for d in glob.glob(os.path.join(args.path, "P*")) if os.path.isdir(d))
        if not subdirs:
            print(f"No P* subdirectories found under {args.path}")
            sys.exit(1)

        results = []
        for d in subdirs:
            info = analyze_dir(d)
            if info:
                results.append(info)
            else:
                print(f"  [SKIP] {os.path.basename(d)}/ — no Excel or images/Train/ missing")

        if not results:
            print("Nothing to process.")
            sys.exit(0)

        print("\n=== Preview ===")
        total_keep = total_delete = 0
        for info in results:
            print_summary(info)
            total_keep   += len(info["keep_set"])
            total_delete += len(info["to_delete"])

        print(f"\n  TOTAL  keep: {total_keep}  |  delete: {total_delete}")

        if total_delete == 0:
            print("\nNothing to delete.")
            sys.exit(0)

        print()
        if not ask("Proceed and delete all listed files? [y/N] "):
            print("Aborted.")
            sys.exit(0)

        for info in results:
            if info["to_delete"]:
                do_delete(info)
        print("\nDone.")

    else:
        if not os.path.isdir(args.path):
            print(f"Not a directory: {args.path}")
            sys.exit(1)

        info = analyze_dir(args.path)
        if info is None:
            print(f"No .xlsx found or images/Train/ missing in {args.path}")
            sys.exit(1)

        print("\n=== Preview ===")
        print_summary(info)

        if not info["to_delete"]:
            print("\nNothing to delete.")
            sys.exit(0)

        print()
        if not ask("Proceed and delete listed files? [y/N] "):
            print("Aborted.")
            sys.exit(0)

        do_delete(info)
        print("Done.")


if __name__ == "__main__":
    main()
