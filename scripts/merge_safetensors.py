#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tool: merge_safetensors.py
Description:
    Fill missing weights of a trained safetensors checkpoint using a full original checkpoint.
    The trained checkpoint overrides matching keys, while all missing keys are restored
    from the original full checkpoint.

Usage:
    python merge_safetensors.py \
        --origin /path/to/origin.safetensors \
        --trained /path/to/trained.safetensors \
        --output /path/to/merged.safetensors
"""

import argparse
from safetensors.torch import load_file, save_file


def merge_checkpoints(origin_path: str, trained_path: str, output_path: str):
    origin = load_file(origin_path)
    trained = load_file(trained_path)

    merged = dict(origin)
    merged.update(trained)

    save_file(merged, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Fill missing weights of a trained checkpoint using a full original checkpoint."
    )
    parser.add_argument("--origin", required=True, help="Path to the original full safetensors file.")
    parser.add_argument("--trained", required=True, help="Path to the trained safetensors file.")
    parser.add_argument("--output", required=True, help="Path to save the merged safetensors file.")

    args = parser.parse_args()
    merge_checkpoints(args.origin, args.trained, args.output)


if __name__ == "__main__":
    main()
