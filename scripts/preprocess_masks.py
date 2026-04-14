#!/usr/bin/env python3
"""
Preprocess MRI data with breast segmentation using BreastDivider.

For each sample:
1. Run BreastDivider on Pre.nii.gz to get a breast mask
2. Apply the mask to Sub_1.nii.gz (zero out non-breast regions)
3. Save as Sub_1_masked.nii.gz in a mirrored directory structure

This is the key preprocessing step from the MeisenMeister paper (ODELIA challenge winner):
"cropping away non-informative background regions led to significant performance gains"

Usage:
    python scripts/preprocess_masks.py \
        --data_root /cluster/projects/vc/courses/TDT17/mic/ODELIA2025/data/ \
        --output_root /cluster/work/konradj/breast_mri/data_masked/
"""

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def apply_mask_and_save(sub1_path: str, mask_path: str, output_path: str):
    """Load Sub_1, apply breast mask, save masked version."""
    sub1_img = nib.load(sub1_path)
    mask_img = nib.load(mask_path)

    sub1_data = sub1_img.get_fdata()
    mask_data = mask_img.get_fdata()

    # BreastDivider outputs: 0=background, 1=left breast, 2=right breast
    # For unilateral data, we want any breast tissue (mask > 0)
    breast_mask = (mask_data > 0).astype(np.float32)

    # Apply mask: zero out non-breast regions
    masked_data = sub1_data * breast_mask

    # Save with same affine and header
    masked_img = nib.Nifti1Image(masked_data.astype(np.float32), sub1_img.affine, sub1_img.header)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(masked_img, output_path)


def main():
    parser = argparse.ArgumentParser(description="Preprocess with breast segmentation")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    temp_mask_dir = output_root / "_temp_masks"
    temp_mask_dir.mkdir(parents=True, exist_ok=True)

    # Import BreastDivider
    try:
        from breastdivider import BreastDividerPredictor
        predictor = BreastDividerPredictor(device=args.device)
        print("BreastDivider loaded successfully")
    except ImportError:
        print("ERROR: breastdivider not installed. Run: pip install breastdivider")
        return

    institutions = ["CAM", "MHA", "RSH", "RUMC", "UKA"]
    total_processed = 0
    total_skipped = 0

    for inst in institutions:
        inst_dir = data_root / inst / "data_unilateral"
        if not inst_dir.exists():
            print(f"Skipping {inst}: directory not found")
            continue

        samples = sorted([d for d in inst_dir.iterdir() if d.is_dir()])
        print(f"\n{inst}: {len(samples)} samples")

        for sample_dir in tqdm(samples, desc=f"Processing {inst}"):
            uid = sample_dir.name
            pre_path = sample_dir / "Pre.nii.gz"
            sub1_path = sample_dir / "Sub_1.nii.gz"

            if not pre_path.exists() or not sub1_path.exists():
                total_skipped += 1
                continue

            # Output paths
            output_dir = output_root / inst / "data_unilateral" / uid
            output_sub1 = output_dir / "Sub_1_masked.nii.gz"

            # Skip if already processed
            if output_sub1.exists():
                total_processed += 1
                continue

            try:
                # Step 1: Run BreastDivider on Pre to get breast mask
                mask_path = str(temp_mask_dir / f"{uid}_mask.nii.gz")
                predictor.predict(input_path=str(pre_path), output_path=mask_path)

                # Step 2: Apply mask to Sub_1 and save
                output_dir.mkdir(parents=True, exist_ok=True)
                apply_mask_and_save(str(sub1_path), mask_path, str(output_sub1))

                # Also copy/link other sequences that might be needed
                # (Pre, T2 etc. — copy as symlinks to save space)
                for seq_name in ["Pre", "Post_1", "Post_2", "Sub_1", "T2"]:
                    seq_src = sample_dir / f"{seq_name}.nii.gz"
                    seq_dst = output_dir / f"{seq_name}.nii.gz"
                    if seq_src.exists() and not seq_dst.exists():
                        os.symlink(str(seq_src), str(seq_dst))

                total_processed += 1

            except Exception as e:
                print(f"Error processing {uid}: {e}")
                total_skipped += 1

    # Also copy metadata
    for inst in institutions:
        meta_src = data_root / inst / "metadata_unilateral"
        meta_dst = output_root / inst / "metadata_unilateral"
        if meta_src.exists() and not meta_dst.exists():
            os.symlink(str(meta_src), str(meta_dst))

    print(f"\nDone! Processed: {total_processed}, Skipped: {total_skipped}")
    print(f"Masked data saved to: {output_root}")


if __name__ == "__main__":
    main()
