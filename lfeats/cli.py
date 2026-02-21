#!/usr/bin/env python3

# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A script to extract latent features from audio using pretrained models."""

import argparse
import logging
import os

logger = logging.getLogger("lfeats")


def get_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the feature extraction script."""
    parser = argparse.ArgumentParser(
        description="Extract latent features from audio using pretrained models."
    )
    parser.add_argument(
        "source",
        type=str,
        help="The source audio file, directory, or .scp file containing file paths.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The directory where the extracted features will be saved.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="npz",
        choices=["npz", "pt", "float", "double"],
        help="The format to save the extracted features.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="hubert",
        help="The model to use for feature extraction.",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default=None,
        help="The variant of the model to use.",
    )
    parser.add_argument(
        "--resampler_type",
        type=str,
        default="torchaudio",
        help="The resampling library to use.",
    )
    parser.add_argument(
        "--resampler_preset",
        type=str,
        default=None,
        help="The resampling preset to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="The device to run the model on.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory to use for caching model files.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="last",
        help=(
            "The layer(s) from which to extract features. Can be 'all', 'last', "
            "a comma-separated list of layer indices, or a single layer index."
        ),
    )
    parser.add_argument(
        "--center",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable center padding for input audio.",
    )
    parser.add_argument(
        "--chunk_length_sec",
        type=int,
        default=30,
        help="The length of audio chunks in seconds for processing.",
    )
    parser.add_argument(
        "--overlap_length_sec",
        type=int,
        default=5,
        help="The length of overlap between audio chunks in seconds.",
    )
    parser.add_argument(
        "--upsample_factor",
        type=int,
        default=1,
        help="The upsampling factor for the extracted features.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="If set, suppresses non-error output during processing.",
    )
    return parser.parse_args()


def main() -> None:
    """Perform the main feature extraction process."""
    args = get_arguments()

    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(message)s",
    )

    import numpy as np
    import torch

    import lfeats
    from lfeats.utils.io import load_audio

    # Get the list of input files from the source argument.
    if os.path.isfile(args.source):
        if args.source.endswith(".scp"):
            with open(args.source) as f:
                input_files = [line.strip() for line in f if line.strip()]
        else:
            input_files = [args.source]
    elif os.path.isdir(args.source):
        input_files = []
        for root, _, files in os.walk(args.source):
            for file in files:
                input_files.append(os.path.join(root, file))
    else:
        raise ValueError(f"Invalid source: {args.source}")

    if len(input_files) == 0:
        raise ValueError(f"No audio files found in the source: {args.source}")
    logger.info(f"Found {len(input_files)} audio files to process.")

    # Parse the layers argument.
    if args.layers in ("all", "last"):
        layers = args.layers
    elif "," in args.layers:
        layers = [int(layer.strip()) for layer in args.layers.split(",")]
    elif args.layers.isdigit():
        layers = int(args.layers)
    else:
        raise ValueError(f"Invalid layers argument: {args.layers}")

    # Prepare the output directory if specified.
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    output_ext = {
        "npz": "npz",
        "pt": "pt",
        "float": "feats",
        "double": "feats",
    }[args.output_format]

    # Initialize the feature extractor.
    extractor = lfeats.Extractor(
        model_name=args.model_name,
        model_variant=args.model_variant,
        resampler_type=args.resampler_type,
        resampler_preset=args.resampler_preset,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    extractor.load()

    # Process each input file and extract features.
    for input_file in input_files:
        if not os.path.isfile(input_file):
            logger.error(f"Could not find file: {input_file}. Skipping.")
            continue

        base, _ = os.path.splitext(os.path.basename(input_file))
        output_file = f"{base}.{output_ext}"
        if args.output_dir is not None:
            output_file = os.path.join(args.output_dir, output_file)

        try:
            audio, sample_rate = load_audio(input_file)
            if audio.ndim > 1:
                audio = audio.mean(dim=0)  # Convert to mono by averaging channels.

            features = extractor(
                source=audio,
                sample_rate=sample_rate,
                layers=layers,
                center=args.center,
                chunk_length_sec=args.chunk_length_sec,
                overlap_length_sec=args.overlap_length_sec,
                upsample_factor=args.upsample_factor,
            )
        except Exception as e:
            logger.error(f"Error processing file {input_file}: {e}. Skipping.")
            continue

        if args.output_format == "npz":
            result = {
                "features": features.array,
                "source": features.source,
                "layers": features.layers,
            }
            np.savez_compressed(output_file, **result)
        elif args.output_format == "pt":
            result = {
                "features": features.tensor.cpu(),
                "source": features.source,
                "layers": features.layers,
            }
            torch.save(result, output_file)
        elif args.output_format == "float":
            features.array.tofile(output_file)
        elif args.output_format == "double":
            features.array.astype(np.float64).tofile(output_file)
        else:
            raise ValueError(f"Unsupported output format: {args.output_format}")

    logger.info("All files processed successfully.")


if __name__ == "__main__":
    main()
