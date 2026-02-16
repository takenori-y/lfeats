# lfeats

*lfeats* provides a unified interface to extract hidden representations from various speech foundation models such as HuBERT and Whisper.

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Requirements

- Python 3.10+
- PyTorch 2.4.0+

## Documentation

- [Reference Manual](https://takenori-y.github.io/lfeats/0.1.0/)

## Installation

The latest stable release can be installed via PyPI:

```sh
pip install lfeats
```

Alternatively, the development version can be installed directly from the GitHub repository:

```sh
pip install git+https://github.com/takenori-y/lfeats.git@master
```

## Supported Models

| Model Name | Model Variant | Layers | Dimension | Source |
| :--- | :--- | :--- | :--- | :--- |
| `contentvec` | `hubert-100` | 12 | 768 | [auspicious3000/contentvec](https://github.com/auspicious3000/contentvec) |
| | `hubert-500` | 12 | 768 | |
| `hubert` | `base` | 12 | 768 | |
| | `large` | 24 | 1024 | |
| `rspin` | `wavlm-32` | 12 | 768 | [vectominist/rspin](https://github.com/vectominist/rspin) |
| | `wavlm-64` | 12 | 768 | |
| | `wavlm-128` | 12 | 768 | |
| | `wavlm-256` | 12 | 768 | |
| | `wavlm-512` | 12 | 768 | |
| | `wavlm-1024` | 12 | 768 | |
| | `wavlm-2048` | 12 | 768 | |
| `spin` | `hubert-128` | 12 | 768 | [vectominist/spin](https://github.com/vectominist/spin) |
| | `hubert-256` | 12 | 768 | |
| | `hubert-512` | 12 | 768 | |
| | `hubert-1024` | 12 | 768 | |
| | `hubert-2048` | 12 | 768 | |
| | `wavlm-128` | 12 | 768 | |
| | `wavlm-256` | 12 | 768 | |
| | `wavlm-512` | 12 | 768 | |
| | `wavlm-1024` | 12 | 768 | |
| | `wavlm-2048` | 12 | 768 | |
| `sslzip` | `tight` | 0 | 16 | [sp-nitech/SSLZip](https://github.com/sp-nitech/SSLZip) |
| | `loose` | 0 | 256 | |
| `whisper` | `tiny` | 4 | 384 | [openai/whisper](https://github.com/openai/whisper) |
| | `base` | 6 | 512 | |
| | `small` | 12 | 768 | |
| | `medium` | 24 | 1024 | |

## Examples

### Simple Usage

`lfeats` simplifies the process of extracting hidden states from various speech foundation models.
You don't need to worry about differences between model types or input/output data types.

```python
import lfeats
import numpy as np

# Prepare an audio waveform (1 second of random noise for this example).
# Either a NumPy array or a Torch tensor are accepted as the input of the extractor.
sample_rate = 16000
waveform = np.random.randn(sample_rate)

# Initialize the extractor.
extractor = lfeats.Extractor(model_name="hubert", model_variant="base", device="cpu")

# Note: The model weights are automatically loaded during the first call to extractor(),
# so calling extractor.load() explicitly is optional.
extractor.load()

# Extract features.
features = extractor(waveform, sample_rate)
print(f"Shape: {features.shape}")  # (1, 50, 768)

# You can access the features as a Numpy array.
print(type(features.array))  # <class 'numpy.ndarray'>

# You can also access the features as a Torch tensor.
print(type(features.tensor))  # <class 'torch.Tensor'>
```

### Layer Selection

`lfeats` allows you to extract features from specific layer(s).
By default, the last layer is used.

```python
import lfeats
import numpy as np

sample_rate = 16000
waveform = np.random.randn(sample_rate)

extractor = lfeats.Extractor(model_name="hubert")

# Get the second-to-last layer output.
features = extractor(waveform, sample_rate, layers=-2)
print(f"Shape: {features.shape}")  # (1, 50, 768)

# Get the multiple layer outputs.
features = extractor(waveform, sample_rate, layers=(11, 12))
print(f"Shape: {features.shape}")  # (1, 50, 1536)

# Get all layer outputs as a concatenated vector.
features = extractor(waveform, sample_rate, layers="all")
print(f"Shape: {features.shape}")  # (1, 50, 9984)
```

### Virtual Upsampling

Since the frame rate of speech foundation models is typically 20ms,
it often doesn't match the 5ms requirement of speech generation tasks.
`lfeats` bridges this gap by sliding the input waveform and interleaving the resulting features,
providing a high-resolution output.

```python
import lfeats
import numpy as np

sample_rate = 16000
waveform = np.random.randn(sample_rate)

extractor = lfeats.Extractor(model_name="hubert")

# Extract features at a 5ms frame rate.
features = extractor(waveform, sample_rate, upsample_factor=4)
print(f"Shape: {features.shape}")  # (1, 200, 768)
```

### Command-line Interface

Once installed via pip, you can use the `lfeats` command directly from your terminal.

```sh
# Basic usage: extract features from a wav file
$ lfeats input.wav --output_format npz

# Process all audio files in a directory
$ lfeats audio_dir --output_dir feats

# Process files listed in a file
$ lfeats input.scp --output_dir feats

# Specify model and layer
$ lfeats input.wav --model_name hubert --model_variant base --layer 12
```

> [!TIP]
> For more details on all available flags and default values, simply run:
>
> ```sh
> $ lfeats --help
> ```

## License

This project is released under the MIT License.

### Third-Party Licenses

`lfeats` incorporates the following repositories:

| Repository | License |
| :--- | :--- |
| [fairseq](https://github.com/facebookresearch/fairseq) | MIT License |
| [R-Spin](https://github.com/vectominist/rspin) | MIT License |
| [S3PRL](https://github.com/s3prl/s3prl) | Apache License 2.0 |
| [Spin](https://github.com/vectominist/spin) | MIT License |
