# lfeats

*lfeats* provides a unified interface to extract hidden representations from various speech foundation models such as HuBERT and Whisper.
While these extracted features are task-independent, the package is primarily designed for speech generation tasks including text-to-speech and voice conversion.

[![Manual](https://img.shields.io/badge/docs-stable-blue.svg)](https://takenori-y.github.io/lfeats/stable/)
[![Downloads](https://static.pepy.tech/badge/lfeats)](https://pepy.tech/project/lfeats)
[![ClickPy](https://img.shields.io/badge/downloads-clickpy-yellow.svg)](https://clickpy.clickhouse.com/dashboard/lfeats)
[![Python Version](https://img.shields.io/pypi/pyversions/lfeats.svg)](https://pypi.python.org/pypi/lfeats)
[![PyPI Version](https://img.shields.io/pypi/v/lfeats.svg)](https://pypi.python.org/pypi/lfeats)
[![PyTorch Version](https://img.shields.io/badge/pytorch-2.6.0-orange.svg)](https://pypi.python.org/pypi/lfeats)
[![License](https://img.shields.io/github/license/takenori-y/lfeats.svg)](https://github.com/takenori-y/lfeats/blob/master/LICENSE)
[![GitHub Actions](https://github.com/takenori-y/lfeats/workflows/package/badge.svg)](https://github.com/takenori-y/lfeats/actions)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Requirements

- Python 3.10+
- PyTorch 2.6.0+

## Documentation

- [**Reference Manual**](https://takenori-y.github.io/lfeats/stable/)

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

### Frame-Level Features

| Model Name | Model Variant | Layers | Dimension | Paper | Source | Model Hub |
| :--- | :--- | ---: | ---: | :---: | :---: | :---: |
| `contentvec` | `hubert-100` | 12 | 768 | [arXiv](https://arxiv.org/abs/2204.09224) | [GitHub](https://github.com/auspicious3000/contentvec) | |
| | `hubert-500` | 12 | 768 | | | |
| `data2vec` | `base` | 12 | 768 | [arXiv](https://arxiv.org/abs/2202.03555) | [GitHub](https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec) | |
| | `large` | 24 | 1024 | | | |
| `data2vec2` | `base` | 8 | 768 | [arXiv](https://arxiv.org/abs/2212.07525) | [GitHub](https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec) | |
| | `large` | 16 | 1024 | | | |
| `emotion2vec` | `base` | 8 | 768 | [arXiv](https://arxiv.org/abs/2312.15185) | [GitHub](https://github.com/ddlBoJack/emotion2vec) | [🤗](https://huggingface.co/emotion2vec/emotion2vec_base) |
| `emotion2vec+` | `seed` | 8 | 768 | | | [🤗](https://huggingface.co/emotion2vec/emotion2vec_plus_seed) |
| | `base` | 8 | 768 | | | [🤗](https://huggingface.co/emotion2vec/emotion2vec_plus_base) |
| | `large` | 8 | 1024 | | | [🤗](https://huggingface.co/emotion2vec/emotion2vec_plus_large) |
| `hubert` | `base` | 12 | 768 | [arXiv](https://arxiv.org/abs/2106.07447) | [GitHub](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert) | [🤗](https://huggingface.co/facebook/hubert-base-ls960) |
| | `large` | 24 | 1024 | | | [🤗](https://huggingface.co/facebook/hubert-large-ll60k) |
| | `xlarge` | 48 | 1280 | | | [🤗](https://huggingface.co/facebook/hubert-xlarge-ll60k) |
| `r-spin` | `wavlm-32` | 12 | 768 | [arXiv](https://arxiv.org/abs/2311.09117) | [GitHub](https://github.com/vectominist/rspin) | |
| | `wavlm-64` | 12 | 768 | | | |
| | `wavlm-128` | 12 | 768 | | | |
| | `wavlm-256` | 12 | 768 | | | |
| | `wavlm-512` | 12 | 768 | | | |
| | `wavlm-1024` | 12 | 768 | | | |
| | `wavlm-2048` | 12 | 768 | | | |
| `spidr` | `base` | 12 | 768 | [arXiv](https://arxiv.org/abs/2512.20308) | [GitHub](https://github.com/facebookresearch/spidr) | |
| `spin` | `hubert-128` | 12 | 768 | [arXiv](https://arxiv.org/abs/2305.11072) | [GitHub](https://github.com/vectominist/spin) | |
| | `hubert-256` | 12 | 768 | | | |
| | `hubert-512` | 12 | 768 | | | |
| | `hubert-1024` | 12 | 768 | | | |
| | `hubert-2048` | 12 | 768 | | | |
| | `wavlm-128` | 12 | 768 | | | |
| | `wavlm-256` | 12 | 768 | | | |
| | `wavlm-512` | 12 | 768 | | | |
| | `wavlm-1024` | 12 | 768 | | | |
| | `wavlm-2048` | 12 | 768 | | | |
| `sslzip` | `tiny` | 0 | 16 | [ISCA](https://www.isca-archive.org/ssw_2025/yoshimura25_ssw.html) | [GitHub](https://github.com/sp-nitech/SSLZip) | [🤗](https://huggingface.co/takenori-y/SSLZip-16) |
| | `base` | 0 | 256 | | | [🤗](https://huggingface.co/takenori-y/SSLZip-256) |
| `unispeech-sat` | `base` | 12 | 768 | [arXiv](https://arxiv.org/abs/2110.05752) | [GitHub](https://github.com/microsoft/UniSpeech) | [🤗](https://huggingface.co/microsoft/unispeech-sat-base) |
| | `base+` | 12 | 768 | | | [🤗](https://huggingface.co/microsoft/unispeech-sat-base-plus) |
| | `large` | 24 | 1024 | | | [🤗](https://huggingface.co/microsoft/unispeech-sat-large) |
| `wav2vec2` | `base` | 12 | 768 | [arXiv](https://arxiv.org/abs/2006.11477) | [GitHub](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec) | |
| | `large` | 24 | 1024 | | | |
| | `xlsr` | 24 | 1024 | [arXiv](https://arxiv.org/abs/2006.13979) | | |
| | `xlsr-v2` | 24 | 1024 | [arXiv](https://arxiv.org/abs/2111.09296) | [GitHub](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec/xlsr) | |
| `wavlm` | `base` | 12 | 768 | [arXiv](https://arxiv.org/abs/2110.13900) | [GitHub](https://github.com/microsoft/unilm/tree/master/wavlm) | [🤗](https://huggingface.co/microsoft/wavlm-base) |
| | `base+` | 12 | 768 | | | [🤗](https://huggingface.co/microsoft/wavlm-base-plus) |
| | `large` | 24 | 1024 | | | [🤗](https://huggingface.co/microsoft/wavlm-large) |
| `whisper` | `tiny` | 4 | 384 | [arXiv](https://arxiv.org/abs/2212.04356) | [GitHub](https://github.com/openai/whisper) | [🤗](https://huggingface.co/openai/whisper-tiny) |
| | `base` | 6 | 512 | | | [🤗](https://huggingface.co/openai/whisper-base) |
| | `small` | 12 | 768 | | | [🤗](https://huggingface.co/openai/whisper-small) |
| | `medium` | 24 | 1024 | | | [🤗](https://huggingface.co/openai/whisper-medium) |
| | `large` | 32 | 1280 | | | [🤗](https://huggingface.co/openai/whisper-large) |
| | `large-v2` | 32 | 1280 | | | [🤗](https://huggingface.co/openai/whisper-large-v2) |
| | `large-v3` | 32 | 1280 | | | [🤗](https://huggingface.co/openai/whisper-large-v3) |

### Utterance-Level Features

| Model Name | Model Variant | Layers | Dimension | Paper | Source | Model Hub |
| :--- | :--- | ---: | ---: | :---: | :---: | :---: |
| `ecapa-tdnn` | `base` | 0 | 192 | [arXiv](https://arxiv.org/abs/2005.07143) | [GitHub](https://github.com/speechbrain/speechbrain) | [🤗](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) |
| `next-tdnn` | `light` | 0 | 192 | [arXiv](https://arxiv.org/abs/2312.08603) | [GitHub](https://github.com/dmlguq456/NeXt_TDNN_ASV) | |
| | `base` | 0 | 192 | | | |
| | `base-v2` | 0 | 192 | | | |
| `r-vector` | `base` | 0 | 256 | [arXiv](https://arxiv.org/pdf/1910.12592) | [GitHub](https://github.com/speechbrain/speechbrain) | [🤗](https://huggingface.co/speechbrain/spkrec-resnet-voxceleb) |
| `x-vector` | `base` | 0 | 512 | [IEEE](https://ieeexplore.ieee.org/document/8461375) | [GitHub](https://github.com/speechbrain/speechbrain) | [🤗](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) |

> [!IMPORTANT]
> Users must comply with the respective licenses of the models.
> Please refer to the original repositories for detailed licensing information.

## Supported Resamplers

| Resampler Type | Quality Preset | Source | License |
| :--- | :--- | :---: | :--- |
| `lilfilter` | `base` | [GitHub](https://github.com/danpovey/filtering) | MIT |
| `soxr` | `quick` | [GitHub](https://github.com/dofuuz/python-soxr) | LGPL v2.1+ |
| | `low` | | |
| | `medium` | | |
| | `high` | | |
| | `very-high` | | |
| `torchaudio` | `kaiser-fast` | [GitHub](https://github.com/pytorch/audio) | BSD 2-Clause |
| | `kaiser-best` | | |

## Examples

### Simple Usage

`lfeats` simplifies the process of extracting hidden states from various speech foundation models.
You don't need to worry about differences between model types or input/output data types.

```python
import lfeats
import numpy as np

# Prepare an audio waveform without zero-mean and unit-variance normalization.
# Either a NumPy array or a Torch tensor are accepted as the input of the extractor.
sample_rate = 16000
waveform = np.random.uniform(-1, 1, sample_rate)

# Initialize the extractor.
extractor = lfeats.Extractor(
    model_name="hubert",
    model_variant="base",
    resampler_type="torchaudio",
    resampler_preset="kaiser-best",
    device="cpu",
)

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
waveform = np.random.uniform(-1, 1, sample_rate)

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

### Audio Chuking

To be computationally efficient and prevent mismatches between training and inference,
long audio files can be processed by splitting them into chunks.

```python
import lfeats
import numpy as np

sample_rate = 16000
waveform = np.random.uniform(-1, 1, 10 * sample_rate)

extractor = lfeats.Extractor(model_name="hubert")

# Processing a 10-second waveform with a 5-second chunk and 1-second overlap.
features = extractor(waveform, sample_rate, chunk_length_sec=5, overlap_length_sec=1)
print(f"Shape: {features.shape}")  # (1, 500, 768)
```

### Sliding-Window Upsampling

Since the frame rate of speech foundation models is typically 20ms,
it often doesn't match the 5ms requirement of speech generation tasks.
`lfeats` bridges this gap by sliding the input waveform and interleaving the resulting features,
providing a high-resolution output.

```python
import lfeats
import numpy as np

sample_rate = 16000
waveform = np.random.uniform(-1, 1, sample_rate)

extractor = lfeats.Extractor(model_name="hubert")

# Extract features at a 5ms frame rate.
features = extractor(waveform, sample_rate, upsample_factor=4)
print(f"Shape: {features.shape}")  # (1, 200, 768)
```

### Utterance-Level Feature Extraction

`lfeats` can extract utterance-level features, e.g., speaker embeddings, as well as frame-level features.

```python
import lfeats
import numpy as np

sample_rate = 16000
waveform = np.random.uniform(-1, 1, sample_rate)

extractor = lfeats.Extractor(model_name="ecapa-tdnn")

# The default aggregation method for utterance-level features is averaging.
features = extractor(waveform, sample_rate, overlap_length_sec=0, reduction="mean")
print(f"Shape: {features.shape}")  # (1, 1, 192)
```

### Command-line Interface

Once installed via pip, you can use the `lfeats` command directly from your terminal.

```sh
# Basic usage: extract features from a wav file
$ lfeats input.wav --output_format npz

# Process all audio files in a directory
$ lfeats input/dir --output_dir feats

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

`lfeats` partially incorporates the following repositories:

| Repository | License |
| :--- | :--- |
| [fairseq](https://github.com/facebookresearch/fairseq) | MIT |
| [NeXt_TDNN_ASV](https://github.com/dmlguq456/NeXt_TDNN_ASV) | Apache-2.0 |
| [R-Spin](https://github.com/vectominist/rspin) | MIT |
| [S3PRL](https://github.com/s3prl/s3prl) | Apache-2.0 |
| [SpeechBrain](https://github.com/speechbrain/speechbrain) | Apache-2.0 |
| [Spin](https://github.com/vectominist/spin) | MIT |
| [timm](https://github.com/huggingface/pytorch-image-models) | Apache-2.0 |
