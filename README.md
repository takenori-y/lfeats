# lfeats

*lfeats* provides a unified interface to extract hidden representations from various speech foundation models such as HuBERT and Whisper.

## Requirements

- Python 3.10+
- PyTorch 2.4.0+

## Documentation

- [**Reference Manual**](https://takenori-y.github.io/lfeats/0.1.0/)

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
| `hubert` | `base` | 12 | 768 | |
| | `large` | 24 | 1024 | |
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

## License

This project is released under the MIT License.

This package incorpolates the following repositories:

| Repository | License |
| :--- | :--- |
| [S3PRL](https://github.com/s3prl/s3prl) | Apache License 2.0 |
| [Spin](https://github.com/vectominist/spin) | MIT License |
