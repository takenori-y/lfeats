# lfeats

*lfeats* provides a unified interface to extract hidden representations from various speech foundation models such as HuBERT and Whisper.

## Requirements

- Python 3.10+
- PyTorch 2.3.1+

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

| Model Name | Variant | Dimension | Description |
| :--- | :--- | :--- | :--- |
| `spin` | `hubert-128` | 768 | HuBERT encoder with codebook size 128 |
| | `hubert-256` | 768 | |
| | `hubert-512` | 768 | |
| | `hubert-1024` | 768 | |
| | `hubert-2048` | 768 | |
| | `wavlm-128` | 768 | WavLM encoder with codebook size 128 |
| | `wavlm-256` | 768 | |
| | `wavlm-512` | 768 | |
| | `wavlm-1024` | 768 | |
| | `wavlm-2048` | 768 | |

## License

This project is released under the MIT License.

This package incorpolates the following repositories:

| Repository | License |
| :--- | :--- |
| [S3PRL](https://github.com/s3prl/s3prl) | Apache License 2.0 |
| [Spin](https://github.com/vectominist/spin) | MIT License |
