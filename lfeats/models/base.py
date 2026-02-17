# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module defining the base model for feature extraction."""

from abc import ABC, abstractmethod

from ..interfaces.types import Audio, Features


class BaseModel(ABC):
    """An abstract base class for feature extraction models."""

    def __init__(self, variant: str | None = None, device: str = "cpu") -> None:
        """Initialize the BaseModel.

        Parameters
        ----------
        variant : str | None, optional
            The variant of the model to use.

        device : str, optional
            The device to run the model on (e.g., 'cpu' or 'cuda').

        """
        pass

    @abstractmethod
    def load(self, model_dir: str) -> None:
        """Load the model from the specified directory.

        Parameters
        ----------
        model_dir : str
            The directory where the model files are stored.

        """
        raise NotImplementedError

    def extract_features(self, audio: Audio, layers: list[int]) -> Features:
        """Extract features from the input audio data.

        Parameters
        ----------
        audio : Audio
            The input audio data.

        layers : list[int]
            The layer(s) from which to extract features.

        Returns
        -------
        out : Features
            The extracted features.

        Raises
        ------
        ValueError
            If the input sample rate does not match the model sample rate.

        """
        if audio.sample_rate != self.sample_rate:
            raise ValueError(
                f"Input sample rate {audio.sample_rate} does not match "
                f"model sample rate {self.sample_rate}."
            )
        return self.extract_features_impl(audio, layers)

    @abstractmethod
    def extract_features_impl(self, audio: Audio, layers: list[int]) -> Features:
        """Extract features from the input audio data.

        Parameters
        ----------
        audio : Audio
            The input audio data.

        layers : list[int]
            The layer(s) from which to extract features.

        Returns
        -------
        out : Features
            The extracted features.

        """
        raise NotImplementedError

    @property
    def sample_rate(self) -> int:
        """Get the sample rate required by the model.

        Returns
        -------
        out : int
            The sample rate in Hz.

        """
        return 16000

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Get the number of available layers in the model.

        Returns
        -------
        out : int
            The number of layers.

        """
        raise NotImplementedError

    @property
    def frame_shift(self) -> int:
        """Get the frame shift of the model.

        Returns
        -------
        out : int
            The frame shift in samples.

        """
        return int(20.0 * self.sample_rate / 1000)

    @property
    def center_offset(self) -> int:
        """Get the center offset of the model.

        The feature extraction layer in the upstream model employs a VALID convolution
        with a receptive field of 25 ms, resulting in a center offset of 12.5 ms.

        Returns
        -------
        out : int
            The center offset in samples.

        """
        return int(12.5 * self.sample_rate / 1000)

    @property
    def chunk_length_sec(self) -> int | None:
        """Get the chunk length in seconds of the model, if applicable.

        Returns
        -------
        out : int | None
            The chunk length in seconds, or None if not applicable.

        """
        return None

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Get the model identifier.

        Returns
        -------
        out : str
            The model identifier.

        """
        raise NotImplementedError
