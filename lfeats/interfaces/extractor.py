# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for extracting features from a specified model."""

from collections import namedtuple
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from platformdirs import user_cache_dir

from ..models import MODEL_MAP, ModelManager
from ..resamplers import RESAMPLER_MAP, ResamplerManager
from .types import Audio, Features

Chunk = namedtuple("Chunk", ["start", "end", "overlap"])


class Extractor:
    """A class for extracting features from a specified model."""

    def __init__(
        self,
        model_name: str,
        model_variant: str | None = None,
        resampler_type: str = "torchaudio",
        resampler_preset: str | None = None,
        device: str = "cpu",
        cache_dir: str | None = None,
    ) -> None:
        """Initialize the Extractor with the specified model.

        Parameters
        ----------
        model_name : str
            The name of the model to use.

        model_variant : str | None, optional
            The variant of the model to use.

        resampler_type : str, optional
            The type of resampler to use.

        resampler_preset : str | None, optional
            The preset for the resampler. If None, the highest quality preset is used.

        device : str, optional
            The device to run the model on (e.g., 'cpu' or 'cuda').

        cache_dir : str | None, optional
            The directory to cache the model files.

        """
        try:
            cls = MODEL_MAP[model_name]
            self.model_manager = ModelManager(cls, variant=model_variant, device=device)
        except KeyError as e:
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Supported models are: {[k for k in MODEL_MAP.keys()]}"
            ) from e

        try:
            cls = RESAMPLER_MAP[resampler_type]
            self.resampler_manager = ResamplerManager(
                cls, preset=resampler_preset, device=device
            )
        except KeyError as e:
            raise ValueError(
                f"Unsupported resampler '{resampler_type}'. "
                f"Supported resamplers are: {[k for k in RESAMPLER_MAP.keys()]}"
            ) from e

        if cache_dir is None:
            self.cache_dir = user_cache_dir("lfeats")
        else:
            self.cache_dir = cache_dir

    def load(self) -> None:
        """Download and load the model if it is not already loaded."""
        self.model_manager.get_model().load(self.cache_dir)

    def __call__(
        self,
        source: np.ndarray | torch.Tensor | Audio,
        sample_rate: int | None = None,
        layers: int | Sequence[int] | Literal["all", "last"] = "last",
        chunk_length_sec: int = 30,
        overlap_length_sec: int = 5,
        upsample_factor: int = 1,
    ) -> Features:
        """Extract features from the input waveform.

        Parameters
        ----------
        source : np.ndarray | torch.Tensor | Audio
            The input waveform data with shape (T,) or (B, T) or an Audio object.

        sample_rate : int | None, optional
            The sample rate of the input waveform.
            Must be provided if `source` is not an Audio object.

        layers : int | list[int] | Literal["all", "last"], optional
            The layer(s) from which to extract features.

        chunk_length_sec : int, optional
            The chunk length in seconds for processing long audio.

        overlap_length_sec : int, optional
            The overlap length in seconds between chunks.

        upsample_factor : int, optional
            The factor by which to upsample the features in the time dimension.

        Returns
        -------
        out : Features
            The extracted features.

        Raises
        ------
        ValueError
            If the given parameters are invalid.

        RuntimeError
            If the number of frames in the extracted features is unexpected.

        """
        if upsample_factor < 1:
            raise ValueError("upsample_factor must be a positive integer.")

        if upsample_factor == 1:
            return self._extract(
                source,
                sample_rate,
                layers,
                chunk_length_sec,
                overlap_length_sec,
            )

        # Prepare the audio data and validate the upsample factor.
        audio = self._create_audio_object(source, sample_rate)
        B, T = audio.data.shape
        frame_shift = self.model_manager.get_model().frame_shift
        if frame_shift % upsample_factor != 0:
            raise ValueError(
                f"Upsample factor {upsample_factor} must be a divisor of "
                f"the model's frame shift {frame_shift}."
            )
        step = frame_shift // upsample_factor

        # Create shifted waveforms for upsampling the features.
        shifted_waveforms = audio.zeros((B * upsample_factor, T))
        for i in range(upsample_factor):
            offset = i * step
            end = T - offset
            shifted_waveforms[i::upsample_factor, :end] = audio.data[:, offset:]  # type: ignore

        # Extract features from the shifted waveforms.
        features = self._extract(
            shifted_waveforms,
            audio.sample_rate,
            layers,
            chunk_length_sec,
            overlap_length_sec,
        )

        # Interleave the features from the shifted waveforms.
        _, N, D = features.data.shape
        interleaved_features = features.zeros((B, upsample_factor * N, D))
        for i in range(upsample_factor):
            interleaved_features[:, i::upsample_factor] = features.data[  # type: ignore
                i::upsample_factor
            ]

        return Features(
            data=interleaved_features, source=features.source, layers=features.layers
        )

    def _extract(
        self,
        source: np.ndarray | torch.Tensor | Audio,
        sample_rate: int | None = None,
        layers: int | Sequence[int] | Literal["all", "last"] = "last",
        chunk_length_sec: int = 30,
        overlap_length_sec: int = 5,
    ) -> Features:
        """Extract features from the input waveform.

        Parameters
        ----------
        source : np.ndarray | torch.Tensor | Audio
            The input waveform data with shape (T,) or (B, T) or an Audio object.

        sample_rate : int | None, optional
            The sample rate of the input waveform.
            Must be provided if `source` is not an Audio object.

        layers : int | list[int] | Literal["all", "last"], optional
            The layer(s) from which to extract features.

        chunk_length_sec : int, optional
            The chunk length in seconds for processing long audio.

        overlap_length_sec : int, optional
            The overlap length in seconds between chunks.

        Returns
        -------
        out : Features
            The extracted features.

        Raises
        ------
        ValueError
            If the given parameters are invalid.

        RuntimeError
            If the number of frames in the extracted features is unexpected.

        """
        if (
            chunk_length_sec < 1
            or overlap_length_sec < 0
            or chunk_length_sec <= overlap_length_sec
        ):
            raise ValueError("Invalid chunk_length_sec and overlap_length_sec values.")

        # Load the model.
        model = self.model_manager.get_model()
        model.load(self.cache_dir)

        if model.chunk_length_sec is not None:
            chunk_length_sec = model.chunk_length_sec

        # Prepare the audio data.
        audio = self._create_audio_object(source, sample_rate)

        expected_num_frames = self._get_num_frames(audio.length, model.frame_shift)
        normalized_layers = self._normalize_layers(layers, model.num_layers)

        # Resample the audio if needed.
        if sample_rate != model.sample_rate:
            resampler = self.resampler_manager.get_resampler(
                audio.sample_rate, model.sample_rate
            )
            audio = resampler.resample(audio)

        # Pad the audio if needed.
        padding = (model.center_offset, max(model.center_offset - 1, 0))
        if model.center_offset > 0:
            audio = audio.pad(padding)

        # Calculate chunk start and end indices considering padding and overlap.
        chunks = self._create_chunks(
            audio.length,
            padding,
            int(chunk_length_sec) * model.sample_rate,
            int(overlap_length_sec) * model.sample_rate,
            model.frame_shift,
        )

        # Extract features for each chunk and concatenate them.
        features = None
        for chunk in chunks:
            audio_chunk = Audio(
                audio.data[:, chunk.start : chunk.end], audio.sample_rate
            )
            chunk_features = model.extract_features(audio_chunk, normalized_layers)

            # Trim the features if the model outputs fixed-length features
            # regardless of the input length.
            if model.chunk_length_sec is not None:
                end = self._get_num_frames(chunk.end - chunk.start, model.frame_shift)
                chunk_features = chunk_features.trim(0, end)

            if features is None:
                features = chunk_features
            else:
                # Merge the chunk features with the previously extracted features.
                overlap = chunk.overlap
                if overlap is None:
                    remaining_frames = expected_num_frames - features.length
                    overlap = chunk_features.length - remaining_frames
                features = features.merge(chunk_features, overlap)

        if features is None:
            raise RuntimeError("No features extracted from the audio.")

        # Validate the number of frames in the extracted features.
        actual_num_frames = features.length
        if expected_num_frames != actual_num_frames:
            raise RuntimeError(
                f"Unexpected number of frames: expected {expected_num_frames}, "
                f"got {actual_num_frames}."
            )

        return features

    @staticmethod
    def _get_num_frames(length: int, frame_shift: int) -> int:
        """Calculate the number of frames for a given sample length.

        Parameters
        ----------
        length : int
            The length of the audio in samples.

        frame_shift : int
            The frame shift in samples of the model.

        Returns
        -------
        out : int
            The number of frames corresponding to the given length.

        """
        return int(np.ceil(length / frame_shift))

    @staticmethod
    def _normalize_layers(
        layers: int | Sequence[int] | Literal["all", "last"], num_layers: int
    ) -> list[int]:
        """Normalize the layer specification to a list of layer indices.

        Parameters
        ----------
        layers : int | list[int] | Literal["all", "last"]
            The layer(s) from which to extract features.

        num_layers : int
            The total number of layers in the model.

        Returns
        -------
        out : list[int]
            The normalized list of layer indices.

        Raises
        ------
        ValueError
            If the layers specification is invalid.

        """
        if layers == "all":
            return list(range(num_layers))
        if layers == "last":
            return [num_layers - 1]
        if isinstance(layers, str):
            raise ValueError(f"Invalid layers specification string: {layers}")

        if isinstance(layers, int):
            layers = [layers]
        if isinstance(layers, Sequence):
            layers = [i if i >= 0 else num_layers + i for i in layers]
            if any(i < 0 or i >= num_layers for i in layers):
                raise ValueError(f"Layer index out of range: {layers}")
            return layers

        raise ValueError(f"Invalid layers specification type: {layers}")

    @staticmethod
    def _create_audio_object(
        source: np.ndarray | torch.Tensor | Audio, sample_rate: int | None
    ) -> Audio:
        """Create an Audio object from the input source.

        Parameters
        ----------
        source : np.ndarray | torch.Tensor | Audio
            The input waveform data with shape (T,) or (B, T) or an Audio object.

        sample_rate : int | None
            The sample rate of the input waveform.

        Returns
        -------
        out : Audio
            The created Audio object.

        Raises
        ------
        ValueError
            If the sample_rate is invalid for the given source.

        """
        if isinstance(source, Audio):
            if sample_rate is not None and source.sample_rate != sample_rate:
                raise ValueError(
                    "sample_rate must be None when source is an Audio object "
                    "or must match the sample rate of the Audio object."
                )
            audio = source
        else:
            if sample_rate is None:
                raise ValueError(
                    "sample_rate must be provided when source is not an Audio object."
                )
            audio = Audio(source, sample_rate)
        return audio

    @staticmethod
    def _create_chunks(
        audio_length: int,
        padding: tuple[int, int],
        chunk_length: int,
        overlap_length: int,
        frame_shift: int,
    ) -> list[Chunk]:
        """Create a list of chunks with start and end indices for processing long audio.

        Parameters
        ----------
        audio_length : int
            The length of the audio in samples.

        padding : tuple[int, int]
            The left and right padding in samples.

        chunk_length : int
            The chunk length in samples.

        overlap_length : int
            The overlap length in samples between chunks.

        frame_shift : int
            The frame shift in samples of the model.

        Returns
        -------
        out : list[Chunk]
            A list of Chunk namedtuples with start, end, and overlap information.

        """
        chunks = []
        left_padding, right_padding = padding
        for chunk_start in range(
            left_padding, audio_length - right_padding, chunk_length - overlap_length
        ):
            s = chunk_start - left_padding
            e = chunk_start + chunk_length + right_padding

            is_last = e >= audio_length
            if is_last:
                s = max(audio_length - chunk_length - sum(padding), 0)
                s = int(np.ceil(s / frame_shift) * frame_shift)  # align to frame shift
                e = audio_length
                overlap_frames = None
            elif len(chunks) == 0:
                overlap_frames = 0
            else:
                overlap_frames = Extractor._get_num_frames(overlap_length, frame_shift)

            chunks.append(Chunk(s, e, overlap_frames))
            if is_last:
                break

        return chunks
