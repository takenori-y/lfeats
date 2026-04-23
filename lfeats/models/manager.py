# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

"""A module for managing multiple models."""

from .base import BaseModel


class ModelManager:
    """A class for managing multiple models."""

    def __init__(
        self, model_cls: type[BaseModel], variant: str | None, device: str
    ) -> None:
        """Initialize the ModelManager with the specified model class.

        Parameters
        ----------
        model_cls : type[BaseModel]
            The model class to manage.

        variant : str
            The variant of the model to use.

        device : str
            The device to run the model on (e.g., 'cpu' or 'cuda').

        """
        self.model_cls = model_cls
        self.variant = variant
        self.device = device

        self._cache: dict[str, BaseModel] = {}

    def to(self, device: str) -> None:
        """Move all models to the specified device.

        Parameters
        ----------
        device : str
            The device to move the models to (e.g., 'cpu' or 'cuda').

        """
        self.device = device
        for model in self._cache.values():
            model.to(device)

    def get_model(self) -> BaseModel:
        """Get the model instance.

        Returns
        -------
        out : BaseModel
            The model instance.

        """
        key = "dummy"
        if key not in self._cache:
            self._cache[key] = self.model_cls(variant=self.variant, device=self.device)
        return self._cache[key]
