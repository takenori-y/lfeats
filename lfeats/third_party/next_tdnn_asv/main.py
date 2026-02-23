import importlib

import torch.nn as nn

from SpeakerNet import SpeakerNet


class NeXtTDNNModel(nn.Module):
    def __init__(self, config_name):
        super().__init__()

        config = importlib.import_module('configs.' + config_name)

        # set feature extractor
        feature_extractor = importlib.import_module('preprocessing.' + config.FEATURE_EXTRACTOR).__getattribute__("feature_extractor")
        feature_extractor = feature_extractor(**config.FEATURE_EXTRACTOR_CONFIG)

        # set speaker embedding extractor
        model = importlib.import_module('models.' + config.MODEL).__getattribute__("MainModel")
        model = model(**config.MODEL_CONFIG)

        # set aggregation
        aggregation = importlib.import_module('aggregation.' + config.AGGREGATION).__getattribute__("Aggregation")
        aggregation = aggregation(**config.AGGREGATION_CONFIG)

        self.speaker_net = SpeakerNet(
            feature_extractor=feature_extractor,
            spec_aug=None,
            model=model,
            aggregation=aggregation,
            loss_function=None,
            print_model=False,
        )

    def forward(self, x):
        return self.speaker_net(x)
