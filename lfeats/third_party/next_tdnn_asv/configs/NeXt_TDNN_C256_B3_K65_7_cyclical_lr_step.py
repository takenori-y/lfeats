CHANNEL_SIZE = 256
EMBEDING_SIZE = 192

FEATURE_EXTRACTOR = 'mel_transform'
FEATURE_EXTRACTOR_CONFIG = {
    'sample_rate': 16000,
    'n_fft': 512,
    'win_length': 400,
    'hop_length': 160,
    'n_mels': 80,
    'coef': 0.97,
}

MODEL = 'NeXt_TDNN'
MODEL_CONFIG = {
    'depths': [3, 3, 3],
    'dims': [CHANNEL_SIZE, CHANNEL_SIZE, CHANNEL_SIZE],
    'kernel_size': [7, 65],
    'block': 'TSConvNeXt',
}

AGGREGATION = 'vap_bn_tanh_fc_bn'
AGGREGATION_CONFIG = {
    'channel_size': int(3*CHANNEL_SIZE),
    'intermediate_size': int(3*CHANNEL_SIZE/8),
    'embeding_size': EMBEDING_SIZE,
}
