from importlib import import_module


_MODEL_EXPORTS = {
    "ConvolutionalFeatureRefinementExtractor",
    "HyDRA",
    "MambaLinearFlowEncoder",
    "MinimalMambaBlock",
    "ResidualTemporalConv1d",
    "TransformerDynamicSequenceEncoder",
}
_OPEN_SET_EXPORTS = {
    "OpenSetDecision",
    "max_softmax_decision",
    "temperature_softmax",
}
_PREPROCESSING_EXPORTS = {
    "central_dft_indices",
    "decompose_iq",
    "lossless_vmd_1d",
    "normalize_iq_power",
}

__all__ = [
    "ConvolutionalFeatureRefinementExtractor",
    "HyDRA",
    "MambaLinearFlowEncoder",
    "MinimalMambaBlock",
    "OpenSetDecision",
    "ResidualTemporalConv1d",
    "TransformerDynamicSequenceEncoder",
    "central_dft_indices",
    "decompose_iq",
    "lossless_vmd_1d",
    "max_softmax_decision",
    "normalize_iq_power",
    "temperature_softmax",
]


def __getattr__(name):
    if name in _MODEL_EXPORTS:
        return getattr(import_module(".model", __name__), name)
    if name in _OPEN_SET_EXPORTS:
        return getattr(import_module(".open_set", __name__), name)
    if name in _PREPROCESSING_EXPORTS:
        return getattr(import_module(".preprocessing", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
