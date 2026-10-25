"""Dataset parsing, manifests, validation, splits and audio preprocessing.

Dataset-specific parsing lives in :mod:`loaders`; every corpus is converted to
the normalised manifest schema in :mod:`manifests`, which is the only format
the feature extractors and models consume.
"""
