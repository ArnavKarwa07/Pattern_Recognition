"""Robust detection of AI-generated speech: three pattern-recognition pipelines.

Class convention used everywhere in this package: 0 = bona fide (genuine human
speech), 1 = spoof (AI-generated or manipulated). Higher scores mean "more
likely spoof".
"""

import os as _os

# This project uses the PyTorch backend of Hugging Face Transformers only; stop
# it from importing TensorFlow/Flax (slow, and broken installs crash imports).
_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("USE_FLAX", "0")

__version__ = "0.1.0"

BONAFIDE = 0
SPOOF = 1
CLASS_NAMES = {BONAFIDE: "bonafide", SPOOF: "spoof"}
