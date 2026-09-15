"""
sigdist:  Signature Distance for comparing empirical distributions.

Core modules:
    distance  :  SignatureDistance class: SD/ED computation, p-values, bootstrap CI
    losses    :  Differentiable training losses: sd, csd, gsd, ediv (summed empirical conventions)
    generators:  FICGenerator architectures for distributional training
    utils     :  Data loading, clustering, visualization helpers
"""

from .distance import SignatureDistance
from . import losses, generators

__version__ = "0.2.1"
__all__ = [
    "SignatureDistance",
    "losses", "generators",
]
