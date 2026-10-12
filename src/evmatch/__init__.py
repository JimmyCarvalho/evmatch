"""Evidence-conditioned correlated matching for circuit-level noise.

    import evmatch, sinter
    sinter.collect(..., decoders=["evmatch"], custom_decoders=evmatch.sinter_decoders())
"""

from evmatch.circuits import surface_code_si1000
from evmatch.decoder import MODES, CompiledEvMatch, EvMatchDecoder, sinter_decoders

__version__ = "0.1.0"

__all__ = ["CompiledEvMatch", "EvMatchDecoder", "MODES", "sinter_decoders", "surface_code_si1000"]
