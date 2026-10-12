import numpy as np
import pytest

from evmatch.circuits import surface_code_si1000
from decoders import all_decoders


@pytest.mark.parametrize("d", [3, 5, 7])
def test_circuit_distance_and_determinism(d):
    c = surface_code_si1000(d, 0.001)
    assert len(c.shortest_graphlike_error()) == d
    assert not surface_code_si1000(d, 0.0).compile_detector_sampler().sample(200).any()


def _predictions(name, dem, dets):
    return all_decoders()[name].compile_decoder_for_dem(dem=dem).decode_shots_bit_packed(
        bit_packed_detection_event_data=dets
    )


@pytest.fixture(scope="module")
def shots():
    c = surface_code_si1000(5, 0.005)
    dem = c.detector_error_model(decompose_errors=True)
    dets, obs = c.compile_detector_sampler(seed=7).sample(400, separate_observables=True, bit_packed=True)
    return dem, dets, obs


@pytest.mark.parametrize(
    "native,proto",
    [
        ("nat-match", "proto-mwpm"),
        ("nat-correlated", "proto-corr"),
        ("nat-evidence", "proto-neg1"),
        ("nat-evidence-correlated", "proto-corr-neg-touch"),
    ],
)
def test_native_matches_prototype(shots, native, proto):
    dem, dets, _ = shots
    assert np.array_equal(_predictions(native, dem, dets), _predictions(proto, dem, dets))


def test_first_decomposition_tracks_pymatching(shots):
    dem, dets, _ = shots
    a = _predictions("nat-match-first", dem, dets)
    b = _predictions("pymatching", dem, dets)
    # Same graph up to merging of parallel mechanisms; ties may break differently.
    assert np.mean(np.any(a != b, axis=1)) < 0.02
