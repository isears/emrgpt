import torch
from emrgpt.sequenceData import EventSequence


def test_round_trip_event_sequence():
    """
    Test creating event sequence
    NOTE: there is information loss going offsets, encodings -> OHE so test starting at OHE
    """
    # TODO: currently doesn't cover edge case of 0 events in a timestep
    original_ohe = torch.tensor(
        [
            [0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        ]
    )

    es = EventSequence.from_ohe(original_ohe)
    recovered_ohe = es.to_ohe()

    assert original_ohe.shape == recovered_ohe.shape, "Shape mismatch after round-trip"
    assert torch.equal(
        original_ohe, recovered_ohe
    ), "OHE does not match after round-trip"
