"""
Unit tests for data labels source of truth and mapping.
"""

from deepfake_detector.data.labels import (
    FAKE,
    REAL,
    build_hf_label_map,
    get_canonical_label_from_name,
    get_label_name,
)


def test_canonical_constants():
    assert REAL == 0
    assert FAKE == 1
    assert get_label_name(REAL) == "real"
    assert get_label_name(FAKE) == "fake"


def test_class_name_normalization():
    assert get_canonical_label_from_name("RealArt") == REAL
    assert get_canonical_label_from_name("real") == REAL
    assert get_canonical_label_from_name("authentic") == REAL
    assert get_canonical_label_from_name("AiArtData") == FAKE
    assert get_canonical_label_from_name("fake") == FAKE
    assert get_canonical_label_from_name("synthetic") == FAKE


def test_hemg_dataset_mapping():
    # Hemg dataset: 0 is 'AiArtData' (Fake), 1 is 'RealArt' (Real)
    names = ["AiArtData", "RealArt"]
    mapping = build_hf_label_map(names)
    assert mapping[0] == FAKE  # Index 0 must map to FAKE (1)
    assert mapping[1] == REAL  # Index 1 must map to REAL (0)


def test_thien_dataset_mapping():
    # Thien0103 dataset: 0 is 'fake', 1 is 'real'
    names = ["fake", "real"]
    mapping = build_hf_label_map(names)
    assert mapping[0] == FAKE  # Index 0 must map to FAKE (1)
    assert mapping[1] == REAL  # Index 1 must map to REAL (0)
