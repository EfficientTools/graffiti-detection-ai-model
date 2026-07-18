import pytest

from scripts.export_coreml import validate_image_size, validate_labels


def test_coreml_export_accepts_single_graffiti_class():
    validate_labels(["graffiti"])


@pytest.mark.parametrize("labels", [[], ["tag"], ["graffiti", "mural"]])
def test_coreml_export_rejects_incompatible_classes(labels):
    with pytest.raises(ValueError, match="exactly one model class named 'graffiti'"):
        validate_labels(labels)


@pytest.mark.parametrize("image_size", [0, -32, 641])
def test_coreml_export_rejects_invalid_image_size(image_size):
    with pytest.raises(ValueError, match="positive multiple of 32"):
        validate_image_size(image_size)
