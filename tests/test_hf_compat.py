"""Tests for the image-processor pinning shim.

The shim exists so a transformers upgrade cannot silently change preprocessed
pixels. Its two contracts are: it must replace the image processor, and it must
leave the tokenizer alone — passing ``use_fast`` to a processor class does the
latter wrong, which is the whole reason the shim exists.
"""

import pytest

from brainscore_core.hf_compat import load_pinned_processor, pin_image_processor


class _FakeImageProcessor:
    def __init__(self, tag):
        self.tag = tag


class _FakeTokenizer:
    pass


class _FakeProcessor:
    def __init__(self):
        self.image_processor = _FakeImageProcessor('default')
        self.tokenizer = _FakeTokenizer()


class _AudioOnlyProcessor:
    """No image_processor attribute — e.g. a Wav2Vec2 feature extractor."""

    def __init__(self):
        self.feature_extractor = object()


@pytest.fixture
def fake_auto_image_processor(monkeypatch):
    calls = []

    class _Auto:
        @staticmethod
        def from_pretrained(model_id, use_fast=None, **kwargs):
            calls.append({'model_id': model_id, 'use_fast': use_fast, **kwargs})
            return _FakeImageProcessor(f'pinned-{use_fast}')

    import sys
    import types
    module = types.ModuleType('transformers')
    module.AutoImageProcessor = _Auto
    monkeypatch.setitem(sys.modules, 'transformers', module)
    return calls


class TestPinImageProcessor:
    def test_replaces_the_image_processor(self, fake_auto_image_processor):
        processor = _FakeProcessor()
        out = pin_image_processor(processor, 'some/model')
        assert out.image_processor.tag == 'pinned-False'

    def test_defaults_to_the_pil_implementation(self, fake_auto_image_processor):
        pin_image_processor(_FakeProcessor(), 'some/model')
        assert fake_auto_image_processor[0]['use_fast'] is False

    def test_use_fast_true_is_honoured(self, fake_auto_image_processor):
        pin_image_processor(_FakeProcessor(), 'some/model', use_fast=True)
        assert fake_auto_image_processor[0]['use_fast'] is True

    def test_tokenizer_is_left_alone(self, fake_auto_image_processor):
        """The bug this shim avoids: use_fast on the processor also swaps the
        tokenizer to the slow implementation."""
        processor = _FakeProcessor()
        original = processor.tokenizer
        pin_image_processor(processor, 'some/model')
        assert processor.tokenizer is original

    def test_processor_without_images_passes_through_untouched(
            self, fake_auto_image_processor):
        """Audio and text processors must not need a caller-side branch."""
        processor = _AudioOnlyProcessor()
        assert pin_image_processor(processor, 'some/model') is processor
        assert fake_auto_image_processor == []      # nothing was loaded

    def test_model_id_is_forwarded(self, fake_auto_image_processor):
        pin_image_processor(_FakeProcessor(), 'openai/clip-vit-base-patch32')
        assert fake_auto_image_processor[0]['model_id'] == 'openai/clip-vit-base-patch32'


class TestLoadPinnedProcessor:
    def test_constructs_then_pins(self, fake_auto_image_processor):
        class _Cls:
            @staticmethod
            def from_pretrained(model_id, **kwargs):
                return _FakeProcessor()

        out = load_pinned_processor(_Cls, 'some/model')
        assert out.image_processor.tag == 'pinned-False'

    def test_forwards_kwargs_to_the_processor_class(self, fake_auto_image_processor):
        seen = {}

        class _Cls:
            @staticmethod
            def from_pretrained(model_id, **kwargs):
                seen.update(kwargs)
                return _FakeProcessor()

        load_pinned_processor(_Cls, 'some/model', revision='abc123')
        assert seen == {'revision': 'abc123'}
