"""Compatibility shims for HuggingFace transformers API drift.

Kept in ``brainscore_core`` because vision, language and unified all need the
same behaviour, and imported lazily so core stays free of heavy dependencies.
"""


def pin_image_processor(processor, model_id, use_fast=False, **kwargs):
    """Freeze which image-processor implementation ``processor`` uses.

    transformers rebound the image-processor class names: in 4.57
    ``CLIPImageProcessor`` is the PIL implementation, in 5.x the same name is the
    torchvision ("fast") one, with the PIL version moved to
    ``CLIPImageProcessorPil``. The *default* therefore changes across the
    upgrade, and preprocessed pixels move with it — measured at 3.0e-2 on
    ``pixel_values`` and 1.8e-2 on CLIP's vision output, which is far above float
    noise and would shift every vision score silently.

    Passing ``use_fast`` straight to ``AutoProcessor`` is **not** a fix: it also
    swaps the tokenizer (``CLIPTokenizerFast`` -> ``CLIPTokenizer``), changing
    the text path and slowing it down. This replaces only the image-processor
    component and leaves the tokenizer alone.

    Both explicit settings are stable across versions — it is only the default
    that moved — so pinning here makes the choice survive the upgrade.

    :param processor: a processor with an ``image_processor`` attribute. Anything
        else is returned untouched, so audio-only and text-only processors can be
        passed through without a caller-side branch.
    :param model_id: the checkpoint the processor was loaded from.
    :param use_fast: ``False`` selects the PIL implementation, matching the
        scores currently anchored on the leaderboard.
    """
    if not hasattr(processor, 'image_processor'):
        return processor
    from transformers import AutoImageProcessor
    processor.image_processor = AutoImageProcessor.from_pretrained(
        model_id, use_fast=use_fast, **kwargs)
    return processor


def load_pinned_processor(processor_cls, model_id, use_fast=False,
                          image_processor_kwargs=None, **kwargs):
    """``processor_cls.from_pretrained(model_id, **kwargs)`` with the image
    processor pinned — see :func:`pin_image_processor`."""
    processor = processor_cls.from_pretrained(model_id, **kwargs)
    return pin_image_processor(processor, model_id, use_fast=use_fast,
                               **(image_processor_kwargs or {}))
