"""Canonical dataset and license vocabularies for schema 2.0 metadata.

Unlisted values produce warnings, never validation errors, so existing plugin
metadata keeps loading. Use ``other`` with free text (dataset ``name``, or
``other: <text>`` for a license) when nothing listed fits.
"""

from .contract import get_path

OTHER = "other"
DATASET_PATHS = (
    "/data/training_datasets",
    "/eval/test_datasets",
    "/eval/validation_datasets",
)

# Aliases are identifiers already present in published metadata.
DATASETS = {
    "brain-score-benchmarks": {
        "name": "Brain-Score benchmark stimulus sets",
        "aliases": (
            "brain-score-benchmark-stimuli-sets-free",
            "brain-score-neural-behavioral-benchmark",
        ),
    },
    "geirhos-cue-conflict": {
        "name": "Geirhos cue-conflict stimuli",
        "aliases": (),
    },
    "ilsvrc-2010": {
        "name": "ImageNet LSVRC-2010",
        "aliases": (
            "imagenet-lsvrc-2010",
        ),
    },
    "imagenet-12k": {
        "name": "ImageNet-12k (timm 11,821-class subset)",
        "aliases": (
            "imagenet-12k-11-821-class-subset",
            "imagenet-12k-11-821-classes",
        ),
    },
    "imagenet-1k": {
        "name": "ImageNet-1k (ILSVRC-2012)",
        "aliases": (
            "imagenet-1k-1001-class-tf-slim-style-tr",
            "imagenet-1k-assumed-standard-classific",
            "imagenet-1k-full-per-scaling-law-study",
            "imagenet-1k-ilsvrc2012-full-training",
            "imagenet-1k-madrylab-robustness-library",
            "imagenet-1k-madrylab-style-adversarial",
            "imagenet-1k-only",
            "imagenet-1k-only-ilsvrc-2012-1000-clas",
            "imagenet-1k-only-no-separate-pretrainin",
            "imagenet-1k-original-tf-pnasnet-trainin",
            "imagenet-1k-standard-validation-split",
            "imagenet-1k-trained-directly",
            "imagenet-1k-val",
            "imagenet-1k-val-top-1-87-29-top-5-98",
            "imagenet-1k-val-top-1-87-86-top-5-98",
            "imagenet-1k-val-top-1-87-90-top-5-98",
            "imagenet-1k-val-top-1-88-16-top-5-98",
            "imagenet-1k-val-top-1-88-26-top-5-98",
            "imagenet-1k-validation-set",
            "imagenet-1k-validation-set-50k-images",
            "imagenet-1k-validation-set-presumed",
            "imagenet-1k-validation-set-presumed-st",
            "imagenet-1k-validation-set-same-split",
            "imagenet-1k-validation-set-same-split-u",
            "imagenet-1k-validatioon-set",
            "imagenet-1k-via-torchvision-imagenet1k",
            "imagenet-1k-via-torchvision-pretrained",
            "imagenet-ilsvrc-2012",
            "imagenet-ilsvrc-2012-validation-set",
            "imagenet-ilsvrc-full-training-set",
            "imagenet-ilsvrc2012",
            "imagenet-large-scale-visual-recognition",
            "imagenet-validation-set",
            "imagenet-validation-set-used-for-robust",
            "imagenet-validation-set-used-in-the-sou",
        ),
    },
    "imagenet-21k": {
        "name": "ImageNet-21k (also called ImageNet-22k)",
        "aliases": (
            "imagenet-22k",
        ),
    },
    "imagenet-a": {
        "name": "ImageNet-A",
        "aliases": (),
    },
    "imagenet-c": {
        "name": "ImageNet-C",
        "aliases": (),
    },
    "instagram-940m": {
        "name": "Instagram 940M hashtag images",
        "aliases": (),
    },
    "jft-300m": {
        "name": "JFT-300M",
        "aliases": (),
    },
    "laion-2b": {
        "name": "LAION-2B-en",
        "aliases": (
            "laion-2b-en-2-3b-pairs",
            "laion-2b-english-subset-of-laion-5b-vi",
        ),
    },
    "laion-aesthetic": {
        "name": "LAION-Aesthetic",
        "aliases": (),
    },
    "stylized-imagenet": {
        "name": "Stylized-ImageNet",
        "aliases": (
            "stylized-imagenet-sin-only",
        ),
    },
    "things-eeg2": {
        "name": "THINGS EEG2",
        "aliases": (
            "things-eeg2-held-out-test-set",
            "things-eeg2-held-out-test-set-novel-obj",
        ),
    },
    "wit-400m": {
        "name": "WIT-400M (OpenAI WebImageText)",
        "aliases": (
            "wit-400m-openai",
            "wit-400m-openai-400m-image-text-pairs",
            "wit-400m-openai-s-private-webimagetext",
        ),
    },
}

# SPDX identifiers, https://spdx.org/licenses/
LICENSES = {
    "Apache-2.0": "Apache License 2.0",
    "BSD-2-Clause": "BSD 2-Clause License",
    "BSD-3-Clause": "BSD 3-Clause License",
    "CC-BY-4.0": "Creative Commons Attribution 4.0",
    "CC-BY-NC-4.0": "Creative Commons Attribution Non Commercial 4.0",
    "CC-BY-NC-SA-4.0": "Creative Commons Attribution Non Commercial Share Alike 4.0",
    "GPL-3.0-only": "GNU General Public License v3.0 only",
    "GPL-3.0-or-later": "GNU General Public License v3.0 or later",
    "MIT": "MIT License",
}
LICENSE_ALIASES = {
    "Apache 2.0": "Apache-2.0",
    "CC-BY-NC 4.0": "CC-BY-NC-4.0",
    "GNU GPL 3.0+": "GPL-3.0-or-later",
    "GNU GPL v3+": "GPL-3.0-or-later",
}


def canonical_dataset(value):
    """Return the listed dataset ID for an ID or alias, else None."""
    key = (value or "").strip().lower()
    for identifier, spec in DATASETS.items():
        if key == identifier or key in spec["aliases"]:
            return identifier
    return None


def canonical_license(value):
    """Return the SPDX ID for a listed ID or alias, else None."""
    value = (value or "").strip()
    return value if value in LICENSES else LICENSE_ALIASES.get(value)


def is_other(value):
    return value == OTHER or (value or "").startswith(OTHER + ":")


def vocabulary_warnings(document):
    """List unlisted dataset and license values; never raises for them."""
    warnings = []
    for model, entry in document["models"].items():
        for path in DATASET_PATHS:
            for item in get_path(entry, path) or []:
                identifier = item.get("identifier")
                if identifier in DATASETS or identifier == OTHER:
                    continue
                hint = canonical_dataset(identifier)
                warnings.append(
                    f"{model} {path}: dataset {identifier or item['name']!r} is not listed; "
                    + (f"use {hint!r}" if hint else "use a listed ID or 'other'")
                )
        license = get_path(entry, "/legal/license")
        if license is None or license in LICENSES or is_other(license):
            continue
        hint = canonical_license(license)
        warnings.append(
            f"{model} /legal/license: {license!r} is not an SPDX ID from the list; "
            + (f"use {hint!r}" if hint else "use a listed ID or 'other: <text>'")
        )
    return warnings
