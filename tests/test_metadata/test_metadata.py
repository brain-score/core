import unittest
from copy import deepcopy
from brainscore_core.metadata import (
    load,
    dump,
    validate,
    MetadataError,
    protected_changes,
    editability,
)
from brainscore_core.metadata.storage import to_tables


def fixture():
    return {
        "schema_version": "2.0",
        "domain": "vision",
        "models": {
            "example": {
                "model": {"parameter_count": 123},
                "sources": {
                    "source": {"kind": "other", "url": "https://example.org/model"}
                },
                "assertions": [
                    {
                        "path": "/model/parameter_count",
                        "status": "probable",
                        "sources": ["source"],
                    }
                ],
            }
        },
    }


class ContractTests(unittest.TestCase):
    def test_roundtrip_and_tables(self):
        doc = fixture()
        self.assertEqual(load(dump(doc)), doc)
        self.assertEqual(to_tables(doc)["models"][0]["parameter_count"], 123)

    def test_rejects_wrong_types_duplicates_aliases_unknown_keys_and_domains(self):
        for content in (
            'schema_version: "2.0"\nschema_version: "2.0"',
            "a: &a [*a]",
            "!!python/object:os.system {}",
        ):
            with self.assertRaises(MetadataError):
                load(content)
        for value in (True, -1, float("nan"), "123"):
            doc = fixture()
            doc["models"]["example"]["model"]["parameter_count"] = value
            with self.assertRaises(MetadataError):
                validate(doc)
        with self.assertRaises(MetadataError):
            validate(fixture(), "language")
        doc = fixture()
        doc["models"]["example"]["oops"] = 1
        with self.assertRaises(MetadataError):
            validate(doc)

    def test_cannot_relabel_huggingface_as_other(self):
        doc = fixture()
        doc["models"]["example"]["sources"]["source"]["url"] = (
            "https://huggingface.co/example/model"
        )
        with self.assertRaises(MetadataError):
            validate(doc)

    def test_protected_values_and_sources_cannot_be_removed_to_unlock(self):
        before = fixture()["models"]["example"]
        before["sources"]["source"]["kind"] = "paper"
        after = deepcopy(before)
        after["model"]["parameter_count"] = 456
        after["sources"]["source"]["kind"] = "other"
        self.assertIn("/model/parameter_count", protected_changes(before, after))
        after = deepcopy(before)
        after["assertions"] = []
        self.assertIn("/model/parameter_count", protected_changes(before, after))

    def test_parent_assertions_protect_child_values(self):
        before = fixture()["models"]["example"]
        before["sources"]["source"]["kind"] = "huggingface"
        before["assertions"][0]["path"] = "/model"
        after = deepcopy(before)
        after["model"]["recurrent"] = True
        self.assertIn("/model/recurrent", protected_changes(before, after))

    def test_unknown_values_locked_empty_values_editable_and_verified_reviewed(self):
        before = {"model": {"parameter_count": 3}}
        self.assertFalse(editability(before, "/model/parameter_count")[0])
        self.assertTrue(editability(before, "/training/loss")[0])
        before = fixture()["models"]["example"]
        after = deepcopy(before)
        after["model"]["parameter_count"] = 456
        self.assertEqual(protected_changes(before, after), [])
        after["assertions"][0]["status"] = "verified"
        self.assertIn("/model/parameter_count", protected_changes(before, after))


class ReviewTests(unittest.TestCase):
    def test_schema_downgrade_and_renaming_outside_root_rejected(self):
        from unittest.mock import patch
        from brainscore_core.metadata.review import check_pr

        pr = {
            "base": {"repo": {"full_name": "brain-score/vision"}, "sha": "base"},
            "head": {"repo": {"full_name": "contributor/vision"}, "sha": "head"},
        }
        path = "brainscore_vision/models/example/metadata.yaml"
        for item in [
            {"filename": path, "status": "modified"},
            {
                "filename": "elsewhere/metadata.yaml",
                "previous_filename": path,
                "status": "renamed",
            },
        ]:
            with (
                patch("brainscore_core.metadata.review.api", return_value=pr),
                patch("brainscore_core.metadata.review.pages", return_value=[item]),
                patch(
                    "brainscore_core.metadata.review.content",
                    side_effect=["models: {}", dump(fixture())],
                ),
            ):
                with self.assertRaises(MetadataError):
                    check_pr(
                        "brain-score/vision", 10, "vision", "brainscore_vision/models"
                    )

    def test_invalid_source_url_and_legacy_values_are_validation_errors(self):
        for source in (
            {"kind": "other", "url": "https://["},
            {"kind": []},
            {"kind": "other", "url": "http://example.org"},
        ):
            doc = fixture()
            doc["models"]["example"]["sources"]["source"] = source
            with self.assertRaises(MetadataError):
                validate(doc)
        for legacy in (
            {"total_parameter_count": True},
            {"runnable": "false"},
            {"model_size_mb": float("inf")},
        ):
            doc = fixture()
            doc["models"]["example"]["legacy"] = legacy
            with self.assertRaises(MetadataError):
                validate(doc)


if __name__ == "__main__":
    unittest.main()
