import unittest
from copy import deepcopy
from unittest.mock import patch

from brainscore_core.metadata import editability, protected_changes
from brainscore_core.metadata.storage import legacy_projection, from_legacy
from brainscore_core.metadata.review import override


class ReviewRegressions(unittest.TestCase):
    def test_incomplete_or_capped_commit_list_cannot_establish_independence(self):
        from brainscore_core.metadata import MetadataError
        from brainscore_core.metadata.review import review_exclusions

        for count, commits in [(251, [{}] * 250), (3, [{}] * 2), (250, [{}] * 250)]:
            with self.assertRaises(MetadataError):
                review_exclusions(
                    {"commits": count, "user": {"login": "author"}}, commits
                )

    def test_new_v2_file_requires_override(self):
        from brainscore_core.metadata import dump, MetadataError
        from brainscore_core.metadata.review import check_pr

        pr = {
            "number": 1,
            "base": {"repo": {"full_name": "brain-score/vision"}},
            "head": {"sha": "head", "repo": {"full_name": "brain-score/vision"}},
        }
        document = {
            "schema_version": "2.0",
            "domain": "vision",
            "models": {"example": {}},
        }
        with (
            patch("brainscore_core.metadata.review.api", return_value=pr),
            patch(
                "brainscore_core.metadata.review.pages",
                return_value=[
                    {
                        "filename": "brainscore_vision/models/example/metadata.yaml",
                        "status": "added",
                    }
                ],
            ),
            patch(
                "brainscore_core.metadata.review.content", return_value=dump(document)
            ),
            patch("brainscore_core.metadata.review.override", return_value=False),
        ):
            with self.assertRaises(MetadataError):
                check_pr("brain-score/vision", 1, "vision", "brainscore_vision/models")

    def test_lossless_bootstrap_and_incremental_legacy_projection(self):
        legacy = {
            "total_parameter_count": 100,
            "trainable_layers": 12,
            "architecture": "DCNN",
        }
        initial = {
            "legacy": legacy,
            "model": {
                "parameter_count": 200,
                "trainable_layers": "7?",
                "architecture": {"family": "other"},
            },
        }
        self.assertEqual(legacy_projection(initial), legacy)
        self.assertEqual(legacy_projection({"legacy": legacy}), legacy)
        changed = deepcopy(initial)
        changed["model"]["parameter_count"] = 300
        self.assertEqual(
            legacy_projection(changed, initial), {"total_parameter_count": 300}
        )
        unrelated = deepcopy(changed)
        unrelated["training"] = {"loss": "New description"}
        self.assertEqual(legacy_projection(unrelated, changed), {})
        missing = deepcopy(changed)
        missing["model"]["parameter_count"] = None
        self.assertEqual(legacy_projection(missing, changed), {})

    def test_transformer_conversion_respects_domain(self):
        for domain, expected in [
            ("vision", "vision_transformer"),
            ("language", "transformer"),
            (None, "transformer"),
        ]:
            self.assertEqual(
                from_legacy({"architecture": "Transformer"}, domain)["model"][
                    "architecture"
                ]["family"],
                expected,
            )

    def test_empty_unknown_fields_editable_but_paper_sources_stay_protected(self):
        entry = {
            "sources": {"source": {"kind": "unreviewed"}},
            "assertions": [
                {"path": "/model", "status": "undocumented", "sources": ["source"]}
            ],
        }
        self.assertTrue(editability(entry, "/model/parameter_count")[0])
        entry["model"] = {"parameter_count": 10}
        self.assertFalse(editability(entry, "/model/parameter_count")[0])
        for kind in ["paper", "huggingface"]:
            entry["model"]["parameter_count"] = None
            entry["sources"]["source"]["kind"] = kind
            self.assertFalse(editability(entry, "/model/parameter_count")[0])

    def test_verified_value_change_requires_downgrade_or_override(self):
        before = {
            "model": {"parameter_count": 10},
            "sources": {"repo": {"kind": "other"}},
            "assertions": [
                {"path": "/model", "status": "verified", "sources": ["repo"]}
            ],
        }
        after = deepcopy(before)
        after["model"]["parameter_count"] = 20
        self.assertIn("/model/parameter_count", protected_changes(before, after))
        after["assertions"].append(
            {
                "path": "/model/parameter_count",
                "status": "probable",
                "sources": ["repo"],
            }
        )
        self.assertEqual(protected_changes(before, after), [])
        self.assertEqual(after["assertions"][0], before["assertions"][0])

    def test_commit_author_or_committer_cannot_supply_override(self):
        pr = {
            "number": 1,
            "user": {"login": "submitter"},
            "head": {"sha": "head"},
            "labels": [{"name": "metadata-source-override"}],
        }
        reviews = [
            {
                "user": {"login": "maintainer", "type": "User"},
                "state": "APPROVED",
                "commit_id": "head",
            }
        ]
        for role in ["author", "committer"]:
            with (
                patch(
                    "brainscore_core.metadata.review.pages",
                    side_effect=lambda path: (
                        [{role: {"login": "maintainer"}}]
                        if path.endswith("/commits")
                        else reviews
                    ),
                ),
                patch(
                    "brainscore_core.metadata.review.api",
                    return_value={"permission": "admin"},
                ),
            ):
                self.assertFalse(override(pr, "brain-score/vision"))
        with (
            patch(
                "brainscore_core.metadata.review.pages",
                side_effect=lambda path: [] if path.endswith("/commits") else reviews,
            ),
            patch(
                "brainscore_core.metadata.review.api",
                return_value={"permission": "admin"},
            ),
        ):
            self.assertTrue(override(pr, "brain-score/vision"))
