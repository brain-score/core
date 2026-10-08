import unittest
from brainscore_core.metadata import validate, vocabulary_warnings


def document(datasets=(), license=None):
    entry = {"data": {"training_datasets": list(datasets)}}
    if license is not None:
        entry["legal"] = {"license": license}
    return {"schema_version": "2.0", "domain": "vision", "models": {"example": entry}}


class VocabularyTests(unittest.TestCase):
    def test_listed_values_and_other_do_not_warn(self):
        doc = document([{"identifier": "imagenet-1k", "name": "ImageNet-1k", "role": "training"},
                        {"identifier": "other", "name": "In-house video corpus", "role": "pretraining"}],
                       "other: weights license unconfirmed")
        self.assertEqual(vocabulary_warnings(validate(doc)), [])
        self.assertEqual(vocabulary_warnings(document(license="Apache-2.0")), [])

    def test_unlisted_values_warn_without_rejecting(self):
        doc = validate(document([{"identifier": "imagenet-ilsvrc2012", "name": "ImageNet (ILSVRC2012)",
                                  "role": "training"},
                                 {"name": "Private corpus", "role": "training"}], "Apache 2.0"))
        warnings = vocabulary_warnings(doc)
        self.assertEqual(len(warnings), 3)
        self.assertIn("use 'imagenet-1k'", warnings[0])
        self.assertIn("'Private corpus' is not listed", warnings[1])
        self.assertIn("use 'Apache-2.0'", warnings[2])


if __name__ == "__main__":
    unittest.main()
