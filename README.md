[![Build Status](https://app.travis-ci.com/brain-score/core.svg?token=vqt7d2yhhpLGwHsiTZvT&branch=main)](https://app.travis-ci.com/brain-score/core)
[![Documentation Status](https://readthedocs.org/projects/brain-score-core/badge/?version=latest)](https://brain-score-core.readthedocs.io/en/latest/?badge=latest)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md) 

Brain-Score is a platform to evaluate computational models of mind and brain function on their match to behavioral and
neural measurements in domains such as vision and language. The intent of Brain-Score is to adopt many (ideally all) the
experimental benchmarks in the field for the purpose of model testing, falsification, and comparison. To that end,
Brain-Score turns experimental data into quantitative benchmarks that evaluate
subjects with compatible inputs, outputs and methods.

> **UMI integration:** `Subject` defines session interaction through
> `interact(session)`. `BrainScoreModel` implements it with extraction and task
> helpers, including `process()`. Existing vision `BrainModel` and language
> `ArtificialSubject` plugins remain supported through adapters. See
> [the interface guide](docs/UMI_MIGRATION.md).

See the [Documentation](https://brain-score-core.readthedocs.io) for more details.

Brain-Score is made by and for the community. To contribute,
please [send in a pull request](https://github.com/brain-score/core/pulls).

## License

MIT license
