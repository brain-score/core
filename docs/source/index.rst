Brain-Score
===========

Brain-Score is a collection of benchmarks and models:
benchmarks combine neural/behavioral data with a metric to score models on their alignment to experimental observations,
and models are evaluated as computational hypotheses of natural intelligence.

This repository implements core functionality including a plugin system to manage data assemblies and models,
as well as metrics to compare e.g. neural recordings or behavioral mesurements.
Data assemblies and model predictions are organized in BrainIO_.

.. _BrainIO: https://github.com/brain-score/brainio

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules/metrics
   modules/benchmarks
   modules/model_tutorial
   modules/benchmark_tutorial
   modules/api_reference
   modules/glossary
