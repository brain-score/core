[![Build Status](https://travis-ci.com/brain-score/brain-score_core.svg?token=??&branch=main)](https://travis-ci.com/brain-score/brain-score_core)
[![Documentation Status](https://readthedocs.org/projects/brain-score_core/badge/?version=latest)](https://brain-score_core.readthedocs.io/en/latest/?badge=latest)

Brain-Score is a platform to evaluate computational models of brain function 
on their match to brain measurements in primate vision. 
The intent of Brain-Score is to adopt many (ideally all) the experimental benchmarks in the field
for the purpose of model testing, falsification, and comparison.
To that end, Brain-Score operationalizes experimental data into quantitative benchmarks 
that any model candidate following the `BrainModel` interface can be scored on.

See the [Documentation](https://brain-score_core.readthedocs.io) for more details.

Brain-Score is made by and for the community. 
To contribute, please [send in a pull request](https://github.com/brain-score/brain-score_core/pulls).


## License

MIT license


## References

If you use Brain-Score in your work, please cite 
["Brain-Score: Which Artificial Neural Network for Object Recognition is most Brain-Like?"](https://www.biorxiv.org/content/10.1101/407007v2) (technical) and 
["Integrative Benchmarking to Advance Neurally Mechanistic Models of Human Intelligence"](https://www.cell.com/neuron/fulltext/S0896-6273(20)30605-X) (perspective) 
as well as the respective benchmark sources.

```bibtex
@article{SchrimpfKubilius2018BrainScore,
  title={Brain-Score: Which Artificial Neural Network for Object Recognition is most Brain-Like?},
  author={Martin Schrimpf and Jonas Kubilius and Ha Hong and Najib J. Majaj and Rishi Rajalingham and Elias B. Issa and Kohitij Kar and Pouya Bashivan and Jonathan Prescott-Roy and Franziska Geiger and Kailyn Schmidt and Daniel L. K. Yamins and James J. DiCarlo},
  journal={bioRxiv preprint},
  year={2018},
  url={https://www.biorxiv.org/content/10.1101/407007}
}

@article{Schrimpf2020integrative,
  title={Integrative Benchmarking to Advance Neurally Mechanistic Models of Human Intelligence},
  author={Schrimpf, Martin and Kubilius, Jonas and Lee, Michael J and Murty, N Apurva Ratan and Ajemian, Robert and DiCarlo, James J},
  journal={Neuron},
  year={2020},
  url={https://www.cell.com/neuron/fulltext/S0896-6273(20)30605-X}
}
```
