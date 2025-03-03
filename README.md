                        *///////////////.
                     //////////////////////*
                   */////////////////////////.
                  ////////////// */////////////
                  /////////*          /////////
                 //////   /////   ////,   /////
                  ////////     ///    /////////
                  /////   /////  ./////   ////*
                   ,////                 ////
                     *////             ////.
                         ///////*///////

    ░█████╗░██╗░░░██╗████████╗██████╗░░█████╗░███╗░░██╗██╗░░██╗
    ██╔══██╗██║░░░██║╚══██╔══╝██╔══██╗██╔══██╗████╗░██║██║░██╔╝
    ██║░░██║██║░░░██║░░░██║░░░██████╔╝███████║██╔██╗██║█████═╝░
    ██║░░██║██║░░░██║░░░██║░░░██╔══██╗██╔══██║██║╚████║██╔═██╗░
    ╚█████╔╝╚██████╔╝░░░██║░░░██║░░██║██║░░██║██║░╚███║██║░╚██╗
    ░╚════╝░░╚═════╝░░░░╚═╝░░░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░╚══╝╚═╝░░╚═╝

[![CI - package](https://github.com/outbrain/outrank/actions/workflows/python-package.yml/badge.svg)](https://github.com/outbrain/outrank/actions/workflows/python-package.yml) [![CI - benchmark](https://github.com/outbrain/outrank/actions/workflows/benchmarks.yml/badge.svg)](https://github.com/outbrain/outrank/actions/workflows/benchmarks.yml) [![CI - selftest](https://github.com/outbrain/outrank/actions/workflows/selftest.yml/badge.svg)](https://github.com/outbrain/outrank/actions/workflows/selftest.yml) [![Unit tests](https://github.com/outbrain/outrank/actions/workflows/python-unit.yml/badge.svg)](https://github.com/outbrain/outrank/actions/workflows/python-unit.yml)

# TLDR
> The design of modern recommender systems relies on understanding which parts of the feature space are relevant for solving a given recommendation task. However, real-world data sets in this domain are often characterized by their large size, sparsity, and noise, making it challenging to identify meaningful signals. Feature ranking represents an efficient branch of algorithms that can help address these challenges by identifying the most informative features and facilitating the automated search for more compact and better-performing models (AutoML). We introduce OutRank, a system for versatile feature ranking and data quality-related anomaly detection. OutRank was built with categorical data in mind, utilizing a variant of mutual information that is normalized with regard to the noise produced by features of the same cardinality. We further extend the similarity measure by incorporating information on feature similarity and combined relevance.

# Getting started
Minimal examples and an interface to explore OutRank's functionality are available as [the docs](https://outbrain-inc.github.io/outrank/outrank.html).

# Contributing
1. Make sure the functionality is not already implemented!
2. Decide where the functionality would fit best (is it an algorithm? A parser?)
3. Open a PR with the implementation

# Bugs and other reports
Feel free to open a PR that contains:
1. Issue overview
2. Minimal example useful for replicating the issue on our end
3. Possible solution

# Citing this work
If you use or build on top of OutRank, feel free to cite:

```
@inproceedings{10.1145/3604915.3610636,
author = {Skrlj, Blaz and Mramor, Bla\v{z}},
title = {OutRank: Speeding up AutoML-Based Model Search for Large Sparse Data Sets with Cardinality-Aware Feature Ranking},
year = {2023},
isbn = {9798400702419},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3604915.3610636},
doi = {10.1145/3604915.3610636},
abstract = {The design of modern recommender systems relies on understanding which parts of the feature space are relevant for solving a given recommendation task. However, real-world data sets in this domain are often characterized by their large size, sparsity, and noise, making it challenging to identify meaningful signals. Feature ranking represents an efficient branch of algorithms that can help address these challenges by identifying the most informative features and facilitating the automated search for more compact and better-performing models (AutoML). We introduce OutRank, a system for versatile feature ranking and data quality-related anomaly detection. OutRank was built with categorical data in mind, utilizing a variant of mutual information that is normalized with regard to the noise produced by features of the same cardinality. We further extend the similarity measure by incorporating information on feature similarity and combined relevance. The proposed approach’s feasibility is demonstrated by speeding up the state-of-the-art AutoML system on a synthetic data set with no performance loss. Furthermore, we considered a real-life click-through-rate prediction data set where it outperformed strong baselines such as random forest-based approaches. The proposed approach enables exploration of up to 300\% larger feature spaces compared to AutoML-only approaches, enabling faster search for better models on off-the-shelf hardware.},
booktitle = {Proceedings of the 17th ACM Conference on Recommender Systems},
pages = {1078–1083},
numpages = {6},
keywords = {Feature ranking, massive data sets, AutoML, recommender systems},
location = {Singapore, Singapore},
series = {RecSys '23}
}

@article{skrlj2023DrifterEO,
  title={Drifter: Efficient Online Feature Monitoring for Improved Data Integrity in Large-Scale Recommendation Systems},
  author={Bla{\vz} {\vS}krlj and Nir Ki-Tov and Lee Edelist and Natalia Silberstein and Hila Weisman-Zohar and Bla{\vz} Mramor and Davorin Kopic and Naama Ziporin},
  journal={ArXiv},
  year={2023},
  volume={abs/2309.08617},
  url={https://api.semanticscholar.org/CorpusID:262045065}
}
```
