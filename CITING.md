The AutoGluon developers and community are committed to open source, and that extends to our research.
Below you can find a list of our published work relating to AutoGluon along with guidelines for citing our work.

## Scientific Publications
- [AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data](https://arxiv.org/pdf/2003.06505.pdf) (*Arxiv*, 2020) ([BibTeX](#general-usage--autogluontabular))
- [Fast, Accurate, and Simple Models for Tabular Data via Augmented Distillation](https://proceedings.neurips.cc/paper/2020/hash/62d75fb2e3075506e8837d8f55021ab1-Abstract.html) (*NeurIPS*, 2020) ([BibTeX](#tabular-distillation))
- [Benchmarking Multimodal AutoML for Tabular Data with Text Fields](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/9bf31c7ff062936a96d3c8bd1f8f2ff3-Paper-round2.pdf) (*NeurIPS*, 2021) ([BibTeX](#autogluonmultimodal))
- [XTab: Cross-table Pretraining for Tabular Transformers](https://proceedings.mlr.press/v202/zhu23k/zhu23k.pdf) (*ICML*, 2023)
- [AutoGluon-TimeSeries: AutoML for Probabilistic Time Series Forecasting](https://arxiv.org/abs/2308.05566) (*AutoML Conf*, 2023) ([BibTeX](#autogluontimeseries))
- [TabRepo: A Large Scale Repository of Tabular Model Evaluations and its AutoML Applications](https://arxiv.org/pdf/2311.02971.pdf) (*AutoML Conf*, 2024)
- [AutoGluon-Multimodal (AutoMM): Supercharging Multimodal AutoML with Foundation Models](https://arxiv.org/pdf/2404.16233) (*AutoML Conf*, 2024) ([BibTeX](#autogluonmultimodal))

## Citing AutoGluon

Please cite the core AutoGluon paper (AutoGluon-Tabular) as well as any module specific paper that is relevant.

### General Usage & autogluon.tabular

If you use AutoGluon in a scientific publication, please cite the following paper (regardless of which module is being used):
- Erickson, Nick, et al. ["AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data."](https://arxiv.org/abs/2003.06505) arXiv preprint arXiv:2003.06505 (2020).

BibTeX entry:

```bibtex
@article{agtabular,
  title={AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data},
  author={Erickson, Nick and Mueller, Jonas and Shirkov, Alexander and Zhang, Hang and Larroy, Pedro and Li, Mu and Smola, Alexander},
  journal={arXiv preprint arXiv:2003.06505},
  year={2020}
}
```

### autogluon.multimodal

If you use AutoGluon's multimodal functionality in a scientific publication, please cite the following paper:

Zhiqiang, Tang, et al. ["AutoGluon-Multimodal (AutoMM): Supercharging Multimodal AutoML with Foundation Models"](https://arxiv.org/pdf/2404.16233), The International Conference on Automated Machine Learning (AutoML), 2024.

BibTeX entry:

```bibtex
@article{tang2024autogluon,
  title={AutoGluon-Multimodal (AutoMM): Supercharging Multimodal AutoML with Foundation Models},
  author={Tang, Zhiqiang and Fang, Haoyang and Zhou, Su and Yang, Taojiannan and Zhong, Zihan and Hu, Tony and Kirchhoff, Katrin and Karypis, George},
  journal={arXiv preprint arXiv:2404.16233},
  year={2024}
}
```

If you use AutoGluon's TextPredictor, please cite the following paper:

Shi, Xingjian, et al. ["Benchmarking Multimodal AutoML for Tabular Data with Text Fields."](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/9bf31c7ff062936a96d3c8bd1f8f2ff3-Paper-round2.pdf) Advances in Neural Information Processing Systems 35, Datasets and Benchmarks Track (2021).

```bibtex
@inproceedings{agmultimodaltext,
  title={Benchmarking Multimodal AutoML for Tabular Data with Text Fields},
  author={Shi, Xingjian and Mueller, Jonas and Erickson, Nick and Li, Mu and Smola, Alexander J},
  journal={Advances in Neural Information Processing Systems Datasets and Benchmarks Track},
  volume={35},
  year={2021}
}
```

### autogluon.timeseries

If you use AutoGluon's time series forecasting functionality in a scientific publication, please cite the following paper:
```bibtex
@inproceedings{agtimeseries,
  title={{AutoGluon-TimeSeries}: {AutoML} for Probabilistic Time Series Forecasting},
  author={Shchur, Oleksandr and Turkmen, Caner and Erickson, Nick and Shen, Huibin and Shirkov, Alexander and Hu, Tony and Wang, Yuyang},
  booktitle={International Conference on Automated Machine Learning},
  year={2023}
}
```

If you use the Chronos pretrained model, please cite:
```bibtex
@article{ansari2024chronos,
  title={Chronos: Learning the Language of Time Series},
  author={Ansari, Abdul Fatir and Stella, Lorenzo and Turkmen, Caner and Zhang, Xiyuan and Mercado, Pedro and Shen, Huibin and Shchur, Oleksandr and Rangapuram, Syama Sundar and Pineda Arango, Sebastian and Kapoor, Shubham and Zschiegner, Jasper and Maddix, Danielle C. and Mahoney, Michael W. and Torkkola, Kari and Gordon Wilson, Andrew and Bohlke-Schneider, Michael and Wang, Yuyang},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2024},
  url={https://openreview.net/forum?id=gerNCVqqtR}
}
```

### Tabular Distillation

If you are using AutoGluon Tabular's model distillation functionality, please cite the following paper:

Fakoor, Rasool, et al. ["Fast, Accurate, and Simple Models for Tabular Data via Augmented Distillation."](https://proceedings.neurips.cc/paper/2020/hash/62d75fb2e3075506e8837d8f55021ab1-Abstract.html) Advances in Neural Information Processing Systems 33 (2020).

BibTeX entry:

```bibtex
@article{agtabulardistill,
  title={Fast, Accurate, and Simple Models for Tabular Data via Augmented Distillation},
  author={Fakoor, Rasool and Mueller, Jonas W and Erickson, Nick and Chaudhari, Pratik and Smola, Alexander J},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
