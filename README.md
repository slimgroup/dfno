[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6981516.svg)](https://doi.org/10.5281/zenodo.6981516)

# Distributed Fourier Neural Operators

Implements a distributed version of [Fourier Neural Operators](https://arxiv.org/pdf/2010.08895.pdf)
based off of the [code](https://github.com/zongyi-li/fourier_neural_operator) written by Zongyi Li, Shuhao Cao, and Jack Griffiths
using [DistDL](https://github.com/distdl/distdl).

## Installation


To install this package you can either clone it and install it in dvelopper mode:


```bash
git clone https://github.com/slimgroup/dfno.git
cd dfno
pip install -e .
```

or directly install it with pip as a standard pythn package `pip install git+https://github.com/slimgroup/dfno.git`

## Author

This package is written by Thomas Grady <tgrady@gatech.edu> at Georgia Institute of Technology


## Citation

If you like and use our package, please cite our preprint:

```
@techreport {grady2022SCtll,
	title = {Model-Parallel Fourier Neural Operators as Learned Surrogates for Large-Scale Parametric PDEs},
	number = {TR-CSE-2022-1},
	year = {2022},
	month = {04},
	keywords = {CCS, deep learning, Fourier neural operators, HPC, large-scale, Model Parallelism, Multiphase Flow, Operator Learning},
	url = {https://arxiv.org/pdf/2204.01205.pdf},
	software = {https://github.com/slimgroup/dfno},
	author = {Thomas J. Grady II and Rishi Khan and Mathias Louboutin and Ziyi Yin and Philipp A. Witte and Ranveer Chandra and Russell J. Hewett and Felix J. Herrmann}
}
```
