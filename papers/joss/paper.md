---
title: '$\varphi$-torch: spark up your physics'
authors:
  - name: Konstantin Karchev
    orcid: 0000-0001-9344-736X
    affiliation: 1
affiliations:
  - name: |
      Theoretical and Scientific Data Science, 
      Scuola Internazionale Superiore di Studi Avanzati (SISSA)
    index: 1
date: 26 April 2642 
bibliography: paper.bib
---

# Summary

The last decade has seen tremendous progress in machine learning, both on the software and hardware sides, underpinned largely by two technologies: automatic differentiation through arbitrary computational graphs and effortless massively parallel computation on graphics processing units (GPUs).

# Statement of need

$\varphi$-torch (`phytorch`) is a collection of utilities that enable writing (astro)physics simulators using PyTorch as backend.
To this end, $\varphi$-torch provides, on one hand, routines for evaluating special mathematical functions, ~~integration~~, interpolation, and root-finding, replete with autograd and an implementation for GPUs owing to bespoke C++/CUDA PyTorch extensions.
Another major part of $\varphi$-torch is a system for representing and manipulating *quantities* carrying unit information, including (fundamental or less so) physical constants. All built-in PyTorch mathematical operations are supported with unit propagation and consistency checks.[^1]
Lastly, the `cosmology` submodule allows calculating cosmographic distances and other quantities within various popular cosmological models, all fully auto-differentiable, GPU-accelerated, and propertly unit-ised.[^2]

[^1]: And those that don't make sense in the context of units, like transcendental functions, are disallowed.
[^2]: A historical note: the cosmology module was the original seed of $\varphi$-torch, out of which grew the necessity for a library of special functions, root-finding routines, and convenient unit conversions.

# Related and similar software

$\varphi$-torch can be seen to a certain extent as a PyTorch port of certain features of SciPy [@scipy] and AstroPy [@astropy:2013,@astropy:2018] to the PyTorch "backend". Namely, $\varphi$-torch's special functions aim to follow the nomenclature of `scipy.special` while the quantity- and cosmology-related functionality is inspired by AstroPy. Moreover, classes in `phytorch.cosmology` allow converting to/from their `astropy.cosmology` counterparts.

PyTorch [@pytorch] was chosen as a backend over other autograd- and GPU-enabled Python frameworks, like JAX [@jax] and Tensorflow [@tensorflow], due to its ease of extensibility: in terms of adding computational routines written in C++/CUDA and integrating those in the autograd engine, as well as extending native tensor data-structures to carry unit information and participate seamlessly in unitful computations.[^3]

[^3]: Last but not least is the ease of deployment of PyTorch compared with other frameworks, in the experience of this author.

# Research enabled by $\varphi$-torch

$\varphi$-torch was used in @Karchev-sicret for cosmographic distance calculations, which were described in greater detail in @Karchev-ptcosmo. @Karchev-sidereal made additional use of $\varphi$-torch's spline interpolation functionality.

# Acknowledgements

I acknowledge the encouragement and support of Roberto Trotta and Christoph Weniger.

# References