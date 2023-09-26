# Greedy EDM-FDE

This repository contains code for Euclidean Distance Matrix-based Fault Detection and Exclusion (FDE) based on the paper "Detection and Exclusion of Multiple Faults using Euclidean Distance Matrices" by Derek Knowles and Grace Gao from the ION GNSS+ 2023 conference.

## Install Dependencies

Install ``gnss_lib_py`` either with ``pip install gnss_lib_py`` or
following the detailed [installation instructions](https://gnss-lib-py.readthedocs.io/en/latest/install.html).

## Run Instructions

ION GNSS+ presentation/paper figures can be replicated through the following:

1. Create the simulated data.
```
cd improved-edm-fde/
python3 simulated_data_creation.py
```
2. Run FDE across the data (takes on the order of hours based on compute).
```
python3 fde_simulated.py
```
3. Create presentation figures by editing the ``<results directory>`` and
``<number>`` in ``presentation_figures.py`` then running the file.
```
python3 presentation_figures.py
```


## Citation
If referencing EDM-based FDE in your work, please cite the following paper:
```
@inproceedings{Knowles2023,
author = {Knowles, Derek and Gao, Grace},
title = {{Detection and Exclusion of Multiple Faults using Euclidean Distance Matrices}},
booktitle = {Proceedings of the 36th International Technical Meeting of the Satellite Divison of the Institute of Navigation, ION GNSS + 2023},
publisher = {Institute of Navigation},
year = {2023}
}
```
