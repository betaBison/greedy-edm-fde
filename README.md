# Greedy EDM FDE

This repository contains code for Greedy Euclidean Distance Matrix-based Fault Detection and Exclusion (FDE) based on the paper "Greedy Detection and Exclusion of Multiple Faults using Euclidean Distance Matrices" by Derek Knowles and Grace Gao from the ION GNSS+ 2023 conference.

Tutorials on how to run EDM FDE using the ``gnss_lib_py`` library can be
found on the ``gnss_lib_py`` [documentation website](https://gnss-lib-py.readthedocs.io/en/latest/tutorials/algorithms/tutorials_fde_notebook.html).

## Install Dependencies

Install all dependencies with ``pip install -r requirements.txt``  

or  

Install ``gnss_lib_py`` either with ``pip install gnss_lib_py`` or
following the detailed [installation instructions](https://gnss-lib-py.readthedocs.io/en/latest/install.html). And install the other needed Python packages in
the ``requirements.txt`` file.

## Run Instructions

ION GNSS+ presentation/paper figures can be replicated through the following:

1. Create the simulated data.
```
cd greedy-edm-fde/
python3 simulated_data_creation.py
```
2. Run FDE across the simulated data (takes on the order of hours based on compute).
```
python3 fde_simulated.py
```
3. Run FDE across the real-world data (takes on the order of hours based on compute)
after updating the ``train_path_2023`` variable in the ``fde_gsdc.py`` with your
local path to the train directory of the [Google Smartphone Decimeter
Challenge 2023](https://www.kaggle.com/competitions/smartphone-decimeter-2023) dataset.
```
python3 fde_gsdc.py
```
4. Create presentation figures by editing the ``<simulated results directory>``,
``<simulated #>``, ``<gsdc results directory>``, and ``<gsdc #>`` variables in ``presentation_figures.py`` then running the file:
```
python3 presentation_figures.py
```


## Citation
If referencing greedy EDM FDE in your work, please cite the following paper:
```
@inproceedings{Knowles2023,
author = {Knowles, Derek and Gao, Grace},
title = {{Detection and Exclusion of Multiple Faults using Euclidean Distance Matrices}},
booktitle = {Proceedings of the 36th International Technical Meeting of the Satellite Divison of the Institute of Navigation, ION GNSS + 2023},
publisher = {Institute of Navigation},
year = {2023}
}
```
