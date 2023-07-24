# Improved EDM-FDE

This repository contains code for Euclidean Distance Matrix-based Fault Detection and Exclusion (FDE) based on the paper "Detection and Exclusion of Multiple Faults using Euclidean Distance Matrices" by Derek Knowles and Grace Gao from the ION GNSS+ 2023 conference.

## Install

- instructions for glp install
- instructions for data install if needed

## Run Instructions

EDM-based FDE algorithm is implemented in

ION GNSS+ presentation/paper figures can be replicated from previously logged data with:

[TU Chemnitz](https://www.tu-chemnitz.de/projekt/smartLoc/gnss_dataset.html.en#Datasets) results can be replicated (up to the random initialization) with `python main_chemnitz.py`

Results from the google android dataset can be replciated by downloading the google data (see above), adding all traces and phone types to the trace list at the end of the [main_google.py](https://github.com/betaBison/edm-fde/blob/main/main_google.py) file, and running `python main_google.py`

## Citation
If referencing EDM-based FDE in your work, please cite the following paper:
```
@inproceedings{Knowles2023,
author = {Knowles, Derek and Gao, Grace},
title = {{Detection and Exclusion of Multiple Faults using Euclidean Distance Matrices}},
booktitle = {Proceedings of the 34th International Technical Meeting of the Satellite Divison of the Institute of Navigation, ION GNSS + 2023},
publisher = {Institute of Navigation},
year = {2023}
}
```
