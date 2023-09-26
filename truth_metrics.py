"""Test FDE through iterating datasets

"""

__authors__ = "D. Knowles"
__date__ = "11 Aug 2023"

import numpy as np
import gnss_lib_py as glp
import matplotlib.pyplot as plt

from lib.dataset_iterators import Android2022Iterator
from lib.dataset_iterators import SmartLocIterator

def main():
    """Iterate over Android 2022 dataset

    """

    # train_path_2022 = "/path/to/2022/dataset/train/directory/"
    # android2022 = Android2022Iterator(train_path_2022)
    # # overwrite run function with what you'd like to test
    # android2022.run = test_function
    # # iterate across dataset
    # android2022.iterate()

    train_path_smart_loc = "/home/derek/datasets/smart_loc/"
    smartloc = SmartLocIterator(train_path_smart_loc)
    # overwrite run function with what you'd like to test
    smartloc.run = test_function
    # iterate across dataset
    smartloc.iterate()

def test_function(derived, gt_data, trace):
    """Test function.

    Parameters
    ----------
    derived : gnss_lib_py.parsers.android.AndroidDerived*
        Derived data.
    ground_truth : gnss_lib_py.parsers.android.AndroidGroundTruth*
        Ground truth data.
    trace : list
        Name of data/place and then name of phone.

    """

    if isinstance(derived, glp.AndroidDerived2022):
        row_name = "MultipathIndicator"
        fig = glp.solve_residuals(derived,gt_data)
    elif "Tracking status (trkStat) []" in derived.rows:
        pass

    glp.plot_metric_by_constellation(derived,"residuals_m",
                                           prefix="minus_raw_1E3_"+trace[0],
                                           save=True)
    pr_residuals = derived.where("gnss_id","gps")["residuals_m"]

    plt.figure()
    plt.hist(pr_residuals, bins=1000)

    plt.figure()
    colors = np.where(derived["fault_gt"]==1,"r","b")
    print(colors)
    print(type(colors))
    print(np.unique(colors))
    plt.scatter(derived["residuals_m"],derived["cn0_dbhz"],c=colors,s=1)

    plt.show()
    # glp.evaluate_fde(derived,"fault_gt",
    #                  fault_truth_row=row_name,
    #                  verbose=True)

if __name__ == "__main__":
    main()
