"""Test FDE through iterating datasets

"""

__authors__ = "D. Knowles"
__date__ = "24 Jul 2023"

import numpy as np
import gnss_lib_py as glp

from lib.dataset_iterators import SmartLocIterator
from lib.dataset_iterators import Android2022Iterator
from lib.dataset_iterators import Android2023Iterator

def main():
    """Iterate over Android 2022 dataset

    """

    # train_path_2023 = "/path/to/2023/dataset/train/directory/"
    train_path_2023 = "/home/derek/datasets/sdc2023/train/"
    android2023 = Android2023Iterator(train_path_2023)
    # overwrite run function with what you'd like to test
    android2023.run = test_function
    # iterate across dataset
    # android2023.single_run(["2020-12-10-22-17-us-ca-sjc-c", "mi8"])
    android2023.iterate()

    # train_path_2022 = "/path/to/2022/dataset/train/directory/"
    # android2022 = Android2022Iterator(train_path_2022)
    # # overwrite run function with what you'd like to test
    # android2022.run = test_function
    # # iterate across dataset
    # android2022.iterate()

    # train_path_smart_loc = "/path/to/smartLoc/directory/"
    # smartloc = SmartLocIterator(train_path_smart_loc)
    # # overwrite run function with what you'd like to test
    # smartloc.run = test_function
    # # iterate across dataset
    # smartloc.iterate()


def test_function(trace, derived, gt_data, raw):
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

    if isinstance(derived, (glp.AndroidDerived2022,glp.AndroidDerived2023)):
        row_name = "MultipathIndicator"
    elif isinstance(derived, glp.SmartLocRaw):
        row_name = "NLOS (0 == no, 1 == yes, 2 == No Information)"
    else:
        raise TypeError("unsupported derived data type")

    for val in (np.unique(derived[row_name])):
        print(val,":",(derived[row_name]==val).sum())

    print("solving all wls")
    wls_all = glp.solve_wls(derived)
    wls_all.rename({"lat_rx_wls_deg":"lat_rx_" + "all" + "_deg",
                    "lon_rx_wls_deg":"lon_rx_" + "all" + "_deg",
                    "alt_rx_wls_m":"alt_rx_" + "all" + "_m",
                    }, inplace=True)
    print("solving nonfaulty wls")
    wls_nonfaulty = glp.solve_wls(derived.where(row_name,0))
    wls_nonfaulty.rename({"lat_rx_wls_deg":"lat_rx_" + "nonfaulty" + "_deg",
                         "lon_rx_wls_deg":"lon_rx_" + "nonfaulty" + "_deg",
                         "alt_rx_wls_m":"alt_rx_" + "nonfaulty" + "_m",
                         }, inplace=True)
    print("plotting")
    fig = glp.plot_map(gt_data,wls_all,wls_nonfaulty)
    fig.show()
    # print(derived["row_name"].counts())

    # glp.evaluate_fde(derived,"edm",
    #                  fault_truth_row=row_name,
    #                  verbose=True)

if __name__ == "__main__":
    main()
