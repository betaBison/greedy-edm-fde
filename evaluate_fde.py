"""Test FDE through iterating datasets

"""

__authors__ = "D. Knowles"
__date__ = "24 Jul 2023"

import gnss_lib_py as glp

from lib.dataset_iterators import Android2022Iterator
from lib.dataset_iterators import SmartLocIterator

def main():
    """Iterate over Android 2022 dataset

    """

    train_path_2022 = "/path/to/2022/dataset/train/directory/"
    android2022 = Android2022Iterator(train_path_2022)
    # overwrite run function with what you'd like to test
    android2022.run = test_function
    # iterate across dataset
    android2022.iterate()

    # train_path_smart_loc = "/path/to/smartLoc/directory/"
    # smartloc = SmartLocIterator(train_path_smart_loc)
    # # overwrite run function with what you'd like to test
    # smartloc.run = test_function
    # # iterate across dataset
    # smartloc.iterate()


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
    elif isinstance(derived, glp.SmartLocRaw):
        row_name = "NLOS (0 == no, 1 == yes, 2 == No Information)"
    else:
        raise TypeError("unsupported derived data type")

    glp.evaluate_fde(derived,"edm",
                     fault_truth_row=row_name,
                     verbose=True)

if __name__ == "__main__":
    main()
