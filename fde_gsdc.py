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
    # android2023.iterate()

    # first trace
    # android2023.single_run(["2020-12-10-22-17-us-ca-sjc-c", "mi8"])

    # negative improvement
    # android2023.single_run(["2021-07-19-20-49-us-ca-mtv-a","sm-g988b"])
    # android2023.single_run(["2021-12-08-20-28-us-ca-lax-c","pixel6pro"])
    # android2023.single_run(["2022-11-15-00-53-us-ca-mtv-a","pixel7pro"])
    # android2023.single_run(["2023-03-08-21-34-us-ca-mtv-u","pixel6pro"])
    # android2023.single_run(["2023-03-08-21-34-us-ca-mtv-u","pixel7pro"])
    # android2023.single_run(["2023-05-09-21-32-us-ca-mtv-pe1","pixel7pro"])
    # android2023.single_run(["2023-05-25-19-10-us-ca-sjc-be2","pixel7pro"])
    # android2023.single_run(["2023-05-25-20-11-us-ca-sjc-he2","pixel7pro"])
    # android2023.single_run(["2023-09-05-23-07-us-ca-routen","pixel7pro"])
    # android2023.single_run(["2023-09-06-00-01-us-ca-routen","pixel6pro"])
    # android2023.single_run(["2023-09-07-19-33-us-ca", "pixel6pro"])
    # android2023.single_run(["2023-09-06-22-49-us-ca-routebb1", "pixel7pro"])
    # android2023.single_run(["2023-09-06-18-47-us-ca","pixel6pro"])

    # nan values
    # android2023.single_run(["2021-04-02-20-43-us-ca-mtv-f","mi8"])
    # android2023.single_run(["2021-07-19-20-49-us-ca-mtv-a","mi8"])
    # android2023.single_run(["2021-08-04-20-40-us-ca-sjc-c","sm-g988b"])
    # android2023.single_run(["2021-08-24-20-32-us-ca-mtv-h","sm-g988b"])
    # android2023.single_run(["2022-01-11-18-48-us-ca-mtv-n","pixel6pro"])
    # android2023.single_run(["2022-01-26-20-02-us-ca-mtv-pe1","mi8"])
    # android2023.single_run(["2022-01-26-20-02-us-ca-mtv-pe1","sm-g988b"])
    # android2023.single_run(["2022-04-01-18-22-us-ca-lax-t","mi8"])
    # android2023.single_run(["2022-04-01-18-22-us-ca-lax-t","pixel6pro"])
    # android2023.single_run(["2022-05-13-20-57-us-ca-mtv-pe1","pixel6pro"])
    # android2023.single_run(["2023-09-06-00-01-us-ca-routen","sm-g955f"])
    # android2023.single_run(["2023-09-07-22-47-us-ca-routebc2", "pixel6pro"])


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

def mean_50_95_horizontal(state_estimate, ground_truth):
    # for x,y,z in glp.loop_time(state_estimate,"gps_millis"):
    #     print(x,y,z)
    glp.interpolate(state_estimate,"gps_millis",
                                     ["x_rx_wls_m","y_rx_wls_m",
                                      "z_rx_wls_m","b_rx_wls_m"],
                                      inplace=True)

    percentile_50 = glp.accuracy_statistics(state_estimate,
                                            ground_truth,
                                            est_type="pos",
                                            statistic="percentile",
                                            direction="horizontal",
                                            percentile=50.)
    percentile_95 = glp.accuracy_statistics(state_estimate,
                                            ground_truth,
                                            est_type="pos",
                                            statistic="percentile",
                                            direction="horizontal",
                                            percentile=95.)

    avg_error = (percentile_50["pos_rx_percentile_50.0_horiz_m"] \
              + percentile_95["pos_rx_percentile_95.0_horiz_m"])/2.
    return avg_error

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

    # derived = derived.copy(cols=list(range(0,500)))

    for val in (np.unique(derived[row_name])):
        print(val,":",(derived[row_name]==val).sum())

    print("solving all wls")
    wls_all = glp.solve_wls(derived)

    wls_all.rename({"lat_rx_wls_deg":"lat_rx_" + "all" + "_deg",
                    "lon_rx_wls_deg":"lon_rx_" + "all" + "_deg",
                    "alt_rx_wls_m":"alt_rx_" + "all" + "_m",
                    }, inplace=True)
    print("solving nonfaulty wls")

    stat_all = mean_50_95_horizontal(wls_all, gt_data)
    print("stat_all:",stat_all)

    wls_nonfaulty = glp.solve_wls(derived.where(row_name,0))
    wls_nonfaulty.rename({"lat_rx_wls_deg":"lat_rx_" + "nonfaulty" + "_deg",
                         "lon_rx_wls_deg":"lon_rx_" + "nonfaulty" + "_deg",
                         "alt_rx_wls_m":"alt_rx_" + "nonfaulty" + "_m",
                         }, inplace=True)
    stat_nonfaulty = mean_50_95_horizontal(wls_nonfaulty, gt_data)
    print("stat_nonfaulty:",stat_nonfaulty)

    print("nonfault is better by:",stat_all - stat_nonfaulty)

    # print("plotting")
    # fig = glp.plot_map(gt_data,wls_all,wls_nonfaulty)
    # fig.show()

    # glp.evaluate_fde(derived,"edm",
    #                  fault_truth_row=row_name,
    #                  verbose=True)

if __name__ == "__main__":
    main()
