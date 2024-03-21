"""Test FDE through iterating datasets

"""

__authors__ = "D. Knowles"
__date__ = "24 Jul 2023"

import os
import time
from multiprocessing import Process

import numpy as np
import gnss_lib_py as glp

from lib.dataset_iterators import Android2023Iterator

# methods and thresholds to test
METHODS = {
            "edm" : [0,0.5,0.6,0.7,1.0],
            "residual" : [0,100,1000,10000],
           }
# number of processes to run at the same time
PROCESS_PARALLEL = 3


def main():
    """Iterate over Google Smartphone Decimeter 2023 dataset

    """

    # train_path_2023 = "/path/to/2023/dataset/train/directory/"
    train_path_2023 = "/home/derek/datasets/sdc2023/train/"
    android2023 = Android2023Iterator(train_path_2023)
    # overwrite run function with what you'd like to test
    android2023.run = test_function
    # iterate across dataset
    trace_list = android2023.iterate(return_traces = True)
    trace_list = trace_list[:3]
    print(trace_list)

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



    time_start = time.time()
    data_dir = os.path.join(os.path.dirname(
               os.path.realpath(__file__)),"data","simulated")
    processes = [Process(target=android2023.single_run,
                         args=(trace,)) \
                         for trace in trace_list]


    for ii in range(int(np.ceil(len(processes)/PROCESS_PARALLEL))):
        process_group = processes[ii*PROCESS_PARALLEL:(ii+1)*PROCESS_PARALLEL]

        for process in process_group:
            process.start()

        for process in process_group:
            process.join()

        print('Done')
        print(ii,"finished in:",
              round((time.time()-time_start)/60,2),"minutes")

    results = glp.NavData()
    results_dir = os.path.join(os.getcwd(),"results",glp.TIMESTAMP)
    for navdata_file in sorted(os.listdir(results_dir)):
        if navdata_file[:9] == "location_":
            results = glp.concat(results,glp.NavData(csv_path=os.path.join(results_dir,
                                                         navdata_file)))

    results.to_csv(prefix="fde_"+str(len(results)))

    state_results_full = glp.NavData()
    for navdata_file in sorted(os.listdir(results_dir)):
        if navdata_file[:10] == "loc_state_":
            state_results_full = glp.concat(state_results_full,glp.NavData(csv_path=os.path.join(results_dir,
                                                         navdata_file)))

    state_results_full.to_csv(prefix="fde_state_"+str(len(results)))

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

    state_results = glp.NavData()
    results = glp.NavData()

    if isinstance(derived, (glp.AndroidDerived2022,glp.AndroidDerived2023)):
        fault_row_name = "MultipathIndicator"
    elif isinstance(derived, glp.SmartLocRaw):
        fault_row_name = "NLOS (0 == no, 1 == yes, 2 == No Information)"
    else:
        raise TypeError("unsupported derived data type")

    # derived = derived.copy(cols=list(range(0,500)))

    print("solving all wls")
    wls_all = glp.solve_wls(derived)

    wls_all.rename({"lat_rx_wls_deg":"lat_rx_" + "all" + "_deg",
                    "lon_rx_wls_deg":"lon_rx_" + "all" + "_deg",
                    "alt_rx_wls_m":"alt_rx_" + "all" + "_m",
                    }, inplace=True)
    print("solving nonfaulty wls")

    stat_all = mean_50_95_horizontal(wls_all, gt_data)
    print("stat_all:",stat_all)

    wls_nonfaulty = glp.solve_wls(derived.where(fault_row_name,0))
    wls_nonfaulty.rename({"lat_rx_wls_deg":"lat_rx_" + "nonfaulty" + "_deg",
                         "lon_rx_wls_deg":"lon_rx_" + "nonfaulty" + "_deg",
                         "alt_rx_wls_m":"alt_rx_" + "nonfaulty" + "_m",
                         }, inplace=True)
    stat_nonfaulty = mean_50_95_horizontal(wls_nonfaulty, gt_data)
    print("stat_nonfaulty:",stat_nonfaulty)
    print("nonfault is better by:",stat_all - stat_nonfaulty)

    state_results["trace"] = np.array([trace[0],trace[0]])
    state_results["phone"] = np.array([trace[1],trace[1]])
    state_results["method"] = np.array(["all","gt_nonfaulty"])
    state_results["threshold"] = np.array([np.nan,np.nan])
    state_results["horizontal_50_95"] = np.array([stat_all,stat_nonfaulty])

    # iterate over methods
    for method, thresholds in METHODS.items():
        print(trace[0],trace[1],"method:",method)
        for threshold in thresholds:
            print(trace[0],trace[1],"threshold:",threshold)

            input_navdata = derived.copy()
            metrics, navdata = glp.evaluate_fde(input_navdata,
                                                method=method,
                                                threshold=threshold,
                                                fault_truth_row=fault_row_name,
                                                # max_faults=num_faults,
                                                verbose=False,
                                                time_fde=True,)

            metrics_navdata = glp.NavData()
            metrics_navdata["trace"] = np.array(trace[0])
            metrics_navdata["phone"] = np.array(trace[1])
            metrics_navdata["threshold"] = threshold
            for k,v in metrics.items():
                metrics_navdata[k] = np.array([v])

            if threshold == 0:
                # str(np.round(0,4)).zfill(4) is '0000', but
                # str(np.round(0.0,4)).zfill(4) is '0.00', so
                threshold = int(0)

            navdata_prefix = [method,trace[0],trace[1],
                              str(np.round(threshold,4)).zfill(4)]
            navdata_prefix = "_".join(navdata_prefix).replace(".","")
            navdata.to_csv(prefix=navdata_prefix)

            results = glp.concat(results,metrics_navdata)

            # compute state results
            wls_method = glp.solve_wls(navdata.where("fault_" + method, 0))
            stat_method = mean_50_95_horizontal(wls_method, gt_data)

            state_results_temp = glp.NavData()
            state_results_temp["trace"] = np.array(trace[0])
            state_results_temp["phone"] = np.array(trace[1])
            state_results_temp["method"] = np.array(method)
            state_results_temp["threshold"] = threshold
            state_results_temp["horizontal_50_95"] = stat_method
            state_results_temp.to_csv(prefix="state_"+navdata_prefix)
            state_results = glp.concat(state_results,state_results_temp)

    results.to_csv(prefix="location_"+trace[0]+"_"+trace[1]+"_"+str(len(results)))
    state_results.to_csv(prefix="loc_state_"+trace[0]+"_"+trace[1]+"_"+str(len(results)))

    # print("plotting")
    # fig = glp.plot_map(gt_data,wls_all,wls_nonfaulty)
    # fig.show()


if __name__ == "__main__":
    main()
