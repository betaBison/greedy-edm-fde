"""Testing EDM FDE on simulated data

"""

__authors__ = "D. Knowles"
__date__ = "15 Aug 2023"

import os
import time
from multiprocessing import Process

import numpy as np
import gnss_lib_py as glp
from gnss_lib_py.utils.file_operations import TIMESTAMP

np.random.seed(314)

# methods and thresholds to test
METHODS = {
            # "edm_2021" : [0.01,0.03,0.1,0.3,1,3,6,10,30,100,300],
            "edm" : [0,0.5,0.54,0.56,0.566,0.568,0.57,0.572,0.574,0.58,0.6],
            "residual" : [0,50,250,500,1000,2000,3000,4000,5000,10000,100000],
           }
NUM_FAULTS = [1,2,4,8,12]
BIAS_VALUES = [60,40,20,10]

def main():

    time_start = time.time()
    data_dir = os.path.join(os.path.dirname(
            os.path.realpath(__file__)),"data","simulated")
    processes = [Process(target=location_fde,
                         args=(os.path.join(data_dir,csv_file),)) \
                         for csv_file in sorted(os.listdir(data_dir))]

    PROCESS_PARALLEL = 2
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
    results_dir = os.path.join(os.getcwd(),"results",TIMESTAMP)
    for navdata_file in sorted(os.listdir(results_dir)):
        if navdata_file[:9] == "location_":
            results = glp.concat(results,glp.NavData(csv_path=os.path.join(results_dir,
                                                         navdata_file)))

    results.to_csv(prefix="fde_"+str(len(results)))

def location_fde(csv_path):
    """Compute FDE on new location.

    Parameters
    ----------
    csv_file : path
        Path to csv file.

    """
    results = glp.NavData()

    print(csv_path)
    location_name = "_".join(os.path.basename(csv_path).split("_")[:-1])
    print("location:",location_name)
    full_data_original = glp.NavData(csv_path=csv_path)

    for num_faults in NUM_FAULTS:
        print(location_name,"faults:",num_faults)

        for bias_value in BIAS_VALUES:
            print(location_name,"bias:",bias_value)

            full_data = full_data_original.copy()

            i = 0
            fault_gt = []
            corr_pr_m = []
            raw_pr_m = []
            for timestamp, _, navdata in glp.loop_time(full_data,"gps_millis"):

                if i % 100 == 0:
                    print("t:",timestamp)

                rand_index_order = np.arange(len(navdata))
                np.random.shuffle(rand_index_order)

                num_faults_added = max(0,min(num_faults,len(navdata)-5))
                faulty_idxs = list(rand_index_order)[:num_faults_added]

                navdata["corr_pr_m",faulty_idxs] += bias_value
                navdata["raw_pr_m",faulty_idxs] += bias_value
                corr_pr_m_subset = navdata["corr_pr_m"]
                raw_pr_m_subset = navdata["raw_pr_m"]

                fault_gt_subset = np.array([0] * len(navdata))
                if bias_value != 0.:
                    fault_gt_subset[faulty_idxs] = 1
                fault_gt += list(fault_gt_subset)
                corr_pr_m += list(corr_pr_m_subset)
                raw_pr_m += list(raw_pr_m_subset)
                i += 1

            full_data["fault_gt"] = fault_gt
            full_data["corr_pr_m"] = corr_pr_m
            full_data["raw_pr_m"] = raw_pr_m

            # iterate over methods
            for method, thresholds in METHODS.items():
                print(location_name,"method:",method)
                for threshold in thresholds:
                    print(location_name,"threshold:",threshold)
                    input_navdata = full_data.copy()
                    metrics, navdata = glp.evaluate_fde(input_navdata,method=method,
                                                        threshold=threshold,
                                                        # max_faults=num_faults,
                                                        verbose=False,
                                                        time_fde=True)

                    metrics_navdata = glp.NavData()
                    metrics_navdata["location_name"] = np.array([location_name])
                    metrics_navdata["bias"] = bias_value
                    metrics_navdata["threshold"] = threshold
                    metrics_navdata["faults"] = num_faults
                    for k,v in metrics.items():
                        metrics_navdata[k] = np.array([v])

                    if threshold == 0:
                        # str(np.round(0,4)).zfill(4) is '0000', but
                        # str(np.round(0.0,4)).zfill(4) is '0.00', so
                        threshold = int(0)

                    navdata_prefix = [method,location_name,str(num_faults),
                                      str(bias_value),str(np.round(threshold,4)).zfill(4)]
                    navdata_prefix = "_".join(navdata_prefix).replace(".","")
                    navdata.to_csv(prefix=navdata_prefix)

                    results = glp.concat(results,metrics_navdata)

        results.to_csv(prefix="location_"+location_name+"_"+str(len(results)))

if __name__ == "__main__":
    main()
