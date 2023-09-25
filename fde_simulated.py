"""Testing EDM FDE on simulated data

"""

__authors__ = "D. Knowles"
__date__ = "15 Aug 2023"

import os

import numpy as np
import gnss_lib_py as glp

np.random.seed(314)

# methods and thresholds to test
METHODS = {
            "edm" : [0.0,0.5,0.54,0.56,0.566,0.568,0.57,0.572,0.574,0.58,0.6],
            "residual" : [0,50,250,500,1000,2000,3000,4000,5000,10000,100000],
           }
NUM_FAULTS = [1,2,4,8,12]
BIAS_VALUES = [60,40,20,10]

results = glp.NavData()

data_dir = os.path.join(os.getcwd(),"data","simulated")
for csv_file in os.listdir(data_dir):

    location_name = "_".join(csv_file.split("_")[:-1])
    print("location:",location_name)
    csv_path = os.path.join(data_dir,csv_file)
    print(csv_path)
    full_data_original = glp.NavData(csv_path=csv_path)

    for num_faults in NUM_FAULTS:
        print("faults:",num_faults)

        for bias_value in BIAS_VALUES:
            print("bias:",bias_value)

            full_data = full_data_original.copy()

            i = 0
            fault_gt = []
            corr_pr_m = []
            raw_pr_m = []
            for timestamp, _, navdata in full_data.loop_time("gps_millis"):

                # navdata = navdata.copy(cols=list(np.arange(10)))
                if i % 100 == 0:
                    print("t:",timestamp)

                # faulty_idx = list(np.random.randint(0,len(navdata),size=int(0.5*len(navdata))))
                rand_index_order = np.arange(len(navdata))
                np.random.shuffle(rand_index_order)

                num_faults_added = max(0,min(num_faults,len(navdata)-5))
                faulty_idxs = list(rand_index_order)[:num_faults_added]
                # print("\n\n\nfault:",faulty_idxs,navdata["gnss_sv_id",faulty_idxs])

                # print("before",navdata["corr_pr_m"][faulty_idxs] )
                navdata["corr_pr_m",faulty_idxs] += bias_value
                navdata["raw_pr_m",faulty_idxs] += bias_value
                corr_pr_m_subset = navdata["corr_pr_m"]
                raw_pr_m_subset = navdata["raw_pr_m"]
                # print("after",navdata["corr_pr_m"][faulty_idxs])

                fault_gt_subset = np.array([0] * len(navdata))
                if bias_value != 0.:
                    fault_gt_subset[faulty_idxs] = 1
                # print(fault_gt_subset)
                fault_gt += list(fault_gt_subset)
                corr_pr_m += list(corr_pr_m_subset)
                raw_pr_m += list(raw_pr_m_subset)
                i += 1

            full_data["fault_gt"] = fault_gt
            full_data["corr_pr_m"] = corr_pr_m
            full_data["raw_pr_m"] = raw_pr_m

            # iterate over methods
            for method, thresholds in METHODS.items():
                print("method:",method)
                for threshold in thresholds:
                    print("threshold:",threshold)

                    input_navdata = full_data.copy()
                    metrics, navdata = glp.evaluate_fde(input_navdata,method=method,
                                                        threshold=threshold,
                                                        # max_faults=10,
                                                        verbose=False,
                                                        time_fde=True)

                    metrics_navdata = glp.NavData()
                    metrics_navdata["location_name"] = np.array([location_name])
                    metrics_navdata["bias"] = bias_value
                    metrics_navdata["threshold"] = threshold
                    metrics_navdata["faults"] = num_faults
                    for k,v in metrics.items():
                        metrics_navdata[k] = np.array([v])

                    navdata_prefix = [method,location_name,str(num_faults),
                                      str(bias_value),str(np.round(threshold,4)).zfill(4)]
                    navdata_prefix = "_".join(navdata_prefix).replace(".","")
                    navdata.to_csv(prefix=navdata_prefix)

                    results.concat(metrics_navdata,inplace=True)

        results.to_csv(prefix="fde_"+str(len(results)))
