"""Testing EDM FDE on simulated data

"""

__authors__ = "D. Knowles"
__date__ = "15 Aug 2023"

import os

import numpy as np
import gnss_lib_py as glp
import matplotlib.pyplot as plt

locations = {
              "calgary" : (51.11056458625996, -114.1179704693596, 0.),
              "cape_town" : (-33.91700025297494, 18.403910329181112, 0.),
              "hong_kong" : (22.327793473417067, 114.17122448832379, 0.),
              "london" : (51.5097085796586, -0.16008158973060102, 0.),
              "munich" : (48.16985710449595, 11.551627945697028, 0.),
              "sao_paulo" : (-23.568026105263545, -46.736620380100675, 0.),
              "stanford_oval" : (37.42984154652992, -122.16946303566934, 0.),
              "sydney" : (-33.859749976799186, 151.22208557691505, 0.),
              "zurich" : (47.407491810621345, 8.500756183071228, 0.),
             }

results = glp.NavData()
NUM_FAULTS = 4

for location_name, location_tuple in locations.items():
    print("location name:",location_name)
    csv_path = os.path.join("/home","derek","improved-edm-fde","data",
                            "simulated",location_name + "_20230314.csv")

    print(csv_path)
    full_data = glp.NavData(csv_path=csv_path)

    full_data = full_data.copy(cols=list(np.arange(2000)))

    bias_values = [20,40,60,80,100]
    # bias_values = [100]

    for bias_value in bias_values:
        print("bias:",bias_value)

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
            faulty_idxs = list(rand_index_order)[:NUM_FAULTS]
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

        # thresholds = np.logspace(8,10,20)
        thresholds = np.linspace(0.5,0.7,21)
        # thresholds = np.linspace(0.,1.,20)
        # [1E8,1E9,1E10]
        for threshold in thresholds:
            print("threshold:",threshold)
            metrics = glp.evaluate_fde(full_data,method="edm",
                                       threshold=threshold,
                                       # max_faults=10,
                                       verbose=False,
                                       debug=True)

            metrics_navdata = glp.NavData()
            metrics_navdata["location_name"] = np.array([location_name])
            # metrics_navdata["location_name"] = location_name
            metrics_navdata["bias"] = bias_value
            metrics_navdata["threshold"] = threshold
            for k,v in metrics.items():
                metrics_navdata[k] = np.array([v])

            results.concat(metrics_navdata,inplace=True)
            # print(metrics_navdata)

    results.to_csv(prefix="metrics_"+str(len(results)))

# plt.figure()

print(np.max(results["balanced_accuracy"]))
glp.plot_metric(results,"threshold","balanced_accuracy",
                groupby="location_name",save=True)
# plt.xscale("log")
glp.plot_metric(results,"threshold","far",
                groupby="location_name",save=True)
# plt.xscale("log")
glp.plot_metric(results,"threshold","mdr",
                groupby="location_name",save=True)
# plt.xscale("log")

plt.show()
