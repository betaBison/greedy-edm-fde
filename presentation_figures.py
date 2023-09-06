"""Reproduce figures for presentation.

"""

__authors__ = "D. Knowles"
__date__ = "16 Aug 2023"

import os
from datetime import datetime, timezone

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import gnss_lib_py as glp
from gnss_lib_py.algorithms.fde import _edm_from_satellites_ranges
from gnss_lib_py.utils.visualizations import _save_figure
from gnss_lib_py.utils.file_operations import TIMESTAMP
np.random.seed(314)

locations = {
              "calgary" : 10,
              "cape_town" : 6,
              "hong_kong" : 1,
              "london" : 21,
              "munich" : 10,
              "sao_paulo" : 17,
              "stanford_oval" : 10,
              "sydney" : 11,
              "zurich" : 2,
             }

def main():

    # timing plots
    # timing_plots_calculations()
    # timing_plots()

    # without measurement noise
    moving_eigenvalues(noise=False)

    # with measurement noise
    moving_eigenvalues(noise=True)

    # singular value U matrix
    # moving_svd(noise=False)

    # skyplots from simulated data
    # world_skyplots_check()
    # world_skyplots()
    # sats_in_view()

    # simulated_data_metrics()

    #accuracy plots
    # accuracy_plots()

    # roc curve
    # roc_curve()
    # auc_table()



    plt.show()

def create_label(items):
    if len(items) == 1:
        return items
    label = items[0]
    for l_index in range(1,len(items)):
        label = np.char.add(label,"_")
        label = np.char.add(label,items[l_index])

    return label

def roc_curve():
    metrics_dir = "/home/derek/improved-edm-fde/results/20230905010421/"
    metrics_path = os.path.join(metrics_dir,"edm_residual_3960_navdata.csv")

    navdata = glp.NavData(csv_path = metrics_path)
    navdata["method_and_bias_m"] = np.char.add(np.char.add(navdata["method"].astype(str),"_"),navdata["bias"].astype(str))
    navdata.rename({"far":"false_alarm_rate","tpr":"true_positive_rate"},inplace=True)
    navdata["glp_label"] = create_label([
                                         navdata["faults"].astype(str),
                                         navdata["location_name"].astype(str),
                                        ])

    for glp_label in ["8_munich"]:

        fig = glp.plot_metric(navdata.where("glp_label",glp_label).where("method","edm"),
                        "false_alarm_rate","true_positive_rate",
                        groupby="method_and_bias_m",
                        save=False,
                        avg_y=True,
                        linewidth=5.0,
                        markersize=10,
                        )
        plt.xlim(0.0,0.9)
        plt.ylim(0.0,1.0)

        plt.gca().set_prop_cycle(None)
        glp.plot_metric(navdata.where("glp_label",glp_label).where("method","residual"),
                        "false_alarm_rate","true_positive_rate",
                        groupby="method_and_bias_m",
                        avg_y=True,
                        save=True,
                        title="",
                        prefix="roc_curve",
                        linewidth=5.0,
                        markersize=10,
                        linestyle="dotted",
                        fig=fig,
                        )
        plt.xlim(0.0,0.9)
        plt.ylim(0.0,1.0)

def auc_table():
    metrics_dir = "/home/derek/improved-edm-fde/results/20230905010421/"
    metrics_path = os.path.join(metrics_dir,"edm_residual_3960_navdata.csv")

    navdata = glp.NavData(csv_path = metrics_path)
    navdata_cropped = navdata.where("bias",60).where("faults",8).concat(navdata.where("bias",20).where("faults",1))

    # navdata["method_and_bias_m"] = np.char.add(np.char.add(navdata["method"].astype(str),"_"),navdata["bias"].astype(str))
    # navdata["glp_label"] = np.char.add(np.char.add(navdata["method"].astype(str),"_"),navdata["bias"].astype(str))
    navdata_cropped["glp_label"] = create_label([navdata_cropped["bias"].astype(str),
                                                 navdata_cropped["faults"].astype(str),
                                                 navdata_cropped["location_name"].astype(str),
                                                 # navdata_cropped["method"].astype(str),
                                                 ])
    navdata_auc = glp.NavData()
    for glp_label in np.unique(navdata_cropped["glp_label"]):
        navdata_method_bias = navdata_cropped.where("glp_label",glp_label)

        label_navdata = glp.NavData()
        # label_navdata["method"] = np.array([navdata_method_bias["method",0]])
        label_navdata["location_name"] = np.array([navdata_method_bias["location_name",0]])
        label_navdata["bias"] = np.array([navdata_method_bias["bias",0]])
        label_navdata["faults"] = np.array([navdata_method_bias["faults",0]])

        # print(method_bias)
        for method in np.unique(navdata_cropped["method"]):

            far = navdata_method_bias.where("method",method)["far"]
            tpr = navdata_method_bias.where("method",method)["tpr"]
            far_sort = np.argsort(far)
            far = far[far_sort]
            tpr = tpr[far_sort]
            # print(navdata_method_bias["far"])
            # print(navdata_method_bias["tpr"])
            interp_value = 0.7
            interp_point = np.interp(0.7,far,tpr)

            # plt.figure()
            # plt.plot(far,tpr,label="before")
            # plt.plot(interp_value,interp_point,label="interp")
            tpr = tpr[far<interp_value].tolist() + [interp_point]
            far = far[far<interp_value].tolist() + [interp_value]
            # plt.plot(far,tpr,label="after")
            #
            # plt.legend()
            # plt.show()

            auc = metrics.auc(far,tpr)


            label_navdata[method + "_auc"] = np.round(auc,2)

        navdata_auc = navdata_auc.concat(label_navdata)
        # print(metrics.auc(far,tpr))

    navdata_auc.to_csv(prefix="auc_latex",sep="&",lineterminator="\\\\\n")
    navdata_auc.to_csv(prefix="auc")

    navdata_auc["edm_wins"] = navdata_auc["edm_auc"] > navdata_auc["residual_auc"]
    print(navdata_auc)

def timing_plots_calculations():
    metrics_dir = "/home/derek/improved-edm-fde/results/20230905010421/"
    metrics_path = os.path.join(metrics_dir,"edm_residual_3960_navdata.csv")
    navdata = glp.NavData(csv_path = metrics_path)
    navdata["glp_label"] = create_label([navdata["faults"].astype(str),
                                         navdata["bias"].astype(str),
                                         navdata["location_name"].astype(str),
                                         navdata["method"].astype(str),
                                         ])
    timing_measurements = {}
    timing_faults = {}
    methods = ("edm","residual")
    for method in methods:
        timing_faults[method] = {}
        timing_measurements[method] = {}

    for glp_label in np.unique(navdata["glp_label"]):
        print("options:",glp_label)
        navdata_cropped = navdata.where("glp_label",glp_label)
        row_idx = np.argmax(navdata_cropped["balanced_accuracy"])
        method = navdata_cropped["method",row_idx].item()
        location_name = navdata_cropped["location_name",row_idx].item()
        faults = navdata_cropped["faults",row_idx].item()
        bias = navdata_cropped["bias",row_idx].item()
        threshold = navdata_cropped["threshold",row_idx].item()
        if threshold == 500.0:
            threshold = 250.
        # print("b",threshold)
        if threshold < 1:
            threshold = str(np.round(threshold,4)).zfill(4)
        else:
            threshold = str(int(threshold)).zfill(4)
        # print("a",threshold)

        file_prefix = [method,location_name,str(faults),str(bias),
                       threshold]
        file_name = "_".join(file_prefix).replace(".","") + "_navdata.csv"
        file_path = os.path.join(metrics_dir,file_name)
        # print("fp:",file_path)
        navdata_file = glp.NavData(csv_path=file_path)

        for _, _, navdata_subset in navdata_file.loop_time("gps_millis"):
            compute_time_ms = navdata_subset["compute_time_s",0]*1000
            faults = np.sum(navdata_subset["fault_gt"])
            if faults not in [1,2,4,8,12]:
                continue
            num_measurements = len(navdata_subset)

            if faults not in timing_faults[method]:
                timing_faults[method][faults] = [compute_time_ms]
            else:
                timing_faults[method][faults].append(compute_time_ms)

            if num_measurements not in timing_measurements[method]:
                timing_measurements[method][num_measurements] = [compute_time_ms]
            else:
                timing_measurements[method][num_measurements].append(compute_time_ms)


    faults_navdata = glp.NavData()
    measurements_navdata = glp.NavData()
    for method in methods:
        for k,v in timing_faults[method].items():
            single_navdata = glp.NavData()
            single_navdata["method"] = np.array([method])
            single_navdata["faults"] = k
            single_navdata["mean_compute_time_ms"] = np.mean(v)
            single_navdata["min_compute_time_ms"] = np.min(v)
            single_navdata["max_compute_time_ms"] = np.max(v)
            single_navdata["median_compute_time_ms"] = np.median(v)
            single_navdata["std_compute_time_ms"] = np.std(v)
            faults_navdata = faults_navdata.concat(single_navdata)
        for k,v in timing_measurements[method].items():
            single_navdata = glp.NavData()
            single_navdata["method"] = np.array([method])
            single_navdata["measurements"] = k
            single_navdata["mean_compute_time_ms"] = np.mean(v)
            single_navdata["min_compute_time_ms"] = np.min(v)
            single_navdata["max_compute_time_ms"] = np.max(v)
            single_navdata["median_compute_time_ms"] = np.median(v)
            single_navdata["std_compute_time_ms"] = np.std(v)
            measurements_navdata = measurements_navdata.concat(single_navdata)

    measurements_navdata.to_csv(prefix="measurements_timing")
    faults_navdata.to_csv(prefix="faults_timing")

def timing_plots():
    # metrics_dir = "/home/derek/improved-edm-fde/results/" + TIMESTAMP + "/"
    metrics_dir = "/home/derek/improved-edm-fde/results/20230905180735/"

    faults_navdata = glp.NavData(csv_path=os.path.join(metrics_dir,"faults_timing_navdata.csv"))
    measurements_navdata = glp.NavData(csv_path=os.path.join(metrics_dir,"measurements_timing_navdata.csv"))

    faults_navdata.sort("faults",inplace=True)
    measurements_navdata.sort("measurements",inplace=True)

    for graph_type in ["faults","measurements"]:
        if graph_type == "faults":
            navdata_plot = faults_navdata
        else:
            navdata_plot = measurements_navdata

        fig = glp.plot_metric(navdata_plot,
                        graph_type,"mean_compute_time_ms",
                        groupby="method",
                        save=False,
                        linewidth=5.0,
                        markersize=10,
                        linestyle="None",
                        )

        plt.gca().set_prop_cycle(None)
        for method in ["edm","residual"]:
            print(method,graph_type)
            navdata_std = navdata_plot.where("method",method)
            print("min:",navdata_std["mean_compute_time_ms"] - 1*navdata_std["std_compute_time_ms"])
            print("max:",navdata_std["mean_compute_time_ms"] + 1*navdata_std["std_compute_time_ms"])
            plt.fill_between(navdata_std[graph_type],
                             navdata_std["mean_compute_time_ms"] - 1*navdata_std["std_compute_time_ms"],
                             navdata_std["mean_compute_time_ms"] + 1*navdata_std["std_compute_time_ms"],
                             alpha=0.5)
        plt.yscale("log")
        # if graph_type == "faults":
        #     plt.ylim([1E-10,1E0])
        _save_figure(fig,graph_type+"_"+"mean_compute_time_ms")

    # file_path = "/home/derek/improved-edm-fde/results/20230816_combined/edm_residual_24_navdata.csv"
    #
    # navdata = glp.NavData(csv_path=file_path)
    #
    # fig=glp.plot_metric(navdata,
    #                 "faults","timestep_mean_ms",
    #                 groupby="method",
    #                 # title="BA Bias of " + str(bias) + "m",
    #                 save=True,
    #                 prefix="timing",
    #                 avg_y=True,
    #                 linewidth=5.0,
    #                 markersize=10,
    #                 )
    # plt.yscale("log")
    # _save_figure(fig,"timing")

def accuracy_plots():
    metrics_dir = "/home/derek/improved-edm-fde/results/20230816_combined/"

    edm_navdata = glp.NavData()
    r_navdata = glp.NavData()
    for file in os.listdir(metrics_dir):
        file_path = os.path.join(metrics_dir,file)
        location_data = glp.NavData(csv_path=file_path)
        if "metrics_144" in file:
            edm_navdata.concat(location_data,inplace=True)
            print(len(edm_navdata))
        elif "residual_432" in file:
            r_navdata.concat(location_data,inplace=True)

    glp.plot_metric(edm_navdata,
                    "threshold","balanced_accuracy",
                    groupby="bias",
                    # title="BA Bias of " + str(bias) + "m",
                    save=True,
                    prefix="edm_bias",
                    avg_y=True,
                    linewidth=5.0,
                    markersize=10,
                    )
    plt.ylim(0.4,1.0)

    glp.plot_metric(edm_navdata.where("bias",60),
                    "threshold","balanced_accuracy",
                    groupby="faults",
                    # title="BA Bias of " + str(bias) + "m",
                    save=True,
                    prefix="edm_faults",
                    avg_y=True,
                    linewidth=5.0,
                    markersize=10,
                    )
    plt.ylim(0.4,1.0)

    glp.plot_metric(r_navdata,
                    "threshold","balanced_accuracy",
                    groupby="bias",
                    # title="BA Bias of " + str(bias) + "m",
                    save=True,
                    prefix="residual_bias",
                    avg_y=True,
                    linewidth=5.0,
                    markersize=10
                    )
    plt.ylim(0.4,1.0)

    edm_navdata["method_and_bias_m"] = np.array(["edm_"+str(b)+"_m" for b in edm_navdata["bias"]])
    fig = glp.plot_metric(edm_navdata.where("threshold",0.5700000000000001),
                    "faults","balanced_accuracy",
                    groupby="method_and_bias_m",
                    save=True,
                    avg_y=True,
                    linewidth=5.0,
                    markersize=10
                    )
    plt.ylim(0.4,1.0)

    r_navdata["method_and_bias_m"] = np.array(["residual_"+str(b)+"_m" for b in r_navdata["bias"]])
    glp.plot_metric(r_navdata.where("threshold",17.),
                    "faults","balanced_accuracy",
                    groupby="method_and_bias_m",
                    save=True,
                    prefix="faults_vs_ba",
                    avg_y=True,
                    linewidth=5.0,
                    markersize=10,
                    linestyle="dotted",
                    fig=fig,
                    )
    plt.ylim(0.4,1.0)


def simulated_data_metrics():

    metrics = glp.NavData()

    print()

    for location_name, i in locations.items():
        print("location name:",location_name)
        csv_path = os.path.join("/home","derek","improved-edm-fde","data",
                                "simulated",location_name + "_20230314.csv")

        navdata = glp.NavData(csv_path=csv_path)

        metrics_subset = glp.NavData()

        sats_in_view = []
        gps_millis = []
        for timestamp,_,subset in navdata.loop_time("gps_millis"):
            sats_in_view.append(len(np.unique(subset["gnss_sv_id"])))
            gps_millis.append(timestamp)

        metrics_subset["sats_in_view"] = sats_in_view
        metrics_subset["gps_millis"] = gps_millis
        metrics_subset["location_name"] = np.array([location_name]*len(sats_in_view))
        metrics.concat(metrics_subset,inplace=True)

    print("total_timesteps:",len(np.unique(metrics["gps_millis"])),len(metrics))
    print("min:",np.min(metrics["sats_in_view"]))
    print("mean:",np.mean(metrics["sats_in_view"]))
    print("max:",np.max(metrics["sats_in_view"]))

def sats_in_view():

    metrics = glp.NavData()

    for location_name, i in locations.items():
        print("location name:",location_name)
        csv_path = os.path.join("/home","derek","improved-edm-fde","data",
                                "simulated",location_name + "_20230314.csv")

        navdata = glp.NavData(csv_path=csv_path)
        if location_name in ("calgary","zurich","london"):
            navdata = navdata.where("el_sv_deg",30.,"geq")

        metrics_subset = glp.NavData()

        sats_in_view = []
        gps_millis = []
        for timestamp,_,subset in navdata.loop_time("gps_millis",delta_t_decimals=-6):
            sats_in_view.append(len(np.unique(subset["gnss_sv_id"])))
            gps_millis.append(timestamp)

        metrics_subset["sats_in_view"] = sats_in_view
        metrics_subset["gps_millis"] = gps_millis
        metrics_subset["location_name"] = np.array([location_name]*len(sats_in_view))
        metrics.concat(metrics_subset,inplace=True)

    metrics.rename({"gps_millis":"gps_time_milliseconds"},inplace=True)

    # plt.rcParams['figure.figsize'] = [3.5, 3.5]
    glp.plot_metric(metrics,"gps_time_milliseconds","sats_in_view",
                    groupby="location_name",save=True)

def world_skyplots():

    for location_name, i in locations.items():
        # only plot those used for presentation
        if location_name in ("sao_paulo","hong_kong","zurich"):
            print("location name:",location_name)
            csv_path = os.path.join("/home","derek","improved-edm-fde","data",
                                    "simulated",location_name + "_20230314.csv")

            navdata = glp.NavData(csv_path=csv_path)
            if location_name in ("calgary","zurich","london"):
                navdata = navdata.where("el_sv_deg",30.,"geq")

            timestamp_start = datetime(year=2023, month=3, day=14, hour=i, tzinfo=timezone.utc)
            timestamp_end = datetime(year=2023, month=3, day=14, hour=i+1, tzinfo=timezone.utc)
            gps_millis = glp.datetime_to_gps_millis(np.array([timestamp_start,timestamp_end]))
            cropped_navdata = navdata.where("gps_millis",gps_millis[0],"geq").where("gps_millis",gps_millis[1],"leq")
            print(i,len(np.unique(cropped_navdata["gnss_id"])),len(np.unique(cropped_navdata["gnss_sv_id"])))
            glp.plot_skyplot(cropped_navdata,cropped_navdata,
                             prefix=location_name,save=True)

def world_skyplots_check():

    for location_name, location_tuple in locations.items():
        print("location name:",location_name)
        csv_path = os.path.join("/home","derek","improved-edm-fde","data",
                                "simulated",location_name + "_20230314.csv")

        navdata = glp.NavData(csv_path=csv_path)

        for i in range(0,23):
          timestamp_start = datetime(year=2023, month=3, day=14, hour=i, tzinfo=timezone.utc)
          timestamp_end = datetime(year=2023, month=3, day=14, hour=i+1, tzinfo=timezone.utc)
          gps_millis = glp.datetime_to_gps_millis(np.array([timestamp_start,timestamp_end]))
          cropped_navdata = navdata.where("gps_millis",gps_millis[0],"geq").where("gps_millis",gps_millis[1],"leq")
          print(i,len(np.unique(cropped_navdata["gnss_id"])),len(np.unique(cropped_navdata["gnss_sv_id"])))

def moving_svd(noise):
    file_path = "/home/derek/improved-edm-fde/data/simulated/stanford_oval_gt_20230314.csv"
    navdata = glp.NavData(csv_path=file_path)

    navdata = navdata.copy(cols=np.arange(9))

    if noise:
        navdata["corr_pr_m"] += np.random.normal(loc=0.0,scale=10,size=len(navdata))

    edm = _edm_from_satellites_ranges(navdata[["x_sv_m","y_sv_m","z_sv_m"]],
                                      navdata["corr_pr_m"])

    dims = edm.shape[1]
    center = np.eye(dims) - (1./dims) * np.ones((dims,dims))
    gram = -0.5 * center @ edm @ center

    # calculate the singular value decomposition
    svd_u, svd_s, svd_vt = np.linalg.svd(gram,full_matrices=True)

    fig = plt.figure(figsize=(3.5,3.5))
    plt.imshow(np.abs(svd_u),cmap="cividis")
    plt.colorbar()
    _save_figure(fig,"u_0_faults")


    navdata["corr_pr_m",3] += 1000.
    edm = _edm_from_satellites_ranges(navdata[["x_sv_m","y_sv_m","z_sv_m"]],
                                      navdata["corr_pr_m"])

    dims = edm.shape[1]
    center = np.eye(dims) - (1./dims) * np.ones((dims,dims))
    gram = -0.5 * center @ edm @ center

    # calculate the singular value decomposition
    svd_u, svd_s, svd_vt = np.linalg.svd(gram,full_matrices=True)

    print(list(np.argsort(np.mean(np.abs(svd_u)[:,3:5],axis=1))[::-1]))

    fig = plt.figure(figsize=(3.5,3.5))
    plt.imshow(np.abs(svd_u),cmap="cividis")
    plt.colorbar()
    _save_figure(fig,"u_1_faults")

def moving_eigenvalues(noise):
    file_path = "/home/derek/improved-edm-fde/data/simulated/stanford_oval_gt_20230314.csv"
    navdata = glp.NavData(csv_path=file_path)

    if noise:
        navdata["corr_pr_m"] += np.random.normal(loc=0.0,scale=10,size=len(navdata))

    edm = _edm_from_satellites_ranges(navdata[["x_sv_m","y_sv_m","z_sv_m"]],
                                      navdata["corr_pr_m"])

    dims = edm.shape[1]
    center = np.eye(dims) - (1./dims) * np.ones((dims,dims))
    gram = -0.5 * center @ edm @ center

    # calculate the singular value decomposition
    svd_u, svd_s, svd_vt = np.linalg.svd(gram,full_matrices=True)

    data = glp.NavData()
    data["eigenvalue_magnitude"] = svd_s
    data["number_of_faults"] = np.array(["0 faults"]*len(data))

    plt.rcParams['figure.figsize'] = [5.9, 3.5]
    fig = glp.plot_metric(data,"eigenvalue_magnitude",
                          groupby="number_of_faults",
                          linestyle="None", title="", markersize=8,)
    plt.xlim([-0.5,len(data)-0.5])
    plt.yscale("log")
    plt.ylim(1E-5,1E17)
    if noise:
        _save_figure(fig,"faults_with_noise_"+str(0).zfill(2))
        _save_figure(fig,"faults_with_noise_"+str(19).zfill(2))
    else:
        _save_figure(fig,"faults_"+str(0).zfill(2))
        _save_figure(fig,"faults_"+str(19).zfill(2))


    for i in range(1,10):

        navdata["corr_pr_m",i] += 1000.
        edm = _edm_from_satellites_ranges(navdata[["x_sv_m","y_sv_m","z_sv_m"]],
                                          navdata["corr_pr_m"])

        dims = edm.shape[1]
        center = np.eye(dims) - (1./dims) * np.ones((dims,dims))
        gram = -0.5 * center @ edm @ center

        # calculate the singular value decomposition
        svd_u, svd_s, svd_vt = np.linalg.svd(gram,full_matrices=True)

        data_fault = glp.NavData()
        data_fault["eigenvalue_magnitude"] = svd_s
        if i == 1:
            data_fault["number_of_faults"] = np.array([str(i) + " fault"]*len(data_fault))
        else:
            data_fault["number_of_faults"] = np.array([str(i) + " faults"]*len(data_fault))

        data.concat(data_fault,inplace=True)
        plt.rcParams['figure.figsize'] = [5.9, 3.5]
        fig = glp.plot_metric(data,"eigenvalue_magnitude",
                              groupby="number_of_faults",
                              linestyle="None", title="", markersize=8,)
        plt.xlim([-0.5,len(data_fault)-0.5])
        plt.yscale("log")
        plt.ylim(1E-5,1E17)
        if noise:
            _save_figure(fig,"faults_with_noise_"+str(i).zfill(2))
            _save_figure(fig,"faults_with_noise_"+str(19-i).zfill(2))
        else:
            _save_figure(fig,"faults_"+str(i).zfill(2))
            _save_figure(fig,"faults_"+str(19-i).zfill(2))


def pre_tests():

    # metrics_path = "/home/derek/improved-edm-fde/results/20230816195428_2000_test/metrics_945_navdata.csv"
    # metrics_path = "/home/derek/improved-edm-fde/results/20230817005344_calgary/metrics_189_navdata.csv"
    # metrics_path = "/home/derek/improved-edm-fde/results/20230817022058_residual/residual_336_navdata.csv"
    # metrics_path = "/home/derek/improved-edm-fde/results/20230817022759/residual_144_navdata.csv"
    # metrics_path = "/home/derek/improved-edm-fde/results/20230816_combined/metrics_144_navdata_calgary.csv"
    metrics_path = "/home/derek/improved-edm-fde/results/20230816_combined/residual_432_navdata_1.csv"

    navdata = glp.NavData(csv_path=metrics_path)

    glp.plot_metric(navdata,
                    "threshold","balanced_accuracy",
                    groupby="bias",
                    # title="BA Bias of " + str(bias) + "m",
                    save=True,
                    avg_y=True,
                    # linewidth=5.0,
                    )
    plt.ylim(0.4,1.0)

    for bias in np.unique(navdata["bias"]):
        glp.plot_metric(navdata.where("bias",bias),
                        "threshold","balanced_accuracy",
                        groupby="faults",
                        title="BA Bias of " + str(bias) + "m",
                        save=True,
                        # avg_y=True,
                        # linewidth=5.0,
                        )
        plt.ylim(0.4,1.0)
    #
    # methods = ["fault_edm","all","fault_gt"]
    # linestyles = ["solid","dotted","dashdot"]


    # fig_mean = None
    # for mmm, method in enumerate(methods):
    #     fig_mean = glp.plot_metric(navdata.where("threshold",0.57),
    #                                "faults",method+"_pos_error_mean",
    #                                groupby="bias",
    #                                save=True,
    #                                fig = fig_mean,
    #                                linestyle=linestyles[mmm]
    #                                )
    # fig_std = None
    # for mmm, method in enumerate(methods):
    #     fig_std = glp.plot_metric(navdata.where("threshold",0.57),
    #                                "faults",method+"_pos_error_std",
    #                                groupby="bias",
    #                                save=True,
    #                                fig = fig_std,
    #                                linestyle=linestyles[mmm]
    #                                )
    # fig_std = None
    # for mmm, method in enumerate(methods):
    #     fig_std = glp.plot_metric(navdata.where("bias",60).where("threshold",0.56,"geq").where("threshold",0.58,"leq"),
    #                                "faults",method+"_pos_error_mean",
    #                                save=True,
    #                                fig = fig_std,
    #                                linestyle=linestyles[mmm]
    #                                )

    # fig_mean = glp.plot_metric(navdata.where("bias",60),
    #                            "threshold","fault_edm_pos_error_std",
    #                            groupby="faults",
    #                            save=True,
    #                            )

    # glp.plot_metric(navdata,
    #                 "faults","all_pos_error_std",
    #                 groupby="bias",
    #                 save=True,
    #                 )


    # glp.plot_metric(navdata,
    #                 "threshold","timestep_max",
    #                 groupby="location_name",
    #                 title="timestep_max" ,
    #                 avg_y = True,
                    # save=True)


    # glp.plot_metric(navdata.where("bias",bias),
    #                 "threshold","mdr",
    #                 groupby="location_name",
    #                 title="MD Bias of " + str(bias) + "m",
    #                 save=True)
    # plt.ylim(0.0,1.0)
    #
    # glp.plot_metric(navdata.where("bias",bias),
    #                 "threshold","far",
    #                 groupby="location_name",
    #                 title="FA Bias of " + str(bias) + "m",
    #                 save=True)
    # plt.ylim(0.0,1.0)

    # glp.plot_metric(navdata,
    #                 "threshold","timestep_min",
    #                 groupby="location_name",
    #                 title="timestep_min",
    #                 avg_y = True,
    #                 save=True)

    glp.plot_metric(navdata,
                    "threshold","timestep_mean_ms",
                    groupby="location_name",
                    title="timestep_mean",
                    avg_y = True,
                    save=True)
    #
    # glp.plot_metric(navdata,
    #                 "threshold","timestep_median_ms",
    #                 groupby="location_name",
    #                 avg_y = True,
    #                 save=True)

    # glp.plot_metric(navdata,
    #                 "measurement_counts_mean",
    #                 groupby="location_name",
    #                 title="measurement_counts_mean",
    #                 # avg_y = True,
    #                 save=True)
    #
    # glp.plot_metric(navdata,
    #                 "measurement_counts_max",
    #                 groupby="location_name",
    #                 title="measurement_counts_max",
    #                 # avg_y = True,
    #                 save=True)
    #
    # glp.plot_metric(navdata,
    #                 "measurement_counts_min",
    #                 groupby="location_name",
    #                 title="measurement_counts_min",
    #                 # avg_y = True,
    #                 save=True)

    # glp.plot_metric(navdata,
    #                 "measurement_counts_mean",
    #                 "timestep_mean",
    #                 groupby="location_name",
    #                 title="measurement_counts_min",
    #                 # avg_y = True,
    #                 linestyle="None",
    #                 save=True)

    # glp.plot_metric(navdata,
    #                 "threshold","timestep_max",
    #                 groupby="location_name",
    #                 title="timestep_max" ,
    #                 avg_y = True,
    #                 save=True)


if __name__ == "__main__":
    main()
