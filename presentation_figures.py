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

    # without measurement noise
    # moving_eigenvalues(noise=False)

    # with measurement noise
    # moving_eigenvalues(noise=True)

    # singular value U matrix
    # moving_svd(noise=False)

    # skyplots from simulated data
    # world_skyplots_check()
    # world_skyplots()
    # sats_in_view()

    # simulated_data_metrics()

    #accuracy plots
    # accuracy_plots()

    # timing plots
    # timing_plots()

    # roc curve
    roc_curve()

def roc_curve():
    metrics_dir = "/home/derek/improved-edm-fde/results/20230904232029/"
    metrics_path = os.path.join(metrics_dir,"edm_residual_88_navdata.csv")

    navdata = glp.NavData(csv_path = metrics_path)
    navdata["method_and_bias_m"] = np.char.add(np.char.add(navdata["method"].astype(str),"_"),navdata["bias"].astype(str))

    fig = glp.plot_metric(navdata.where("method","edm"),
                    "far","tpr",
                    groupby="method_and_bias_m",
                    save=False,
                    # avg_y=True,
                    linewidth=5.0,
                    markersize=10
                    )
    plt.xlim(0.0,1.0)
    plt.ylim(0.0,1.0)

    plt.gca().set_prop_cycle(None)
    glp.plot_metric(navdata.where("method","residual"),
                    "far","tpr",
                    groupby="method_and_bias_m",
                    save=True,
                    prefix="faults_vs_ba",
                    linewidth=5.0,
                    markersize=10,
                    linestyle="dotted",
                    fig=fig,
                    )
    plt.xlim(0.0,1.0)
    plt.ylim(0.0,1.0)

    # navdata["threshold"] *= 1000
    glp.plot_metric(navdata.where("method","residual").where("bias",20),
                    "far","tpr",
                    groupby="threshold",
                    save=True,
                    prefix="faults_vs_ba",
                    linewidth=5.0,
                    markersize=10,
                    )

    for method_bias in np.unique(navdata["method_and_bias_m"]):
        print(method_bias)
        navdata_method_bias = navdata.where("method_and_bias_m",method_bias)

        far = navdata_method_bias["far"]
        tpr = navdata_method_bias["tpr"]
        far_sort = np.argsort(far)
        far = far[far_sort]
        tpr = tpr[far_sort]
        # print(navdata_method_bias["far"])
        # print(navdata_method_bias["tpr"])
        interp_point = np.interp(0.7,far,tpr)

        # plt.figure()
        # plt.plot(far,tpr,label="before")
        # plt.plot(0.5,interp_point,label="interp")
        tpr = tpr[far<0.5].tolist() + [interp_point]
        far = far[far<0.5].tolist() + [0.5]
        # plt.plot(far,tpr,label="after")

        # plt.legend()
        # plt.show()

        print(metrics.auc(far,tpr))
        # print(metrics.auc(navdata_method_bias["far"], navdata_method_bias["tpr"]))


    plt.show()


def timing_plots():
    file_path = "/home/derek/improved-edm-fde/results/20230816_combined/edm_residual_24_navdata.csv"

    navdata = glp.NavData(csv_path=file_path)

    fig=glp.plot_metric(navdata,
                    "faults","timestep_mean_ms",
                    groupby="method",
                    # title="BA Bias of " + str(bias) + "m",
                    save=True,
                    prefix="timing",
                    avg_y=True,
                    linewidth=5.0,
                    markersize=10,
                    )
    plt.yscale("log")
    _save_figure(fig,"timing")

    plt.show()


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

    plt.show()


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

    plt.show()

def world_skyplots():

    for location_name, i in locations.items():
        print("location name:",location_name)
        csv_path = os.path.join("/home","derek","improved-edm-fde","data",
                                "simulated",location_name + "_20230314.csv")

        navdata = glp.NavData(csv_path=csv_path)


        timestamp_start = datetime(year=2023, month=3, day=14, hour=i, tzinfo=timezone.utc)
        timestamp_end = datetime(year=2023, month=3, day=14, hour=i+1, tzinfo=timezone.utc)
        gps_millis = glp.datetime_to_gps_millis(np.array([timestamp_start,timestamp_end]))
        cropped_navdata = navdata.where("gps_millis",gps_millis[0],"geq").where("gps_millis",gps_millis[1],"leq")
        print(i,len(np.unique(cropped_navdata["gnss_id"])),len(np.unique(cropped_navdata["gnss_sv_id"])))
        glp.plot_skyplot(cropped_navdata,cropped_navdata,
                         prefix=location_name,save=True)

    plt.show()


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

    plt.show()

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
    data["singular_value"] = svd_s
    data["number_of_faults"] = np.array(["0 faults"]*len(data))

    plt.rcParams['figure.figsize'] = [5.9, 3.5]
    fig = glp.plot_metric(data,"singular_value",
                          groupby="number_of_faults",
                          linestyle="None", title="", markersize=8,)
    plt.xlim([-0.5,len(data)-0.5])
    plt.yscale("log")
    plt.ylim(1E-5,1E17)
    _save_figure(fig,"0_faults")


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
        data_fault["singular_value"] = svd_s
        if i == 1:
            data_fault["number_of_faults"] = np.array([str(i) + " fault"]*len(data_fault))
        else:
            data_fault["number_of_faults"] = np.array([str(i) + " faults"]*len(data_fault))

        data.concat(data_fault,inplace=True)
        plt.rcParams['figure.figsize'] = [5.9, 3.5]
        fig = glp.plot_metric(data,"singular_value",
                              groupby="number_of_faults",
                              linestyle="None", title="", markersize=8,)
        plt.xlim([-0.5,len(data_fault)-0.5])
        plt.yscale("log")
        plt.ylim(1E-5,1E17)
        _save_figure(fig,str(i)+"_faults")

    plt.show()


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

    plt.show()

if __name__ == "__main__":
    main()
