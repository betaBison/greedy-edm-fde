"""Reproduce figures for presentation.

"""

__authors__ = "D. Knowles"
__date__ = "16 Aug 2023"

import numpy as np
import gnss_lib_py as glp
import matplotlib.pyplot as plt

metrics_path = "/home/derek/improved-edm-fde/results/20230816195428/metrics_945_navdata.csv"

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

glp.plot_metric(navdata,
                "threshold","balanced_accuracy",
                groupby="bias",
                # title="BA Bias of " + str(bias) + "m",
                save=True,
                avg_y=True,
                # linewidth=5.0,
                )
plt.ylim(0.4,1.0)

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

# glp.plot_metric(navdata,
#                 "threshold","timestep_mean",
#                 groupby="location_name",
#                 title="timestep_mean",
#                 avg_y = True,
#                 save=True)

glp.plot_metric(navdata,
                "threshold","timestep_median_ms",
                groupby="location_name",
                avg_y = True,
                save=True)

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
