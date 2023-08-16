"""Simulation test

"""

__authors__ = "D. Knowles"
__date__ = "15 Aug 2023"

import numpy as np
import gnss_lib_py as glp
import matplotlib.pyplot as plt
from datetime import datetime, timezone

start_datetime = datetime(2023,3,14,0,tzinfo=timezone.utc)
# end_datetime = datetime(2023,3,14,23,59,59,999999,tzinfo=timezone.utc)
end_datetime = datetime(2023,3,15,0,tzinfo=timezone.utc)
start_gps_millis = int(glp.datetime_to_gps_millis(start_datetime))
end_gps_millis = int(glp.datetime_to_gps_millis(end_datetime))

print(start_gps_millis)
print(end_gps_millis)

# sp3_paths = glp.load_ephemeris("sp3",
#                                gps_millis = [start_gps_millis,end_gps_millis],
#                                verbose=True)
# sp3 = glp.Sp3(sp3_paths)
#
# clk_paths = glp.load_ephemeris("clk",
#                                gps_millis = [start_gps_millis,end_gps_millis],
#                                verbose=True)
# clk = glp.Clk(clk_paths)
#
# sp3_gnss_ids = set(np.unique(sp3["gnss_sv_id"]))
# clk_gnss_ids = set(np.unique(clk["gnss_sv_id"]))
#
# print("diffs")
# print(sp3_gnss_ids - clk_gnss_ids)
# print(clk_gnss_ids - sp3_gnss_ids)
# print(len(sp3_gnss_ids))
# print(sorted(sp3_gnss_ids))

navdata = glp.NavData()
gnss_ids = ["beidou"] * 37 + ["galileo"] * 25 + ["gps"] * 32 \
         + ["qzss"] * 3 + ["glonass"] * 20

sv_ids = list(np.arange(6,15)) + [16] + list(np.arange(19,31)) \
       + list(np.arange(32,47)) \
       + list(np.arange(1,6)) + list(np.arange(7,16)) + [18,19,21] \
       + [24,25,26,27,31,33,34,36] \
       + list(np.arange(1,33)) \
       + [2,3,4] \
       + [1,2,3,4,5,7,8,9] + list(np.arange(11,22)) + [24]


navdata["gnss_id"] = np.array(gnss_ids)
navdata["sv_id"] = np.array(sv_ids)


# stanford oval
lat, lon, alt = 37.42984154652992, -122.16946303566934, 0.


# create receiver state
x_rx_m, y_rx_m, z_rx_m = glp.geodetic_to_ecef(np.array([[lat,lon,alt]]))[0]
navdata["x_rx_m"] = x_rx_m
navdata["y_rx_m"] = y_rx_m
navdata["z_rx_m"] = z_rx_m
navdata["b_rx_m"] = 0.

navdata_without_time = navdata.copy()

navdata_full = glp.NavData()
# for random_time in np.linspace(start_gps_millis,end_gps_millis,49):
for random_time in np.linspace(start_gps_millis,end_gps_millis,int(24*60)+1):

    # random_time = np.random.randint(start_gps_millis,end_gps_millis)
    # random_time = np.random.randint(start_gps_millis,end_gps_millis)
    # random_time = 1362865253267
    # print("random")
    # print(random_time)
    # print(glp.gps_millis_to_datetime(random_time))

    navdata_without_time["gps_millis"] = random_time

    navdata_full = navdata_full.concat(navdata_without_time.copy(),axis=1)


navdata = glp.add_sv_states(navdata_full, source="precise",
                                 verbose=True)
glp.add_el_az(navdata,navdata,inplace=True)
navdata = navdata.where("el_sv_deg",10,"geq")

true_pr_m = np.linalg.norm(navdata[["x_rx_m","y_rx_m","z_rx_m"]] -
                             navdata[["x_sv_m","y_sv_m","z_sv_m"]],axis=0)

navdata["corr_pr_m"] = true_pr_m
navdata["raw_pr_m"] = true_pr_m - navdata["b_sv_m"]

navdata.to_csv("data/stanford_oval_20230314.csv")

num_sats = []
for timestamp, _, navdata_subset in navdata.loop_time("gps_millis"):

    # print(len(np.unique(navdata["gnss_sv_id"])))
    num_sats.append(len(np.unique(navdata_subset["gnss_sv_id"])))
    # print(np.unique(navdata["gnss_sv_id"]))
    glp.plot_skyplot(navdata_subset,navdata_subset,prefix=str(int(timestamp)),save=True)
    # plt.show()
    glp.close_figures()

print(num_sats)

# ids = ['C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'C16', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30', 'C32', 'C33', 'C34', 'C35', 'C36', 'C37', 'C38', 'C39', 'C40', 'C41', 'C42', 'C43', 'C44', 'C45', 'C46',
# 'E01', 'E02', 'E03', 'E04', 'E05', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E18', 'E19', 'E21', 'E24', 'E25', 'E26', 'E27', 'E30', 'E31', 'E33', 'E34', 'E36',
# 'G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 'G17', 'G18', 'G19', 'G20', 'G21', 'G22', 'G23', 'G24', 'G25', 'G26', 'G27', 'G28', 'G29', 'G31', 'G32',
# 'J02', 'J03', 'J04',
# 'R01', 'R02', 'R03', 'R04', 'R05', 'R07', 'R08', 'R09', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R24']
