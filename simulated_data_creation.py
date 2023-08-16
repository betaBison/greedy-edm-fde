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



locations = {
              "stanford_oval" : (37.42984154652992, -122.16946303566934, 0.),
              "munich" : (48.16985710449595, 11.551627945697028, 0.),
              "london" : (51.5097085796586, -0.16008158973060102, 0.),
              "hong_kong" : (22.327793473417067, 114.17122448832379, 0.),
              "zurich" : (47.407491810621345, 8.500756183071228, 0.),
              "cape_town" : (-33.91700025297494, 18.403910329181112, 0.),
              "calgary" : (51.11056458625996, -114.1179704693596, 0.),
              "sydney" : (-33.859749976799186, 151.22208557691505, 0.),
              "sao_paulo" : (-23.568026105263545, -46.736620380100675, 0.),
             }

for location_name, location_tuple in locations.items():
    print("location name:",location_name)
    lat, lon, alt = location_tuple

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

    # create receiver state
    x_rx_m, y_rx_m, z_rx_m = glp.geodetic_to_ecef(np.array([[lat,lon,alt]]))[0]
    navdata["x_rx_m"] = x_rx_m
    navdata["y_rx_m"] = y_rx_m
    navdata["z_rx_m"] = z_rx_m
    navdata["b_rx_m"] = 0.




    navdata_without_time = navdata.copy()

    navdata_full = glp.NavData()
    # for random_time in np.linspace(start_gps_millis,end_gps_millis,49):
    print("concatenating times")
    for random_time in np.linspace(start_gps_millis,end_gps_millis,int(24*60)+1):

        # random_time = np.random.randint(start_gps_millis,end_gps_millis)
        # random_time = np.random.randint(start_gps_millis,end_gps_millis)
        # random_time = 1362865253267
        # print("random")
        # print(random_time)
        # print(glp.gps_millis_to_datetime(random_time))

        navdata_without_time["gps_millis"] = random_time

        navdata_full = navdata_full.concat(navdata_without_time.copy(),axis=1)


    print("adding sv times")
    navdata = glp.add_sv_states(navdata_full, source="precise",
                                     verbose=True)
    print("adding el az")
    glp.add_el_az(navdata,navdata,inplace=True)
    navdata = navdata.where("el_sv_deg",10,"geq")

    true_pr_m = np.linalg.norm(navdata[["x_rx_m","y_rx_m","z_rx_m"]] -
                                 navdata[["x_sv_m","y_sv_m","z_sv_m"]],axis=0)

    navdata["corr_pr_m"] = true_pr_m + np.random.normal(loc=0.0,scale=10,size=len(navdata))
    navdata["raw_pr_m"] = navdata["corr_pr_m"] - navdata["b_sv_m"]

    print("saving csv")
    navdata.to_csv("data/simulated/" + location_name + "_20230314.csv")

    num_sats = []
    for timestamp, _, navdata_subset in navdata.loop_time("gps_millis"):

        # print(len(np.unique(navdata["gnss_sv_id"])))
        num_sats.append(len(np.unique(navdata_subset["gnss_sv_id"])))
        # print(np.unique(navdata["gnss_sv_id"]))
        # glp.plot_skyplot(navdata_subset,navdata_subset,prefix=str(int(timestamp)),save=True)
        # plt.show()
        # glp.close_figures()

    print(num_sats)
    num_sats = np.array(num_sats)
    print(np.min(num_sats),np.mean(num_sats),np.max(num_sats))
