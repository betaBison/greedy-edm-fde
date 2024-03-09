"""Iterate through dataset files.

Currently compatible for:
- Android 2023 (AndroidDerived2023, AndroidGroundTruth2023)
- Android 2021 (AndroidDerived2021, AndroidGroundTruth2021)
- Android 2022 (AndroidDerived2022, AndroidGroundTruth2022)
- SmartLoc

"""

__authors__ = "D. Knowles"
__date__ = "06 Sep 2022"

import os

import numpy as np
import gnss_lib_py as glp

class Android2023Iterator():

    def __init__ (self, train_path, load_gt=True,
                  load_raw=False, filter_measurements=True):
        """Load Android 2023 Dataset

        Parameters
        ----------
        train_path : string or path-like
            Path to train folder of data
        load_gt : bool
            If true, will load ground truth data.
        load_raw : bool
            If true, will load raw data.

        """
        self.train_path = train_path
        self.load_gt = load_gt
        self.load_raw = load_raw
        self.filter_measurements = filter_measurements

    def run(self, trace, derived=None, ground_truth=None, raw=None):
        """Run function to overwrite.

        trace : list
            Name of data/place and then name of phone.
        derived : gnss_lib_py.parsers.google_decimeter.AndroidDerived2022
            Derived data.
        ground_truth : gnss_lib_py.parsers.google_decimeter.AndroidGroundTruth2022
            Ground truth data.
        raw : gnss_lib_py.parsers.google_decimeter.AndroidRawGnss
            Raw data.

        """
        raise NotImplementedError("must overwrite run function")

    def single_run(self, trace):
        """Load and run single trace.

        Parameters
        ----------
        trace : list
            Includes trace run and phone name as strings.

        """
        data_path = self.train_path

        trace_path = os.path.join(data_path,trace[0],trace[1],"device_gnss.csv")
        # convert data to Measurement class
        derived_data = glp.AndroidDerived2023(trace_path)

        if self.load_gt:
            gt_path = os.path.join(data_path,trace[0],trace[1],"ground_truth.csv")
            gt_data = glp.AndroidGroundTruth2023(gt_path)
        else:
            gt_data = None

        if self.load_raw:
            raw_path = os.path.join(data_path,trace[0],trace[1],
                                    "supplemental","gnss_log.txt")
            raw_data = glp.AndroidRawGnss(raw_path,
                                      filter_measurements=self.filter_measurements)
        else:
            raw_data = None

        print("2023",trace[0],trace[1])

        output = self.run(trace, derived_data, gt_data, raw_data)

        return output

    def iterate(self):
        """Iterate over entire dataset.

        Calls ``run`` function on each derived/ground truth pair.

        """
        data_path = self.train_path

        # get all trace options
        trace_names = sorted(os.listdir(data_path))
        # create a list of all traces with phone types
        trace_list = []
        for trace_name in trace_names:
            trace_path = os.path.join(data_path,trace_name)
            if not os.path.isdir(trace_path):
                continue
            for phone_type in sorted(os.listdir(trace_path)):
                if phone_type in ("mi8",
                                  "sm-g988b",
                                  "sm-g955f",
                                  "samsungs21ultra",
                                  "pixel6pro",
                                  "pixel7",
                                  "pixel7pro",
                                  ):
                    trace_list.append((trace_name,phone_type))

        outputs = []
        for trace_idx, trace in enumerate(trace_list):
            print(trace_idx+1,"/",len(trace_list))
            output = self.single_run(trace)
            outputs.append(output)

        return outputs

class Android2021Iterator():

    def __init__ (self, train_path):
        """Load Android 2021 Dataset

        Parameters
        ----------
        train_path : string or path-like
            Path to train folder of data

        """
        self.train_path = train_path

    def run(self, derived, ground_truth, trace):
        """Run function to overwrite.

        derived : gnss_lib_py.parsers.android.AndroidDerived2021
            Derived data.
        ground_truth : gnss_lib_py.parsers.android.AndroidGroundTruth2021
            Ground truth data.
        trace : list
            Name of data/place and then name of phone.

        """
        raise NotImplementedError("must overwrite run function")

    def single_run(self,trace):
        """Load and run single trace.

        Parameters
        ----------
        trace : list
            Includes trace run and phone name as strings.

        """
        data_path = self.train_path

        trace_path = os.path.join(data_path,trace[0],trace[1],trace[1]+"_derived.csv")
        gt_path = os.path.join(data_path,trace[0],trace[1],"ground_truth.csv")

        # convert data to Measurement class
        derived_data = glp.AndroidDerived2021(trace_path)
        gt_data = glp.AndroidGroundTruth2021(gt_path)

        print("2021",trace[0],trace[1])

        output = self.run(derived_data, gt_data, trace)

        return output

    def iterate(self):
        """Iterate over entire dataset.

        Calls ``run`` function on each derived/ground truth pair.

        """
        data_path = self.train_path

        # get all trace options
        trace_names = sorted(os.listdir(data_path))

        # create a list of all traces with phone types
        trace_list = []
        for trace_name in trace_names:
            trace_path = os.path.join(data_path,trace_name)
            for phone_type in sorted(os.listdir(trace_path)):
                trace_list.append((trace_name,phone_type))

        outputs = []
        for trace_idx, trace in enumerate(trace_list):
            print(trace_idx+1,"/",len(trace_list))
            output = self.single_run(trace)
            outputs.append(output)

        return output

class Android2022Iterator():

    def __init__ (self, train_path, load_gt=True):
        """Load Android 2022 Dataset

        Parameters
        ----------
        train_path : string or path-like
            Path to train folder of data

        """
        self.train_path = train_path
        self.load_gt = load_gt

    def run(self, derived, ground_truth, trace):
        """Run function to overwrite.

        derived : gnss_lib_py.parsers.android.AndroidDerived2022
            Derived data.
        ground_truth : gnss_lib_py.parsers.android.AndroidGroundTruth2022
            Ground truth data.
        trace : list
            Name of data/place and then name of phone.

        """
        raise NotImplementedError("must overwrite run function")

    def single_run(self, trace):
        """Load and run single trace.

        Parameters
        ----------
        trace : list
            Includes trace run and phone name as strings.

        """
        data_path = self.train_path

        trace_path = os.path.join(data_path,trace[0],trace[1],"device_gnss.csv")
        # convert data to Measurement class
        derived_data = glp.AndroidDerived2022(trace_path)

        if self.load_gt:
            gt_path = os.path.join(data_path,trace[0],trace[1],"ground_truth.csv")
            gt_data = glp.AndroidGroundTruth2022(gt_path)
        else:
            gt_data = None


        print("2022",trace[0],trace[1])

        output = self.run(derived_data, gt_data, trace)

        return output

    def iterate(self):
        """Iterate over entire dataset.

        Calls ``run`` function on each derived/ground truth pair.

        """
        data_path = self.train_path

        # get all trace options
        trace_names = sorted(os.listdir(data_path))

        # create a list of all traces with phone types
        trace_list = []
        for trace_name in trace_names:
            trace_path = os.path.join(data_path,trace_name)
            for phone_type in sorted(os.listdir(trace_path)):
                if phone_type in ("XiaomiMi8","GooglePixel6Pro",
                                  "SamsungGalaxyS20Ultra"):
                    trace_list.append((trace_name,phone_type))

        outputs = []
        for trace_idx, trace in enumerate(trace_list):
            print(trace_idx+1,"/",len(trace_list))
            output = self.single_run(trace)
            outputs.append(output)

        return outputs

class SmartLocIterator():

    def __init__ (self, train_path):
        """Load SmartLoc Dataset

        Expected folder structure:

        train_path/
           |-- berlin1_potsdamer_platz/
                |-- RXM-RAWX.csv
           |-- berlin2_gendarmenmarkt/
                |-- RXM-RAWX.csv
           |-- frankfurt1_maintower/
                |-- RXM-RAWX.csv
           |-- frankfurt2_westendtower/
                |-- RXM-RAWX.csv

        Parameters
        ----------
        train_path : string or path-like
            Path to train folder of data

        """

        self.train_path = train_path

    def run(self, data, gt, trace):
        """Run function to overwrite.

        data : gnss_lib_py.parsers.smart_loc.SmartLocRaw
            Raw data.
        gt : gnss_lib_py.parsers.smart_loc.SmartLocRaw
            Ground truth
        trace : list
            Name of data/place.

        """
        raise NotImplementedError("must overwrite run function")

    def single_run(self, trace):
        """Load and run single trace.

        Parameters
        ----------
        trace : list
            Includes trace run and phone name as strings.

        """
        data_path = self.train_path

        trace_path = os.path.join(data_path,trace[0],trace[1])

        print("smartLoc",trace[0],trace[1])

        # convert data to Measurement class
        data = glp.SmartLocRaw(trace_path)
        # data = data.where("NLOS (0 == no, 1 == yes, 2 == No Information)",0)
        data["b_sv_m"] = 0

        data["gps_millis_original"] = data["gps_millis"]

        print("adding biases")
        for i in range(1):
            print("loop:",i)

            data["gps_millis"] = data["gps_millis_original"] \
                               - np.round((data["raw_pr_m"]/consts.C)*1E3,-2) \
                               # + np.round((data["b_sv_m"]/consts.C)*1E3,-2)
            # data["gps_millis"] -= np.round((wls_state_estimate["rx_wls_b_m"]/consts.C)*1E3,-2)

            if i == 0:
                data.rename({"NLOS (0 == no, 1 == yes, 2 == No Information)":"fault_gt"},
                            inplace=True)

            print("adding smartLoc SV states")
            data = glp.add_sv_states(data,
                                     verbose=True)

            print("post add",np.mean(data["b_sv_m"]))
            print("post add",np.max(data["b_sv_m"]))

            # add corrected pseudorange
            data["corr_pr_m"] = data["raw_pr_m"] + data["b_sv_m"]

            # add ECEF coordinates
            ecef_xyz = glp.geodetic_to_ecef(data[["lat_rx_gt_deg",
                                                  "lon_rx_gt_deg",
                                                  "alt_rx_gt_m"
                                                ]])
            data["x_rx_gt_m"] = ecef_xyz[0,:]
            data["y_rx_gt_m"] = ecef_xyz[1,:]
            data["z_rx_gt_m"] = ecef_xyz[2,:]
            data.to_csv("example_smartloc_pre_wls.csv")

            # estimate receiver clock bias
            print("solving WLS for receiver clock bias")
            wls_state_estimate = glp.solve_wls(data,
                                               only_bias = True,
                                               # delta_t_decimals=6,
                                               receiver_state=data,
                                               )

            print("post rx",np.mean(wls_state_estimate["b_rx_wls_m"]))
            print("post rx",np.max(wls_state_estimate["b_rx_wls_m"]))

        print("calculating residuals")
        glp.solve_residuals(data,wls_state_estimate)

        data.to_csv("example_smartloc.csv")

        output = self.run(data, data.copy(), trace)

        return output

    def iterate(self):
        """Iterate over entire dataset.

        Calls ``run`` function on each derived/ground truth pair.

        """
        data_path = self.train_path

        # get all trace options
        trace_names = sorted(os.listdir(data_path))

        # create a list of all traces with phone types
        trace_list = []
        for trace_name in trace_names:
            trace_list.append((trace_name,"RXM-RAWX.csv"))

        outputs = []
        for trace_idx, trace in enumerate(trace_list):
            print(trace_idx+1,"/",len(trace_list))
            output = self.single_run(trace)
            outputs.append(output)

        return outputs
