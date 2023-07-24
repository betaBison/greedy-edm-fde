"""Iterate through dataset files.

Currently compatible for:
- Android 2021 (AndroidDerived2021, AndroidGroundTruth2021)
- Android 2022 (AndroidDerived2022, AndroidGroundTruth2022)
- SmartLoc

"""

__authors__ = "D. Knowles"
__date__ = "06 Sep 2022"

import os

import gnss_lib_py as glp

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
        # convert data to Measurement class

        data = glp.SmartLocRaw(trace_path)

        print("smartLoc",trace[0],trace[1])

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
