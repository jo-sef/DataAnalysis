#### This is a copy of sims.py
#### The goal is to make a version based on classes, to better keep track of what is being done.

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import os
import re


def getFile(filename):
    """
    Load data from file into a pandas dataframe
    :param filename: name of sims output file (text format)
    :return: a dataframe of depth,Al Counts, Zn Counts
    """
    print(filename)
    with open(filename, "r") as fh:
        data = pd.read_table(fh, names=["Al_depth", "Al_count", "Zn_depth", "Zn_count"], skiprows=15, delimiter="\t*",
                             skipfooter=140, engine="python")

        x = np.linspace(int(min(data.Al_depth)), int(max(data.Zn_depth)),
                        2 * len(data.Al_depth))

        al_y = interp1d(data.Al_depth, data.Al_count, bounds_error=False,
                        fill_value=np.NaN)(x)
        zn_y = interp1d(data.Zn_depth, data.Zn_count, bounds_error=False,
                        fill_value=np.NaN)(x)
        new_data = {"Depth": x, "Al Counts": al_y, "Zn Counts": zn_y}
        out_data = pd.DataFrame(new_data, columns=["Depth", "Al Counts", "Zn Counts"])

        return out_data


def t_index(data):
    """
    Find the depth from the Zn counts. Depth is where the rate of change in the negative direction is minimum. 
    Above 25 nm because of surface effects
    """

    col = "Zn Counts"
    x = data.dropna().loc[:, "Depth"]
    y = data.dropna().loc[:, col]

    dydx = y.diff().dropna()[x > 25]

    mxi = y.idxmin()

    return mxi

class sims_class:
    """
    
    """

    normalized = False
    # Al = "not normalized yet"
    # Al_df = "not normalized yet"
    filename = None
    data = None
    mxi = None

    window = None
    thickness = None
    raw = None

    def __init__(self, filename):
        ## Raw data ##
        self.filename = filename
        self.data = getFile(self.filename)

        ## thickness and selected portion (75%)
        self.mxi = t_index(self.data)
        window_size = 0.75 * self.mxi
        self.window = (int(self.mxi / 2 - window_size / 2), int(self.mxi / 2 + window_size / 2))
        self.thickness = self.data.iloc[self.mxi]["Depth"]

        ##average Counts
        self.raw = self.data.iloc[self.window[0]:self.window[1]].mean()

        ## Correcting for Zn variations ##
        corrected = self.data["Al Counts"] / self.data["Zn Counts"] * self.raw["Zn Counts"]
        self.data["corrected Al Counts"] = corrected.iloc[self.window[0]:self.window[1]]

    def normalize(self, Zn_standard):
        window = self.data.iloc[self.window[0]:self.window[1]]
        self.data["normalized Al Counts"] = window["corrected Al Counts"] * Zn_standard / self.raw["Zn Counts"]
        self.Al = self.data["normalized Al Counts"].mean()
        self.Al_error = self.data["normalized Al Counts"].std()


def getFolder(folder, figures=False):
    """ create SIMS_data list from all SIMS data in a folder
        return SIMS_data a dictionary with filename, run_no, sub, and data as a df
    """
    if not folder.endswith("/"):
        folder = folder + "/"

    sample_index = "SIMS_sample_index.txt"
    rsf = 4e15

    sims_data = []

    with open(folder + sample_index, "r") as f:
        for lines in f:
            if not lines.startswith("#"):
                search = re.findall("(\S*.dp_rpc_asc)\t(\d+)([aAMmRrCc])", lines)
                if len(search) != 0:
                    filename, run_no, sub = search[0]
                    run_no = int(run_no)

                if len(search) < 1 and re.search("Al_uf", lines):
                    run_no = 0
                    sub = None
                    filename = lines.split()[0]
            file_data = sims_class(folder + filename)

            sims_data.append({"filename": filename, "data": file_data, "run_no": run_no, "sub": sub})

    sims_data = sorted(sims_data, key=lambda k: k["run_no"])
    cal_data = [s["data"] for s in sims_data if s["run_no"] == 0][0]

    Zn_cal = cal_data.raw["Zn Counts"]

    for s in sims_data:
        sims_object = s["data"]
        sims_object.normalize(Zn_cal)
        s.update({"Al_content": sims_object.Al * rsf,
                  "Al_error": sims_object.Al_error * rsf,
                  "SIMS_T": sims_object.thickness})

        #### Save figure for visual check of SIMS data fit
        if figures == True:

            figure_folder = folder + "SIMS_fits/"

            if not os.path.exists(figure_folder):
                os.makedirs(figure_folder)
            save_name = "SIMS_%s%s.png" % (s["run_no"], s["sub"])

            f, ax = plt.subplots(1, 1)
            ax2 = ax.twinx()
            data = sims_object.data
            x = data["Depth"]
            y = data["Zn Counts"]
            y2 = data["Al Counts"] * rsf
            y3 = data["normalized Al Counts"] * rsf
            ax.plot(x, y, label="Zn")
            ax2.plot(x, y2, color="red", label="raw Al")
            ax2.plot(x, y3, color="green", label="normalized Al")
            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")



        window = (data.iloc[sims_object.window[0]]["Depth"], data.iloc[sims_object.window[1]]["Depth"])
        ax.axvline(window[0])
        ax.axvline(window[1])
        f.savefig(figure_folder + save_name)

    return sims_data