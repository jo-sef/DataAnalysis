import pandas as pd
import numpy as np
import re
import pycap as pc
import os
import matplotlib.pyplot as plt
from lmfit.models import LinearModel

"""Load data from files in a folder into a list of dictionaries with data extracted from filename"""
def correcting_for_hump(in_panda):
    ##difference
    # at 991-1009
    hump_min = 1010
    hump_max = 1592
    step_end = 1604

    step_start = 992

    l_step = in_panda.index[in_panda.loc[:, "wl"] == step_start][0]
    l_hump = in_panda.index[in_panda.loc[:, "wl"] == hump_min][0]

    sl = in_panda.loc[l_step, "T"]
    el = in_panda.loc[l_hump, "T"]
    # start_i = in_panda.index[]
    dl = el - sl

    # at 1591-1009
    u_hump = in_panda.index[in_panda.loc[:, "wl"] == hump_max][0]
    u_step = in_panda.index[in_panda.loc[:, "wl"] == step_end][0]

    su = in_panda.loc[u_hump, "T"]
    eu = in_panda.loc[u_step, "T"]
    du = su - eu

    t = in_panda.copy().loc[l_step:u_step, :]

    ##start and end of hump

    hump_rows = u_hump - l_hump

    for i in range(hump_rows):
        d = dl + i / hump_rows * (du - dl)
        # val = t.loc[l_hump+i,"T"]
        t.loc[l_hump + i, "T"] = t.loc[l_hump + i, "T"] - d

    ##getting the end spikes:
    rows = l_hump - l_step

    for i in range(rows):
        val = t.loc[l_step + i, "T"]
        t.loc[l_step + i] = val - i / rows * dl

    rows = u_step - u_hump

    for i in range(rows):
        val = t.loc[u_hump + i, "T"]
        t.loc[u_hump + i] = val - (1 - i / rows) * du

    corrected = in_panda.copy()

    corrected.loc[l_step:u_step] = t.loc[:]

    return corrected["T"]
class uvSample:
    import pycap as pc
    """Read a UV sample file, and perform some basic operations on the data:
        1. Make a dataframe with the data.
        2. Calculate the absorption coefficient
        3. Calculate the derivative of the absorption coefficient
        4. Perform Tauc analysis.
    """
    def correcting_for_hump(self,in_panda):
        ##difference
        # at 991-1009
        hump_min = 1010
        hump_max = 1592
        step_end = 1604

        step_start = 992

        l_step = in_panda.index[in_panda.loc[:, "wl"] == step_start][0]
        l_hump = in_panda.index[in_panda.loc[:, "wl"] == hump_min][0]

        sl = in_panda.loc[l_step, "T"]
        el = in_panda.loc[l_hump, "T"]
        # start_i = in_panda.index[]
        dl = el - sl

        # at 1591-1009
        u_hump = in_panda.index[in_panda.loc[:, "wl"] == hump_max][0]
        u_step = in_panda.index[in_panda.loc[:, "wl"] == step_end][0]

        su = in_panda.loc[u_hump, "T"]
        eu = in_panda.loc[u_step, "T"]
        du = su - eu

        t = in_panda.copy().loc[l_step:u_step, :]

        ##start and end of hump

        hump_rows = u_hump - l_hump

        for i in range(hump_rows):
            d = dl + i / hump_rows * (du - dl)
            # val = t.loc[l_hump+i,"T"]
            t.loc[l_hump + i, "T"] = t.loc[l_hump + i, "T"] - d

        ##getting the end spikes:
        rows = l_hump - l_step

        for i in range(rows):
            val = t.loc[l_step + i, "T"]
            t.loc[l_step + i] = val - i / rows * dl

        rows = u_step - u_hump

        for i in range(rows):
            val = t.loc[u_hump + i, "T"]
            t.loc[u_hump + i] = val - (1 - i / rows) * du

        corrected = in_panda.copy()

        corrected.loc[l_step:u_step] = t.loc[:]

        return corrected["T"]

    def readUVfile(self,filename):
        """ Read the Angles and intensities off the file. Assume that the intensity of the AL2O3 peak is constant for all samples because
        the films are thin. Then we can normalise the intensities."""

        data = pd.read_table(filename, skiprows=2, names=["wl", "T"])

        data.loc[:, "c_T"] = self.correcting_for_hump(data)
        data.loc[:, "hv"] = 1240 / data.loc[:, "wl"]
        data.loc[:, "alpha"] = -np.log(data.loc[:, "c_T"] / 100)
        data.loc[:, "Tauc"] = (data.loc[:, "alpha"] * data.loc[:, "hv"]) ** 2
        data.loc[:, "d_alpha"] = self.pc.functions.derivative.smoothed(data.loc[:, "hv"], data.loc[:, "alpha"])
        data.loc[:, "d_Tauc"] = self.pc.functions.derivative.smoothed(data.loc[:, "hv"], data.loc[:, "Tauc"])
        return data
    def runFromName(self):
        try:
            run_no,sub =re.findall("(\d\d\d)([rRCcaAmM])", self.filename)[0]
            self.sub = sub.upper()
            self.run_no = int(run_no)
        except:
            if re.search("[Aa][Ii][Rr]", self.filename):
                run_no = None
                sub = None
                self.sub =None

            if re.search("[Ss][Uu][bB]", self.filename):
                run_no = None
                sub = re.findall("([RrCcAaMm])-",self.filename)[0]
                self.sub = sub.upper()
            else:
                print(self.filename)

            self.run_no = 0
            return("run: %i sub: %s" %(self.run_no,self.sub))
    def nan_helper(self,y):
        """Helper to handle indices and logical indices of NaNs.

        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
              to convert logical indices of NaNs to 'equivalent' indices
        Example:
            #>>> # linear interpolation of NaNs
            #>>> nans, x= nan_helper(y)
            #>>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        """

        return np.isnan(y), lambda z: z.nonzero()[0]

    def __init__(self,filename):
        self.filename = filename
        self.data = self.readUVfile(filename)
        self.runFromName()



def getFile(filename):
    """ Get UV data from file from text file. calculate tauc, derivatives, correction of Transmission, and
    absorption coefficient
    """
    data = pd.read_table(filename, skiprows=2, names=["wl", "T"])

    data.loc[:, "c_T"] = correcting_for_hump(data)
    data.loc[:, "hv"] = 1240 / data.loc[:, "wl"]
    data.loc[:, "alpha"] = -np.log(data.loc[:, "c_T"] / 100)
    data.loc[:, "Tauc"] = (data.loc[:, "alpha"] * data.loc[:, "hv"]) ** 2
    data.loc[:, "d_alpha"] = pc.functions.derivative.smoothed(data.loc[:, "hv"], data.loc[:, "alpha"])
    data.loc[:, "d_Tauc"] = pc.functions.derivative.smoothed(data.loc[:, "hv"], data.loc[:, "Tauc"])

    try:
        run_no, sub = re.findall("(\d\d\d)([AaMmCcRr])", filename)[0]
    except:
        if re.search("[Aa][Ii][Rr]", filename):
            run_no = None
            sub = None
        if re.search("[Ss][Uu][bB]", filename):
            run_no = None
            sub = re.findall("([RrCcAaMm])-",filename)[0]
    dataset = {"run_no":run_no,"sub":sub, "data":data}
    return dataset

def getFolder(folder):
    """ create UV_data list from all UV-vis in a folder
    """
    folder_index = "folderIndex.txt"
    files_in_folder = os.listdir(folder)
    files_to_list = []
    for filename in files_in_folder:
        if re.search(".txt", filename) and filename != folder_index:
            files_to_list.append(filename)

    list_keys = sorted(files_to_list)

    with open(folder + folder_index, "w") as index_fh:

        for items in list_keys:
            index_fh.write("%s \n" % items)


    UV_data = []
    with open(folder+folder_index, 'r') as uv_fh:

        for line in uv_fh:

            filename = line.strip()
            if len(filename)<1:
                continue

            UV_data.append(getFile(folder+filename))
    return UV_data

def taucplot(dataset, UV_folder="./AZO_2016_UV/", plot = False):
    """Fit tauc plot according to highest d_tauc/d_hv and a few points above and below.

    """
    report_folder = UV_folder+"UV_fit/"
    if not os.path.exists(report_folder):
        os.makedirs(report_folder)

    ################ pull out variables from dataset ########################
    run_no = dataset["run_no"]
    sub = dataset["sub"]
    data = dataset["data"]
    data = data[(data["hv"] > 3) & (data["hv"] < 4.5)]
    x = data["hv"]
    y = data["Tauc"]
    max_x = np.inf
    if run_no == "706" and sub == "C":
        max_x =4
    max_a = max(data["d_Tauc"][x<max_x])
    mxi = data[data["d_Tauc"] == max_a].index.values[0]

    ################ Setup points to fit #####################################
    xlim = 5
    xliml = 2
    if run_no in ["719","720"]:
        xlim=1
        xliml=1
    points = data.loc[mxi - xliml:mxi+xlim, ["hv", "Tauc", "d_Tauc"]]
    px = np.array(points["hv"])
    py = np.array(points["Tauc"])

    ####Setup model and variables
    line = LinearModel()
    pars = line.guess(py, x=px)
    pars["slope"].set(value=max_a, vary=True)

    init = line.eval(pars, x=px)
    out = line.fit(py, pars, x=px)

    a = out.best_values["slope"]
    b = out.best_values["intercept"]
    bandgap = -b / a
    dataset.update({"t_bandgap": bandgap})

    #################### plotting and recording results from fit ####################
    with open(report_folder + "UV_fit_%s%s.txt" % (run_no, sub), "w") as report_fh:
        report_fh.write(out.fit_report())

    f, ax = plt.subplots(1, figsize=(4, 4))

    ax.plot(x, y, label="data")
    ax.plot(px, init, color="k", ls="--", label="initial guess")
    ax.plot(x, a * x + b, color="r", label="best fit", lw=0.5)
    ax.legend(loc = "lower right")

    ax.annotate("Bandgap: \n %1.3f eV" % bandgap, xy=(0.4, 0.4), xycoords='axes fraction', fontsize=16,
                horizontalalignment='right', verticalalignment='bottom')

    ax.set_ylim(0, max(data.loc[:, "Tauc"]))
    ax.set_xlim(3.2, 4.5)
    ax.set_ylabel(r"($\alpha$h$\nu$)$^2$", fontsize=16)
    ax.set_xlabel("(eV)", fontsize=16)
    f.suptitle("Tauc plot %s %s" % (run_no, sub), fontsize=16)
    f.savefig(report_folder + "Tauc_%s%s.png" % (run_no, sub), dpi=300,bbox_fit = "tight")
    if plot == False:
        plt.close()
    else:
       plt.show()

def taucfile(t_file):
    """
    :param t_file: contains the fit report produced by the taucplot function
    :return: a dictionary with run_no, substrate and calculated bandgap
    (provided the run_no and substrate is part of the filename)
    """
    run_no,sub = re.findall("(\d\d\d)([aAmMcCrR])", t_file)[0]
    with open(t_file) as fin:
        content = fin.read()
        try:
            slope = re.findall("slope:\s*(\d+.\d+)", content)[0]
            intercept = re.findall("intercept:\s+(\S\d+\.\d+)", content)[0]
            bandgap = -float(intercept)/float(slope)
        except:
            print("slope and intercept not found in file: "+t_file)
    return {"t_bandgap": bandgap, "run_no": run_no, "sub": sub}

def taucfolder(t_folder):
    """
    :param t_folder: folder with fit reports (.txt) files
    :return: a dictionary with calculated bandgap from all of them
    """
    if not t_folder.endswith("/"):
        t_folder = t_folder+"/"
    uv_data = []
    files_in_folder = [ file for file in os.listdir(t_folder) if re.search("UV_fit_\d\d\d[mMaArRCc].txt", file)]
    for filename in files_in_folder:
        run_no, sub = re.findall("(\d\d\d)([aAmMcCrR])", filename)[0]
        results = taucfile(t_folder+filename)
        if len(uv_data)>0 and run_no+sub in [s["run_no"]+s["sub"] for s in uv_data]:
            continue
        uv_data.append(results)
    return uv_data

def to_list(samples_data, uv_data):
    for dataset in samples_data:
        for u_sample in uv_data:
            if dataset["run_no"] == u_sample["run_no"] and dataset["sub"] == u_sample["sub"]:
                dataset.update(
                    {"t_bandgap": u_sample["t_bandgap"]})
