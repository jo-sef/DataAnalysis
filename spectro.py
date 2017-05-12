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
            sub = re.findall("([RrCcAaMm])-",filename)
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