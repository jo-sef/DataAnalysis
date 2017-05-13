import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import os
import re

def getFile(filename):
    """
    Load data into a pandas dataframe
    :param filename: name of sims output file (text format)
    :return: a dataframe of depth,Al Counts, Zn Counts, and diff (diff is the difference along Al Counts)
    """

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
        out_data.loc[:, "diff"] = out_data["Al Counts"].diff()

        return out_data

def getFolder(folder):
    """ create SIMS_data list from all SIMS data in a folder
        return SIMS_data a dictionary with filename, run_no, sub, and data as a df
    """
    folder_index = "folderIndex.txt"
    sample_index = "SIMS_sample_index.txt"
    files_in_folder = os.listdir(folder)
    files_to_list = []

    for filename in files_in_folder:
        if re.search(".dp_rpc_asc", filename) and filename != folder_index:
            files_to_list.append(filename)

    list_keys = sorted(files_to_list)

    with open(folder + folder_index, "w") as index_fh:

        for items in list_keys:
            index_fh.write("%s \n" % items)

    SIMS_data = []
    with open(folder+folder_index, 'r') as uv_fh:

        for line in uv_fh:

            filename = line.strip()
            if len(filename)<1:
                continue
            if filename not in [dic["filename"] for dic in SIMS_data]:
                file_data = getFile(folder+filename)
                SIMS_data.append({"filename":filename, "data":file_data})

    with open(folder + sample_index, "r") as fh:
        for lines in fh:
            filename, name = lines.split()
            for dataset in SIMS_data:
                if dataset["filename"] == filename:
                    if name == "Al_uf":
                        name = name
                        sub = None

                    else:
                        sub = name[3]
                        name = name[0:3]
                        dataset.update({"run_no": name, "sub": sub})

    return SIMS_data

def calculate(sims_data):
    for dataset in sims_data:
        data = dataset["data"]

        x = data.dropna().loc[:, "Depth"]
        y = data.dropna().loc[:, "Al Counts"]
        dydx = data.dropna().loc[:, "diff"]

        mxi = y[dydx == min(dydx[x > 25])].index.tolist()[0]
        thick = x[dydx == min(dydx[x > 25])].tolist()[0]
        sampler_window = mxi * 0.95

        lower_sampler_index = int(mxi / 2 - sampler_window / 2)
        upper_sampler_index = int(mxi / 2 + sampler_window / 2)

        al_cont = data.loc[lower_sampler_index:upper_sampler_index, "Al Counts"].mean()
        error = data.loc[lower_sampler_index:upper_sampler_index, "Al Counts"].std()

        dataset.update({"Al_content": float(al_cont), "Al_error": float(error), "thick": float(thick)})

def to_list(samples_data, sims_data):
        for dataset in samples_data:
            for sim in sims_data:
                if samples_data["run_no"] == sim["run_no"] and samples_data["sub"] == sim["sub"]:
                    dataset.update(
                        {"Al_content": sim["Al_content"],
                         "Al_error": sim["Al_error"],
                         "SIMS_T": sim["thick"]})
