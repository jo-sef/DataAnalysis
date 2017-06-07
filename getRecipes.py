import re
#runs_location = "M:\\PhD Main Folder\\09MOVPE\\"
runs_location = "./"

import zipfile
import xml.etree.cElementTree
import pandas as pd
import numpy as np


def p_press(MO, bub_temp):
    """
    
    :param MO: DEZn, TEGa, tBuOH, or TMAl
    :param bub_temp: temperature of bubbler
    :return: partial pressure in mmHg (Torr)
    """
    gas_con = pd.DataFrame(columns=["DEZn", "tBuOH", "TMAl", "TEGa"], index=["A", "B", "C", "MP"])
    gas_con["DEZn"] = [2109, 8.28, 0, -28]
    gas_con["tBuOH"] = [1080.55, 7.15711, -103, 24]
    gas_con["TMAl"] = [2134, 8.224, 0, 15]
    gas_con["TEGa"] = [2162, 8.083, 0, -82]

    press = 10.0**(gas_con.loc["B",MO]-gas_con.loc["A",MO]/(273.15+float(bub_temp)+gas_con.loc["C",MO]))
    return float(press)

def runDoc(runs_location="./"):
    """
    
    :param runs_location: 
    :return: raw dataframe of runs.docx
    """
    if runs_location[-1]!="/":
        runs_location = runs_location+"/"

    start_row = 670
    WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    PARA = WORD_NAMESPACE + 'p'
    TEXT = WORD_NAMESPACE + 't'
    TABLE = WORD_NAMESPACE + 'tbl'
    ROW = WORD_NAMESPACE + 'tr'
    CELL = WORD_NAMESPACE + 'tc'

    with zipfile.ZipFile(runs_location + "Runs.docx") as docx:
        tree = xml.etree.cElementTree.XML(docx.read('word/document.xml'))

    i = 1
    runs_df = pd.DataFrame()
    header_col = 0
    headers = []
    for table in tree.iter(TABLE):
        for row in table.iter(ROW):
            col_in_row = len(row)
            if header_col < col_in_row:
                header_col = col_in_row

            if i == 1:
                for cell in row.iter(CELL):
                    headers.append(''.join(node.text for node in cell.iter(TEXT)))
                runs_df = pd.DataFrame(columns=headers)

            if i > start_row:
                data = []
                for cell in row.iter(CELL):
                    data.append(''.join(node.text for node in cell.iter(TEXT)))

                data = dict(zip(headers, data))

                row_series = pd.Series(data)
                run_no = row_series["RUN"]
                try:
                    run_no = float(run_no)
                except:
                    continue
                if run_no > start_row:
                    row_series.name = row_series["RUN"].strip()
                    runs_df = runs_df.append(row_series)

            i += 1
    runs_df.columns = ['RUN', 'date', 'Recipe', 'mat', 'Gas', 'DEZn bath',
                               'tBuOH bath', 'DEZn flow', 'DEZn mol',
                               'MO carrier', 'tBuOH flow', 't-BuOH mol',
                               'Gas carrier', 'Heat treat', 'Temp',
                               'Press', 'Rotation', 'Time', 'Substrates',
                               'Front thick', 'Mid thick', 'Back thick',
                               'Ave thick']
    return runs_df

def get_flow(runs_df,precursor, sample):
    """
    
    :param runs_df: runs = runDoc()
    :param precursor: DEZn, TMAl, tBuOH, TEGa, MO_carrier, Gas_carrier
    :param sample: run_no of recipe
    :return: flow of given precursor as float
    """

    if precursor == "tBuOH":
        value = runs_df.loc[sample, "tBuOH flow"]
        try:
            f = float(value.split("/")[0].strip())
            return f
        except:
            return 0
    if precursor == "MO_carrier":
        value = runs_df.loc[sample, "MO carrier"]
        try:
            return float(runs_df.loc[sample, "MO carrier"].replace(",", "."))
        except:
            print(sample+"no MO carrier")

    if precursor == "Gas_carrier":
        gas = runs_df.loc[sample, "Gas carrier"]
        try:
            return float(gas)
        except:
            print(gas)

    if precursor == "TMAl":
        for cols in runs_df.loc[sample]:
            if re.search("TMAl", str(cols)):
                f = re.findall("TMAl\s?\d+\.[0-9].*/(\d+\.?\d?)\s?sccm", cols)
                if len(f) > 0:
                    return float(f[0])
    if precursor == "TEGa":
        for cols in runs_df.loc[sample]:
            if re.search("TEGa", str(cols)):
                f = re.findall("TEGa\s?\d+\.[0-9].*/(\d+\.?\d?)\s?sccm", cols)
                if len(f) > 0:
                    return float(f[0])
    if precursor == "DEZn":
        try:
            d = float(runs_df.loc[sample, "DEZn flow"].split("/")[0].strip().replace(",", "."))
            return d
        except:
            return 0
    else:
        # print(sample+" precursor not found "+precursor)
        return 0

def get_temp(runs,precursor, sample):
    """
    
    :param runs: 
    :param precursor: 
    :param sample: 
    :return: temperature of bubbler in C
    """
    if precursor == "tBuOH":
        try:
            return float(runs["tBuOH bath"][sample].split("/")[0].strip().replace(",", "."))
        except:
            return 0
    if precursor == "TMAl":
        for cols in runs.loc[sample]:
            if re.search("TMAl", str(cols)):
                f = re.findall("TMAl\s?(\d+\.[0-9]).*/\d+\.?\d?\s?sccm", cols)
                if len(f) > 0:
                    return float(f[0])

    if precursor == "TEGa":
        for cols in runs.loc[sample]:
            if re.search("TEGa", str(cols)):
                f = re.findall("TEGa\s?(\d+\.[0-9]).*/\d+\.?\d?\s?sccm", cols)
                if len(f) > 0:
                    return float(f[0])
    if precursor == "DEZn":
        temp = runs["DEZn bath"][sample]
        if type(temp) == "str":
            temp = temp.replace(",", ".")
        try:
            return float(temp)
        except:
            print(sample)
    else:
        # print(sample + " no temp "+precursor)
        return 0

def get_runs(run_location):
    """Read runs.docx and return a pandas dataframe with the columns:
    'RUN', 'mat', 'DEZn flow', 'DEZn temp','tBuOH flow', 'tBuOH temp', "TMAl flow",
    "TMAl temp", "TEGa flow", "TEGa temp",'MO_carrier', "Gas_carrier"
    
    run_location contains path and name of folder containing runs.docx
    """
    run_df = runDoc(run_location)

    new_runs = {"TEGa flow": [get_flow(run_df,"TEGa", samples.strip()) for samples in run_df.loc[:, "RUN"]],
                "TEGa temp": [get_temp(run_df,"TEGa", samples.strip()) for samples in run_df.loc[:, "RUN"]],
                "TMAl flow": [get_flow(run_df,"TMAl", samples.strip()) for samples in run_df.loc[:, "RUN"]],
                "TMAl temp": [get_temp(run_df,"TMAl", samples.strip()) for samples in run_df.loc[:, "RUN"]],
                "DEZn flow": [get_flow(run_df,"DEZn", samples.strip()) for samples in run_df.loc[:, "RUN"]],
                "DEZn temp": [get_temp(run_df,"DEZn", samples.strip()) for samples in run_df.loc[:, "RUN"]],
                "tBuOH flow": [get_flow(run_df,"tBuOH", samples.strip()) for samples in run_df.loc[:, "RUN"]],
                "tBuOH temp": [get_temp(run_df,"tBuOH", samples.strip()) for samples in run_df.loc[:, "RUN"]],
                "MO_carrier": [get_flow(run_df,"MO_carrier", samples.strip()) for samples in run_df.loc[:, "RUN"]],
                "Gas_carrier": [get_flow(run_df,"Gas_carrier", samples.strip()) for samples in run_df.loc[:, "RUN"]],
                "mat": run_df.loc[:, "mat"],
                "RUN": run_df.loc[:, "RUN"].str.strip()}

    new_runs_df = pd.DataFrame(new_runs, columns=['RUN', 'mat', 'DEZn flow', 'DEZn temp',
                                           'tBuOH flow', 'tBuOH temp', "TMAl flow",
                                           "TMAl temp", "TEGa flow", "TEGa temp",
                                           'MO_carrier', "Gas_carrier"])
    return new_runs_df

def add_flows(a_list,runs):
    """ Add flows from runs to a_list. runs is obtained by importing getRecipes : 
    runs = getRecipes.get_runs(runs_location)
    """

    for sample in a_list:
        run_no = sample["run_no"]

        if run_no is None:
            continue

        TMAl_flow = runs.loc[run_no, "TMAl flow"]
        tBuOH_flow = runs.loc[run_no, "tBuOH flow"]
        MO_carrier = runs.loc[run_no, "MO_carrier"]
        DEZn_flow = runs.loc[run_no, "DEZn flow"]
        TEGa_flow = runs.loc[run_no, "TEGa flow"]
        Gas_carrier = runs.loc[run_no, "Gas_carrier"]

        p_DEZn = p_press("DEZn", runs.loc[run_no,"DEZn temp"])
        p_TMAl = p_press("TMAl", runs.loc[run_no,"TMAl temp"])
        p_tBuOH = p_press("tBuOH", runs.loc[run_no, "tBuOH temp"])
        p_TEGa = p_press("TEGa", runs.loc[run_no, "TEGa temp"])


        if float(TMAl_flow) != 0 or float(TEGa_flow) != 0:
            MO_total = 2000 + MO_carrier * 1000
        else:
            MO_total = 1000 + MO_carrier * 1000

        Gas_total = Gas_carrier * 1000 + 1000

        DEZn_molar_flow = DEZn_flow / 22400 * p_DEZn / (900 - p_DEZn)
        TMAl_molar_flow = TMAl_flow/ 22400 * p_TMAl / (900 - p_TMAl)
        TEGa_molar_flow = TEGa_flow/ 22400 * p_TEGa / (900 - p_TEGa)
        O_molar_flow = tBuOH_flow / 22400 * p_tBuOH / (900 - p_tBuOH)

        if DEZn_flow != 0 and O_molar_flow != 0:
            vi_ii = O_molar_flow /DEZn_molar_flow * MO_total / Gas_total
        else:
            vi_ii = np.nan

        if TMAl_flow != 0 or TEGa_flow !=0:
            vi_mo = O_molar_flow/(DEZn_molar_flow+TEGa_molar_flow+TMAl_molar_flow) * MO_total / Gas_total




        sample.update({"DEZn molar": DEZn_molar_flow,
                       "TEGa molar": TEGa_molar_flow,
                       "TMAl molar": TMAl_molar_flow,
                       "tBuOH molar": O_molar_flow,
                       "vi_ii": vi_ii,
                        "vi_mo":vi_mo})


        sample.update({'TMAl_flow': float(TMAl_flow),
                       'tBuOH_flow': float(tBuOH_flow),
                       "MO_carrier": float(MO_carrier),
                       "DEZn_flow":float(DEZn_flow),
                       "TEGa_flow":float(TEGa_flow),
                       "MO_carrier": float(MO_carrier),
                       "Gas_carrier": float(Gas_carrier)})

def make_samples_data(start_run=688, end_run=722):
    """
    :param start_run: lower limit of run_no's included
    :param end_run: upper limit of run_no's included
    :return: list of dictionaries that can serve as samples_data
    """
    samples_data = []

    first_sample = start_run
    last_sample = end_run
    samples_with_AM_substrates = [717,718]

    if len(samples_data) == 0:
        samples_data = []
        for sub in ["R", "C"]:
            for x in range(first_sample, last_sample+1, 1):
                samples_data.append({"run_no": str(x),
                                     "sub": sub})
        for sub in ["A", "M"]:
            for x in samples_with_AM_substrates:
                samples_data.append({"run_no": str(x),
                                     "sub": sub})

    samples_data = sorted(samples_data, key=lambda k: k["run_no"])

    return samples_data

