####Definitions of hall extractions

HALL_data = []
import logging
import os
import re
import pandas as pd
from pathlib import Path
from .utils.merge import merge_measurements


logger = logging.getLogger(__name__)

###Get results from a single file
def getFile(result_filename, folder="./", thickness_file="AZO_hall_thicknesses_20170314.txt"):
    if not folder.endswith("/"):
        folder = folder + "/"
    if re.search("(\d\d\d)([RrCcMmAa])", result_filename):
        sample, sub = re.findall("(\d\d\d)([RrCcMmAa])", result_filename)[0]

    with open(result_filename,"r",encoding = "latin-1") as fh:
        file_content = fh.read()

        file_thickness = re.findall("Thickness =\s*?\t*?(\d?\d?\d\.\d)",file_content)[0]
        file_thickness = float(file_thickness)

        # read results
        hall_results = {"field":[],"p":[],"coeff":[],"pn":[],"n":[],"mob":[],"temp":[]}
        hall_col = ["field","p","coeff","pn","n","mob","temp"]


        pattern=re.compile('(\d\S+)\s+?\t*?(\S+)\s+?\t*?(\S+)\s+?\t*?([np])\s+?\t*?(\S+)\s+?\t*?(\S+)\s+?\t*?(\S+)')

        for l in file_content.splitlines():
            m = pattern.match(l)

            if m:
                l = pattern.findall(l)[0]

                for i, col in enumerate(hall_col):
                    try:
                        hall_results[col].append(float(l[i]))
                    except ValueError:
                        hall_results[col].append(l[i])
                    except Exception as exc:
                        logger.exception(
                            "Unexpected error parsing column %s in %s: %s",
                            col,
                            result_filename,
                            exc,
                        )
                        raise

        hall_results = pd.DataFrame(hall_results)


        with open(folder + thickness_file) as thickness_fh:
            content = thickness_fh.read()
            thickness_index = re.findall("(\d\d\d)([RrMmCcAa])\t?\s+?(\d\d\d)", content)
            samples_in_index = ["{0}{1}".format(x[0].strip(), x[1].strip()) for x in thickness_index]

            if sample + sub not in samples_in_index:
                print(sample + sub + " not in thickness file %s" % thickness_file)
                return sample, sub, hall_results["p"].mean(), hall_results["mob"].mean(), hall_results[
                    "n"].mean(), file_thickness

        for samples in thickness_index:
            t_sample = samples[0].strip()
            t_sub = samples[1].strip()
            t_thick = float(samples[2])


            if t_sample + t_sub == sample + sub:
                hall_results["c_p"] = hall_results["p"] * file_thickness / t_thick
                hall_results["c_n"] = hall_results["n"] * file_thickness / t_thick
                ##n = constant / d

                return sample, sub, hall_results["c_p"].mean(), hall_results["mob"].mean(),hall_results["c_n"].mean(), t_thick




##get results from all files in folder
def getFolder(folder, thickness_file="AZO_hall_thicknesses_20170314.txt"):
    hall_data = []
    folder = Path(folder)

    files_in_folder = [x for x in os.listdir(folder) if (x.endswith(".txt") and x != thickness_file and x != "folderIndex.txt")]

    for items in files_in_folder:
        run_no_in_name = re.search("(\d\d\d)([RrCcMmAa])", items)
        if not run_no_in_name:
            print(items, "run_no missing")
            continue

        file = items.strip()

        results = getFile(folder / file, folder=str(folder), thickness_file=thickness_file)

        if results == "nothing":
            continue

        sample, sub, resistivity, mobility, carrier_density, thickness = results

        if len(hall_data) > 0:
            if sample + sub in [items["run_no"] + items["sub"] for items in HALL_data]:
                continue
#        print(sample + sub, resistivity, mobility, carrier_density, thickness)

        hall_data.append({"run_no": sample,
                          "sub": sub,
                          "hall_p": resistivity,
                          "hall_mob": mobility,
                          "hall_n": carrier_density,
                          "thickness": thickness})
    else:
    #    print(sample + sub, resistivity, mobility, carrier_density, thickness)
        hall_data.append({"run_no": sample,
                          "sub": sub,
                          "hall_p": resistivity,
                          "hall_mob": mobility,
                          "hall_n": carrier_density,
                          "thickness": thickness})
    return hall_data

def to_list(samples_data, hall_data):
    """Merge hall measurements into the provided sample collection.

    Parameters
    ----------
    samples_data: list of dict
        Sample dictionaries to be updated in-place.
    hall_data: list of dict
        Hall measurement dictionaries.

    Returns
    -------
    list of dict
        The updated ``samples_data`` collection.
    """
    prepared = []
    for h_sample in hall_data:
        if isinstance(h_sample, dict):
            sample = dict(h_sample)
            if "thickness" in sample:
                sample["hall_thick"] = sample.pop("thickness")
            prepared.append(sample)

    return merge_measurements(samples_data, prepared)
