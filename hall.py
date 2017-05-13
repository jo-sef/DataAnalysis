####Definitions of hall extractions

HALL_data = []
import os
import re

###Get results from a single file
def getFile(result_filename, folder="./", thickness_file="AZO_hall_thicknesses_20170314.txt"):
    if not folder.endswith("/"):
        folder = folder + "/"
    if re.search("(\d\d\d)([RrCcMmAa])", result_filename):
        sample, sub = re.findall("(\d\d\d)([RrCcMmAa])", result_filename)[0]

        thickness_in_filename = re.search("(\d?\d\d)nm", result_filename)

        if thickness_in_filename:
            file_thickness = float(re.findall("(\d?\d\d)nm", result_filename)[0])
        else:
            file_thickness = 400

        with open(result_filename, "r", encoding = "latin-1") as result_fh:

            for line in result_fh:

                if line.startswith("<Step 3: Variable Field Measurement>"):
                    for i in range(23):
                        next(result_fh)
                    break

            carrier_density = 0.0
            mobility = 0.0
            resistivity = 0.0
            i = 0
            for line in result_fh:
                if len(line) > 2:
                    new_density = float(line.split()[4].strip())
                    new_mobility = float(line.split()[5].strip())
                    new_resistivity = float(line.split()[1].strip())

                    carrier_density += new_density
                    mobility += new_mobility
                    resistivity += new_resistivity

                    i += 1
            ###get averages
            av_resistivity = resistivity / i
            av_mobility = mobility / i
            av_carrier_density = carrier_density / i

            with open(folder + thickness_file) as thickness_fh:
                content = thickness_fh.read()
                thickness_index = re.findall("(\d\d\d)([RrMmCcAa])\t?\s+?(\d\d\d)", content)

                for samples in thickness_index:
                    t_sample = samples[0]
                    t_sub = samples[1]


                    if t_sample + t_sub == sample + sub:
                        t_thick = float(samples[2])
                        corrected_resistivity = av_resistivity / t_thick * file_thickness
                        corrected_carrier_density = av_carrier_density / t_thick * file_thickness

        return sample, sub, corrected_resistivity, av_mobility, corrected_carrier_density, t_thick
    else:
        return "nothing"

##get results from all files in folder
def getFolder(folder,thickness_file="AZO_hall_thicknesses_20170314.txt"):
    hall_data = []
    if not folder.endswith("/"):
        folder = folder+"/"

    files_in_folder = os.listdir(folder)

    for items in files_in_folder:
        run_no_in_name = re.search("(\d\d\d)([RrCcMmAa])", items)
        if not items.endswith(".txt") or not run_no_in_name:
            continue

        file = items.strip()
        try:
            results = getFile(folder+file, folder=folder,thickness_file=thickness_file)
        except:
            print("getFile error: "+file)

        if results == "nothing":
            continue

        sample, sub, resistivity, mobility, carrier_density, thickness = results

        if len(hall_data) > 0:
            if sample + sub in [items["run_no"] + items["sub"] for items in HALL_data]:
                continue
        print(sample + sub, resistivity, mobility, carrier_density, thickness)

        hall_data.append({"run_no": sample,
                          "sub": sub,
                          "hall_p": resistivity,
                          "hall_mob": mobility,
                          "hall_n": carrier_density,
                          "thickness": thickness})
    else:
        print(sample + sub, resistivity, mobility, carrier_density, thickness)
        hall_data.append({"sample": sample,
                          "sub": sub,
                          "hall_p": resistivity,
                          "hall_mob": mobility,
                          "hall_n": carrier_density,
                          "thickness": thickness})
    return hall_data

def to_list(samples_data, hall_data):
    for dataset in samples_data:
        for h_sample in hall_data:
            if samples_data["run_no"] == h_sample["run_no"] and samples_data["sub"] == h_sample["sub"]:
                dataset.update({"hall_n": h_sample["hall_n"],
                                "hall_p": h_sample["hall_p"],
                                "hall_mob": h_sample["hall_mob"],
                           "hall_thick": h_sample["thickness"]})

                print("ok", dataset["sample"], dataset["sub"])