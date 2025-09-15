#### This is a copy of sims.py
#### The goal is to make a version based on classes, to better keep track of what is being done.

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from pathlib import Path
from .utils.merge import merge_measurements


def getFile(filename, skiprows=15, skipfooter=140):
    """Load SIMS text output into a :class:`pandas.DataFrame`.

    Parameters
    ----------
    filename : str or :class:`pathlib.Path`
        Name of SIMS output file. The file must contain four tab separated
        columns representing ``Al_depth``, ``Al_count``, ``Zn_depth`` and
        ``Zn_count``.
    skiprows : int, optional
        Number of header lines to skip. Defaults to ``15`` as produced by the
        measurement software.
    skipfooter : int, optional
        Number of footer lines to skip. Defaults to ``140``.

    Returns
    -------
    :class:`pandas.DataFrame`
        DataFrame with the required columns ``Depth``, ``Al Counts`` and
        ``Zn Counts`` where the depth grid has been interpolated onto a common
        axis for aluminium and zinc.
    """
    with open(filename, "r") as fh:
        data = pd.read_table(
            fh,
            names=["Al_depth", "Al_count", "Zn_depth", "Zn_count"],
            skiprows=skiprows,
            delimiter="\t*",
            skipfooter=skipfooter,
            engine="python",
        )

        x = np.linspace(int(min(data.Al_depth)), int(max(data.Zn_depth)), 2 * len(data.Al_depth))

        al_y = interp1d(data.Al_depth, data.Al_count, bounds_error=False, fill_value=np.NaN)(x)
        zn_y = interp1d(data.Zn_depth, data.Zn_count, bounds_error=False, fill_value=np.NaN)(x)
        new_data = {"Depth": x, "Al Counts": al_y, "Zn Counts": zn_y}
        out_data = pd.DataFrame(new_data, columns=["Depth", "Al Counts", "Zn Counts"])

        return out_data


def t_index(data, surface_exclusion_nm=15):
    """Index of the minimum zinc count beneath the surface region.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        DataFrame containing the columns ``Depth`` and ``Zn Counts``.
    surface_exclusion_nm : float, optional
        Depth in nanometres to exclude from the analysis to avoid surface
        effects. Only points deeper than this value are considered when
        searching for the minimum. Defaults to ``15``.

    Returns
    -------
    int
        Index in ``data`` corresponding to the minimum value of
        ``Zn Counts`` deeper than ``surface_exclusion_nm``.
    """

    col = "Zn Counts"
    x = data.dropna().loc[:, "Depth"]
    y = data.dropna().loc[:, col]

    mxi = y[x > surface_exclusion_nm].idxmin()

    return mxi

class sims_class:
    """Container for SIMS measurements and normalisation routines.

    The class reads a SIMS depth profile, determines a region of interest
    around the zinc minimum and provides utilities for normalising aluminium
    counts to an external calibration.
    """

    normalized = False
    filename = None
    data = None
    mxi = None

    window = None
    thickness = None
    raw = None

    def __init__(
        self,
        filename,
        rsf_threshold=1e15,
        rsf_high=1,
        rsf_low=4e15,
        window_fraction=0.75,
        header_lines=15,
        footer_lines=140,
    ):
        """Load and preprocess a SIMS depth profile.

        Parameters
        ----------
        filename : str or :class:`pathlib.Path`
            Path to the SIMS output file.
        rsf_threshold : float, optional
            Threshold on average aluminium counts used to decide which RSF
            value to apply. Defaults to ``1e15``.
        rsf_high : float, optional
            RSF applied when the aluminium counts exceed ``rsf_threshold``.
            Defaults to ``1``.
        rsf_low : float, optional
            RSF applied when the aluminium counts fall below ``rsf_threshold``.
            Defaults to ``4e15``.
        window_fraction : float, optional
            Fraction of the total profile around the zinc minimum used when
            calculating average counts and standard deviations. Defaults to
            ``0.75``.
        header_lines : int, optional
            Number of header lines to skip when reading ``filename``.
        footer_lines : int, optional
            Number of footer lines to skip when reading ``filename``.
        """

        self.filename = filename
        self.data = getFile(self.filename, skiprows=header_lines, skipfooter=footer_lines)

        ## thickness and selected portion (75%)
        self.mxi = t_index(self.data)
        window_size = window_fraction * self.mxi
        self.window = (int(self.mxi / 2 - window_size / 2), int(self.mxi / 2 + window_size / 2))
        self.thickness = self.data.iloc[self.mxi]["Depth"]

        ##average Counts

        self.raw = self.data.iloc[self.window[0] : self.window[1]].mean()
        self.std = self.data.iloc[self.window[0] : self.window[1]].std()

        rsf = rsf_high if self.raw["Al Counts"] > rsf_threshold else rsf_low
        self.data["Al Counts"] = self.data.loc[:, "Al Counts"] * rsf

        ## Correcting for Zn variations ##

        corrected_Al = self.data.loc[:, "Al Counts"] / self.data.loc[:, "Zn Counts"] * self.raw["Zn Counts"]

        self.data.loc[:, "corrected Al Counts"] = corrected_Al.iloc[self.window[0] : self.window[1]]

    def normalize(self, Zn_standard):
        """Normalise aluminium counts using an external zinc calibration.

        Parameters
        ----------
        Zn_standard : float
            Average zinc counts measured on a standard sample. This value is
            required to normalise the aluminium counts.

        Raises
        ------
        ValueError
            If ``Zn_standard`` is ``None``.
        """

        if Zn_standard is None:
            raise ValueError("Zn calibration data is required for normalisation")

        self.Zn_correction_factor = Zn_standard / self.raw["Zn Counts"]
        self.data.loc[:, "normalized Al Counts"] = self.data.loc[:, "corrected Al Counts"] * self.Zn_correction_factor

        self.Al = self.data["normalized Al Counts"].mean()
        self.Al_error = self.data["normalized Al Counts"].std()


def getFolder(folder, figures=False):
    """ create SIMS_data list from all SIMS data in a folder
        return SIMS_data a dictionary with filename, run_no, sub, and data as a df

    """
    folder = Path(folder)

    sample_index = "SIMS_sample_index.txt"

    sims_data = []

### Record data from each file so that the calibration files can be used in next loop.
    with open(folder / sample_index, "r") as f:
        for lines in f:

            if lines.startswith("#") or not lines.strip():
                continue
            search = re.findall("(\S*.dp_rpc_asc)\t?\s*?(\d+)([aAMmRrCc])", lines)
            if len(search) != 0:
                filename, run_no, sub = search[0]
                run_no = int(run_no)

            if len(search) == 0 and re.search("Al_uf", lines):
                if 0 in [s["run_no"] for s in sims_data]:
                    run_no = 1
                else:
                    run_no = 0
                sub = None
                filename = lines.split()[0]

            file_data = sims_class(folder / filename)

            sims_data.append({"filename": filename, "data": file_data, "run_no": run_no, "sub": sub})

    sims_data = sorted(sims_data, key=lambda k: k["run_no"])
    cal_data = [s for s in sims_data if (s["run_no"] == 0) or (s["run_no"]==1)]
### use calibration for
    for s in sims_data:
        Zn_cal = None
        Al_content = None

### Get class instance and find the matching cal file.
        sims_object = s["data"]
        date = re.findall("[0-9]+",s["filename"])[0]
        for cal in cal_data:
            cal_date = re.findall("[0-9]+",cal["filename"])[0]
            if cal_date == date:
                Zn_cal = cal["data"].raw["Zn Counts"]
            else:
                continue

        sims_object.normalize(Zn_cal)

        Al_content = sims_object.Al
        error = sims_object.Al_error

        s.update({"Al_content": Al_content,
                  "Al_error": error,
                  "Zn_content": sims_object.raw["Zn Counts"],
                  "Zn_error": sims_object.std["Zn Counts"],
                  "SIMS_T": sims_object.thickness})

        #### Save figure for visual check of SIMS data fit
        if figures == True:

            figure_folder = folder / "SIMS_fits"

            if not figure_folder.exists():
                figure_folder.mkdir(parents=True, exist_ok=True)
            save_name = "SIMS_%s%s.png" % (s["run_no"], s["sub"])

            f, ax = plt.subplots(1, 1)
            f.suptitle("%s%s" %(s["run_no"],s["sub"]))
            ax2 = ax.twinx()
            data = sims_object.data
            x = data["Depth"]
            y = data["Zn Counts"]

            y2 = data["Al Counts"]
            y3 = data["normalized Al Counts"]
            y4 = data["corrected Al Counts"]
            ax.plot(x, y, label="Zn")
            ax2.plot(x, y2, color="red", label="raw Al")
            ax2.plot(x, y3, color="green", label="normalized Al")
            ax2.plot(x, y4, color="black", label="corrected Al")
            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")
            f.text(0.3,0.3,"calibration Zn:%1.3f" %Zn_cal)
            f.text(0.3,0.2,"Al content:%1.3g" %Al_content)



            window = (data.iloc[sims_object.window[0]]["Depth"], data.iloc[sims_object.window[1]]["Depth"])
            ax.axvline(window[0])
            ax.axvline(window[1])
            f.savefig(figure_folder / save_name)
            plt.close("all")
    return sims_data

def to_list(samples_data, sims_data):
    """Merge SIMS measurements into the provided sample collection."""
    return merge_measurements(samples_data, sims_data)
