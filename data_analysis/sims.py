#### This is a copy of sims.py
#### The goal is to make a version based on classes, to better keep track of what is being done.

from __future__ import annotations
import re
from pathlib import Path
from typing import Iterator, List, Mapping, Sequence, Tuple, Union

try:  # pragma: no cover - optional dependency, exercised indirectly
    import matplotlib.pyplot as plt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled lazily in getFolder
    plt = None

try:  # pragma: no cover - optional dependency, exercised indirectly
    import numpy as _np
except ModuleNotFoundError:  # pragma: no cover - handled lazily in getFile
    _np = None

try:  # pragma: no cover - optional dependency, exercised indirectly
    import pandas as _pd
except ModuleNotFoundError:  # pragma: no cover - handled lazily in getFile and sims_class
    _pd = None

try:  # pragma: no cover - optional dependency, exercised indirectly
    from scipy.interpolate import interp1d as _interp1d
except ModuleNotFoundError:  # pragma: no cover - handled lazily in getFile
    _interp1d = None

from .utils.merge import merge_measurements


def _require_numpy():
    """Return the NumPy module or raise an informative ImportError."""

    if _np is None:  # pragma: no cover - exercised when dependency missing
        raise ImportError(
            "NumPy is required for SIMS file loading. Install numpy to use getFile()."
        )
    return _np


def _require_pandas():
    """Return the pandas module or raise an informative ImportError."""

    if _pd is None:  # pragma: no cover - exercised when dependency missing
        raise ImportError(
            "pandas is required for SIMS data manipulation. Install pandas to use this functionality."
        )
    return _pd


def _require_interp1d():
    """Return SciPy's interp1d helper or raise an informative ImportError."""

    if _interp1d is None:  # pragma: no cover - exercised when dependency missing
        raise ImportError(
            "scipy is required for SIMS interpolation. Install scipy to use getFile()."
        )
    return _interp1d


def _ensure_matplotlib():
    """Return matplotlib.pyplot or raise an informative ImportError."""

    if plt is None:  # pragma: no cover - exercised when dependency missing
        raise ImportError(
            "matplotlib is required for creating SIMS figures. Install matplotlib or disable figure generation."
        )
    return plt


def _iter_depth_series(
    data: Union["_pd.DataFrame", Mapping[str, Sequence[float]], Sequence[Mapping[str, float]]],
    column: str,
) -> Iterator[Tuple[int, float, float]]:
    """Yield ``(index, depth, value)`` tuples from SIMS-like data structures.

    The helper accepts a pandas ``DataFrame`` when pandas is available as well as
    plain mappings (``{"Depth": [...], column: [...]}``) or sequences of
    mapping objects. Rows containing ``None`` values are ignored to emulate the
    behaviour of ``dropna`` in pandas.
    """

    if _pd is not None and isinstance(data, _pd.DataFrame):
        subset = data.loc[:, ["Depth", column]].dropna()
        for idx, row in subset.iterrows():
            yield int(idx), float(row["Depth"]), float(row[column])
        return

    if isinstance(data, Mapping):
        depths = list(data.get("Depth", []))
        values = list(data.get(column, []))
        for idx, (depth, value) in enumerate(zip(depths, values)):
            if depth is None or value is None:
                continue
            yield idx, float(depth), float(value)
        return

    if isinstance(data, Sequence):
        for idx, row in enumerate(data):
            if isinstance(row, Mapping):
                depth = row.get("Depth")
                value = row.get(column)
            else:
                try:
                    depth, value = row
                except (TypeError, ValueError):
                    continue
            if depth is None or value is None:
                continue
            yield idx, float(depth), float(value)
        return

    raise TypeError(
        "Unsupported data type for SIMS operations; expected pandas DataFrame or mapping-like structure."
    )


def getFile(filename):
    """
    Load data from file into a pandas dataframe
    :param filename: name of sims output file (text format)
    :return: a dataframe of depth,Al Counts, Zn Counts
    """
    pd = _require_pandas()
    np = _require_numpy()
    interp1d = _require_interp1d()

    with open(filename, "r") as fh:
        data = pd.read_table(
            fh,
            names=["Al_depth", "Al_count", "Zn_depth", "Zn_count"],
            skiprows=15,
            delimiter="\t*",
            skipfooter=140,
            engine="python",
        )

        x = np.linspace(int(min(data.Al_depth)), int(max(data.Zn_depth)), 2 * len(data.Al_depth))

        al_y = interp1d(data.Al_depth, data.Al_count, bounds_error=False, fill_value=np.NaN)(x)
        zn_y = interp1d(data.Zn_depth, data.Zn_count, bounds_error=False, fill_value=np.NaN)(x)
        new_data = {"Depth": x, "Al Counts": al_y, "Zn Counts": zn_y}
        out_data = pd.DataFrame(new_data, columns=["Depth", "Al Counts", "Zn Counts"])

        return out_data


def t_index(
    data: Union["_pd.DataFrame", Mapping[str, Sequence[float]], Sequence[Mapping[str, float]]]
):
    """
    Find the depth from the Zn counts. Depth is where the rate of change in the negative direction is minimum.
    Above 25 nm because of surface effects
    """

    threshold_nm = 15
    candidates: List[Tuple[int, float]] = []
    for idx, depth, value in _iter_depth_series(data, "Zn Counts"):
        if depth > threshold_nm:
            candidates.append((idx, value))

    if not candidates:
        raise ValueError("No data points available beyond 15 nm to compute the t_index.")

    return min(candidates, key=lambda item: item[1])[0]

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
        self.std = self.data.iloc[self.window[0]:self.window[1]].std()

        if self.raw["Al Counts"] > 1e15:
            rsf = 1
        else:
            rsf = 4e15
        self.data["Al Counts"] = self.data.loc[:,"Al Counts"]*rsf

        ## Correcting for Zn variations ##

        corrected_Al = self.data.loc[:,"Al Counts"] / self.data.loc[:,"Zn Counts"] * self.raw["Zn Counts"]

        self.data.loc[:,"corrected Al Counts"] = corrected_Al.iloc[self.window[0]:self.window[1]]

    def normalize(self, Zn_standard):
        self.Zn_correction_factor =Zn_standard/self.raw["Zn Counts"]
        self.data.loc[:,"normalized Al Counts"] = self.data.loc[:,"corrected Al Counts"] *self.Zn_correction_factor

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

            mpl = _ensure_matplotlib()
            f, ax = mpl.subplots(1, 1)
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
            mpl.close("all")
    return sims_data

def to_list(samples_data, sims_data):
    """Merge SIMS measurements into the provided sample collection."""
    return merge_measurements(samples_data, sims_data)
