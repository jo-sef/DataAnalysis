import math
from data_analysis.xrd_simple import xrdSample


def synthetic_sample():
    angles = [20.0, 25.0, 25.5, 26.0, 30.0]
    psd = [10.0, 10.0, 60.0, 10.0, 10.0]
    sample = xrdSample(exp_ang={"001": 25.5})
    sample.load_arrays(angles, psd)
    return sample


def test_remove_background():
    s = synthetic_sample()
    s.removeBackground()
    assert s.data["PSD-b.g."][0] == 0.0


def test_extract_peak():
    s = synthetic_sample()
    s.removeBackground()
    s.extract_peak()
    peak = s.data["peak001"]
    values = [p for p in peak if not math.isnan(p)]
    assert values and max(values) == 50.0


def test_fit_peak():
    s = synthetic_sample()
    s.removeBackground()
    s.extract_peak()
    report = s.fit_peak("001")
    assert "fit" in report
    assert s.data["fit001"][2] == s.data["peak001"][2]
