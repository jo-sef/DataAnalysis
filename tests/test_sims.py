import pytest
from data_analysis.sims import t_index, sims_class

SIMS_TEXT = (
    "0\t100\t0\t400\n"
    "10\t100\t10\t300\n"
    "20\t100\t20\t200\n"
    "30\t100\t30\t100\n"
)


def make_sims(tmp_path):
    file = tmp_path / "sample.asc"
    file.write_text(SIMS_TEXT)
    return sims_class(
        file,
        header_lines=0,
        footer_lines=0,
        rsf_threshold=0,
        rsf_high=1,
        rsf_low=1,
    )


def test_t_index(tmp_path):
    sim = make_sims(tmp_path)
    idx = t_index(sim.data, surface_exclusion_nm=15)
    assert idx == 7


def test_normalize(tmp_path):
    sim = make_sims(tmp_path)
    sim.normalize(100)
    factor = 100 / sim.raw["Zn Counts"]
    assert sim.Zn_correction_factor == pytest.approx(factor)
    assert sim.data["normalized Al Counts"].iloc[0] == pytest.approx(
        sim.data["corrected Al Counts"].iloc[0] * factor
    )


def test_normalize_requires_calibration(tmp_path):
    sim = make_sims(tmp_path)
    with pytest.raises(ValueError):
        sim.normalize(None)
