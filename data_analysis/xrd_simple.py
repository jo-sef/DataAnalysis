"""Lightweight XRD tools used for testing without heavy dependencies."""

from math import isnan

class XRDIO:
    """Input/output helper methods for XRD data."""

    def load_arrays(self, angles, psd):
        """Store raw XRD arrays.

        Parameters
        ----------
        angles : list[float]
            Angle values in degrees.
        psd : list[float]
            Measured intensities.

        Side Effects
        ------------
        Initializes ``self.data`` with ``Angle`` and ``PSD`` columns.
        """
        self.data = {"Angle": list(angles), "PSD": list(psd)}


class XRDBackground:
    """Background removal utilities."""

    def removeBackground(self):
        """Subtract a constant background from the signal.

        Requires
        --------
        ``self.data`` must contain a ``PSD`` list.

        Side Effects
        ------------
        Adds ``background``, ``PSD-b.g.`` and ``y-b.g.`` lists to ``self.data``.
        """
        psd = self.data["PSD"]
        bg = min(psd)
        self.data["background"] = [bg for _ in psd]
        self.data["PSD-b.g."] = [p - bg for p in psd]
        max_val = max(self.data["PSD-b.g."]) or 1.0
        self.data["y-b.g."] = [p / max_val * 100 for p in self.data["PSD-b.g."]]


class XRDPeakFit:
    """Peak extraction and fitting utilities."""

    def extract_peak(self, tol=0.5):
        """Extract peaks near expected angles.

        Requires
        --------
        ``self.exp_ang`` mapping and ``self.data`` with ``Angle`` and ``PSD-b.g.``.

        Side Effects
        ------------
        Creates ``peak{hkl}`` lists in ``self.data`` for each expected peak.
        """
        angles = self.data["Angle"]
        signal = self.data["PSD-b.g."]
        for hkl, pos in self.exp_ang.items():
            peak = []
            for ang, val in zip(angles, signal):
                peak.append(val if abs(ang - pos) < tol else float("nan"))
            self.data[f"peak{hkl}"] = peak

    def fit_peak(self, hkl):
        """Trivial peak fit using the extracted peak values.

        Requires
        --------
        ``self.data`` must contain ``peak{hkl}``.

        Side Effects
        ------------
        Adds ``fit{hkl}`` list mirroring the peak values.
        Returns a simple string report.
        """
        col = f"peak{hkl}"
        fit = list(self.data.get(col, []))
        self.data[f"fit{hkl}"] = fit
        return f"fit {hkl}"


class xrdSample(XRDIO, XRDBackground, XRDPeakFit):
    """Minimal xrdSample composed from IO, background and peak-fit helpers."""

    def __init__(self, exp_ang=None):
        self.exp_ang = exp_ang or {}
        self.data = {}
