"""Utility helpers for merging measurement data into sample collections."""
from typing import Iterable, Mapping, Sequence, Tuple, List


def merge_measurements(samples: Iterable[dict],
                       measurements: Iterable[Mapping],
                       keys: Sequence[str] = ("run_no", "sub")) -> List[dict]:
    """Merge measurement dictionaries into sample dictionaries.

    Parameters
    ----------
    samples: iterable of dict
        Collection of sample dictionaries to update in-place.
    measurements: iterable of mapping
        Measurement dictionaries containing values to merge into the samples.
    keys: sequence of str, optional
        Keys used to match samples and measurements.  Defaults to ("run_no", "sub").

    Returns
    -------
    list of dict
        The updated ``samples`` collection.
    """
    key_tuple: Tuple[str, ...] = tuple(keys)

    for measurement in measurements:
        # Skip measurements missing any of the required keys
        if not all(k in measurement for k in key_tuple):
            continue

        for sample in samples:
            if not all(k in sample for k in key_tuple):
                continue
            # Compare values as strings to allow numeric/string mismatches
            if all(str(sample[k]) == str(measurement[k]) for k in key_tuple):
                extra = {k: v for k, v in measurement.items() if k not in key_tuple}
                sample.update(extra)
    return list(samples)
