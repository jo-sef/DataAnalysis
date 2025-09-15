from dataclasses import dataclass
from typing import Optional

@dataclass
class Sample:
    run_no: str
    sub: str
    hall_n: Optional[float] = None
    hall_p: Optional[float] = None
    hall_mob: Optional[float] = None
    hall_thick: Optional[float] = None
    t_bandgap: Optional[float] = None
    Al_content: Optional[float] = None
    Al_error: Optional[float] = None
    Zn_content: Optional[float] = None
    Zn_error: Optional[float] = None
    SIMS_T: Optional[float] = None
    DEZn_molar: Optional[float] = None
    TEGa_molar: Optional[float] = None
    TMAl_molar: Optional[float] = None
    tBuOH_molar: Optional[float] = None
    MO_total: Optional[float] = None
    vi_ii: Optional[float] = None
    vi_mo: Optional[float] = None
    ii: Optional[float] = None
    iii: Optional[float] = None
    vi: Optional[float] = None
    TMAl_flow: Optional[float] = None
    tBuOH_flow: Optional[float] = None
    MO_carrier: Optional[float] = None
    DEZn_flow: Optional[float] = None
    TEGa_flow: Optional[float] = None
    Gas_carrier: Optional[float] = None
    rocking13: Optional[float] = None
    rocking23: Optional[float] = None
