from utils import positioning as pos
from utils import kplotlib as kpl
import utils.tool_belt as tool_belt
from utils.constants import Axes, CoordsKey, NVSig, VirtualLaserKey
if __name__ == "__main__":
    ### Shared parameters

    green_laser = "laserglow_532"
    yellow_laser = "laserglow_589"
    red_laser = "cobolt_638"

    sample_xy = [0.0,0.0] # piezo XY voltage input (1.0=1V) (not coordinates, relative)
    coord_z = 20  # piezo z voltage (0 is the set midpoint, absolute) (negative is closer to smaple, move unit steps in sample; 37 is good surface focus with bs for Lovelace; 20 is good for dye)
    pixel_xy = [0.0, 0.0]  # galvo ref

    nv_sig = NVSig(
        name=f"rubin",
        coords={
            CoordsKey.SAMPLE: sample_xy,
            CoordsKey.Z: coord_z,
            CoordsKey.PIXEL: pixel_xy,
        },
        disable_opt=False,
        disable_z_opt=True,
        expected_counts=13,
        pulse_durations={
            VirtualLaserKey.IMAGING: int(5e6),
            VirtualLaserKey.CHARGE_POL: int(1e4),
            VirtualLaserKey.SPIN_POL: 2000,
            VirtualLaserKey.SINGLET_DRIVE: 300,  # placeholder
        },
    )

    # ### Routines to execute
    # pos.set_xyz_on_nv(nv_sig)
    # pulsegen_server = (
    # tool_belt.get_server_pulse_streamer()
    # counter_server = tool_belt.get_server_counter()