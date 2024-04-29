"""
subclass for Thorlab SLM hardware control in :mod:`slmsuite`.
Outlines which SLM superclass functions must be implemented.

@author Saroj B Chand

"""
import os
import ctypes
import warnings
import sys

sys.path.append('c:/Users/Saroj Chand/Documents/dioptric/servers/inputs')
# from .slm import SLM
from slm import SLM

DEFAULT_SDK_PATH = "C:/Users/Saroj Chand/Documents/dioptric/servers/inputs/Thorlabs_EXULUS_PythonSDK"
class thorSLM(SLM):
    """
    Template for implementing a new SLM subclass. Replace :class:`Template`
    with the desired subclass name. :class:`~slmsuite.hardware.slms.slm.SLM` is the
    superclass that sets the requirements for :class:`Template`.
    """
    def __init__(self, verbose=True, sdk_path=DEFAULT_SDK_PATH, lut_path=None, dx_um=8, dy_um=8, **kwargs):
        r"""
        Initializes an instance of a Meadowlark SLM.

        Caution
        ~~~~~~~
        :class:`.Meadowlark` defaults to 8 micron SLM pixel size
        (:attr:`.SLM.dx_um` = :attr:`.SLM.dy_um` = 8).
        This is valid for most Meadowlark models, but not true for all!

        Arguments
        ---------
        verbose : bool
            Whether to print extra information.
        sdk_path : str
            Path of the Blink SDK folder. Stored in :attr:`sdk_path`.
        lut_path : str OR None
            Passed to :meth:`load_lut`.
        kwargs
            See :meth:`.SLM.__init__` for permissible options.
        """
        # Validates the DPI awareness of this context, which is presumably important for scaling.
        if verbose: print("Validating DPI awareness...", end="")

        awareness = ctypes.c_int()
        error_get = ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
        error_set = ctypes.windll.shcore.SetProcessDpiAwareness(2)
        success = ctypes.windll.user32.SetProcessDPIAware()

        if not success:
            raise RuntimeError(
                "Meadowlark failed to validate DPI awareness."
                "Errors: get={}, set={}, awareness={}".format(error_get, error_set, awareness.value)
            )
        if verbose: print("success")

        # Open the SLM library
        if verbose: print("Constructing Blink SDK...", end="")

        dll_path = os.path.join(sdk_path, "SDK", "Blink_C_wrapper")
        try:
            ctypes.cdll.LoadLibrary(dll_path)
            self.slm_lib = ctypes.CDLL("Blink_C_wrapper")
        except:
            print("failure")
            raise ImportError(
                "Meadowlark .dlls did not did not import correctly. Is '{}' the correct path?"
                .format(dll_path)
            )

        self.sdk_path = sdk_path

        # Initialize the SDK. The requirements of Matlab, LabVIEW, C++ and Python are different, so pass
        # the constructor a boolean indicating if we are calling from C++/Python (true), or Matlab/LabVIEW (false)
        bool_cpp_or_python = ctypes.c_uint(1)
        self.slm_lib.Create_SDK(bool_cpp_or_python)

        # Adjust pre- and post-ramp slopes for accurate voltage setting
        # (otherwise, custom LUT calibration is not properly implemented [this feature is not implemented in slmsuite]).
        # You may need a special version of the SDK sent to you from Meadowlark to have access to these parameters.
        # self.slm_lib.SetPreRampSlope(20) # default is 7
        # self.slm_lib.SetPostRampSlope(24) # default is 24

        if verbose: print("success")

        # Load LUT.
        if verbose: print("Loading LUT file...", end="")

        try:
            true_lut_path = self.load_lut(lut_path)
        except RuntimeError:
            print("failure\n(could not find .lut file)")
        else:
            if verbose and true_lut_path != lut_path:
                print("success\n(loaded from '{}')".format(true_lut_path))


        # Instantiate the superclass
         # Construct other variables.
        super().__init__(
            self.slm_lib.Get_Width(),
            self.slm_lib.Get_Height(),
            bitdepth=self.slm_lib.Get_Depth(),
            name="Meadowlark",
            dx_um=dx_um,
            dy_um=dy_um,
            **kwargs
        )

        if self.bitdepth > 8:
            warnings.warn(
                "Bitdepth of {} > 8 detected; this has not been tested and might fail.".format(self.bitdepth)
            )

        self.write(None)


    def load_lut(self, lut_path=None):
        """
        Loads a voltage lookup table (LUT) to the SLM.
        This converts requested phase values to physical voltage perturbing
        the liquid crystals.

        Parameters
        ----------
        lut_path : str OR None
            Path to look for an LUT file in.
            If this is a .lut file, then this file is loaded to the SLM.
            If this is a directory, then searches all files inside the
            directory, and loads either the first .lut file, or if possible
            an .lut file starting with `"slm"`
            (which is more likely to correspond to the LUT customized to an SLM,
            as Meadowlark sends such files prefiexed by `"slm"` such as `"slm5758_at532.lut"`).

        Raises
        ------
        RuntimeError
            If a .lut file is not found.

        Returns
        -------
        str
            The path which was used to load the LUT.
        """
        # If a path is not given, search inside the SDK path.
        if lut_path is None:
            lut_path = os.path.join(self.sdk_path, "LUT Files")

        # If we already have a .lut file, proceed.
        if len(lut_path) > 4 and lut_path[-4:] == ".lut":
            pass
        else:   # Otherwise, treat the path like a folder and search inside the folder.
            lut_file = None

            for file in os.listdir(lut_path):
                # Only examine .lut files.
                if len(file) >= 4 and file[-4:].lower() == ".lut":
                    # Choose the first one.
                    if lut_file is None:
                        lut_file = file

                    # Or choose the first one that starts with "slm"
                    if file[:3].lower() == "slm" and not lut_file[:3].lower() == "slm":
                        lut_file = file
                        break

            # Throw an error if we didn't find a .lut file.
            if lut_file is not None:
                lut_path = os.path.join(lut_path, lut_file)
            else:
                raise RuntimeError(
                    "Could not find a .lut file at path '{}'".format(lut_path)
                )

        # Finally, load the lookup table.
        self.slm_lib.Load_lut(lut_path)

        return lut_path
    
    @staticmethod
    def info(verbose=True):
        """
        Discovers all SLMs detected by an SDK.
        Useful for a user to identify the correct serial numbers / etc.

        Parameters
        ----------
        verbose : bool
            Whether to print the discovered information.

        Returns
        --------
        list of str
            List of serial numbers or identifiers.
        """
        raise NotImplementedError()
        serial_list = get_serial_list()     # TODO: Fill in proper function.
        return serial_list

    def _write_hw(self, phase):
        """
        Low-level hardware interface to write ``phase`` data onto the SLM.
        When the user calls the :meth:`.SLM.write` method of
        :class:`.SLM`, ``phase`` is error checked before calling
        :meth:`_write_hw()`. See :meth:`.SLM._write_hw` for further detail.
        """
        # TODO: Insert code here to write raw phase data to the SLM.