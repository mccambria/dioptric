class SpotHologram(FeedbackHologram):
    """
    Holography optimized for the generation of optical focal arrays.

    Is a subclass of :class:`FeedbackHologram`, but falls back to non-camera-feedback
    routines if :attr:`cameraslm` is not passed.

    Tip
    ~~~
    Quality of life features to generate noise regions for mixed region amplitude
    freedom (MRAF) algorithms are supported. Specifically, set ``null_region``
    parameters to help specify where the noise region is not.

    Attributes
    ----------
    spot_knm, spot_kxy, spot_ij : array_like of float OR None
        Stored vectors with shape ``(2, N)`` in the style of
        :meth:`~slmsuite.holography.toolbox.format_2vectors()`.
        These vectors are floats.
        The subscript refers to the basis of the vectors, the transformations between
        which are autocomputed.
        If necessary transformations do not exist, :attr:`spot_ij` is set to ``None``.
    spot_knm_rounded : array_like of int
        :attr:`spot_knm` rounded to nearest integers (indices).
        These vectors are integers.
        This is necessary because
        GS algorithms operate on a pixel grid, and the target for each spot in a
        :class:`SpotHologram` is a single pixel (index).
    spot_kxy_rounded, spot_ij_rounded : array_like of float
        Once :attr:`spot_knm_rounded` is rounded, the original :attr:`spot_kxy`
        and :attr:`spot_ij` are no longer accurate. Transformations are again used
        to backcompute the positions in the ``"ij"`` and ``"kxy"`` bases corresponding
        to the true computational location of a given spot.
        These vectors are floats.
    spot_amp : array_like of float
        The target amplitude for each spot.
        Must have length corresponding to the number of spots.
        For instance, the user can request dimmer or brighter spots.
    external_spot_amp : array_like of float
        When using ``"external_spot"`` feedback or the ``"external_spot"`` stat group,
        the user must supply external data. This data is transferred through this
        attribute. For iterative feedback, have the ``callback()`` function set
        :attr:`external_spot_amp` dynamically. By default, this variable is set to even
        distribution of amplitude.
    spot_integration_width_knm : int
        For spot-specific feedback methods, better SNR is achieved when integrating over
        many farfield pixels. This variable stores the width of the integration region
        in ``"knm"`` (farfield) space.
    spot_integration_width_ij : int
        For spot-specific feedback methods, better SNR is achieved when integrating over
        many camera pixels. This variable stores the width of the integration region
        in ``"ij"`` (camera) space.
    null_knm : array_like of float OR None
        In addition to points where power is desired, :class:`SpotHologram` is equipped
        with quality of life features to select points where power is undesired. These
        points are stored in :attr:`null_knm` with shape ``(2, M)`` in the style of
        :meth:`~slmsuite.holography.toolbox.format_2vectors()`. A region around these
        points is set to zero (null) and not allowed to participate in the noise region.
    null_radius_knm : float
        The radius in ``"knm"`` space around the points :attr:`null_knm` to zero or null
        (prevent from participating in the ``nan`` noise region).
        This is useful to prevent power being deflected to very high orders,
        which are unlikely to be properly represented in practice on a physical SLM.
    null_region_knm : array_like of bool OR ``None``
        Array of shape :attr:`shape`. Where ``True``, sets the background to zero
        instead of nan. If ``None``, has no effect.
    subpixel_beamradius_knm : float
        The radius in knm space corresponding to the beamradius of the Gaussian spot
        which is targeted when ``subpixel`` features are enabled.
        In the future, a non-Gaussian kernel might be used instead.
        This radius is computed based upon the stored amplitude in the SLM (if passed).
        This is an experimental feature and should be used with caution.
    """

    def __init__(
        self,
        shape,
        spot_vectors,
        basis="knm",
        spot_amp=None,
        cameraslm=None,
        null_vectors=None,
        null_radius=None,
        null_region=None,
        null_region_radius_frac=None,
        subpixel=False,
        **kwargs
    ):
        """
        Initializes a :class:`SpotHologram` targeting given spots at ``spot_vectors``.

        Parameters
        ----------
        shape : (int, int)
            Computational shape of the SLM. See :meth:`.Hologram.__init__()`.
        spot_vectors : array_like
            Spot position vectors with shape ``(2, N)`` in the style of
            :meth:`~slmsuite.holography.toolbox.format_2vectors()`.
        basis : str
            The spots can be in any of the following bases:

            - ``"ij"`` for camera coordinates (pixels),
            - ``"kxy"`` for centered normalized SLM k-space (radians).
            - ``"knm"`` for computational SLM k-space (pixels).

            Defaults to ``"knm"`` if ``None``.
        spot_amp : array_like OR None
            The amplitude to target for each spot.
            See :attr:`SpotHologram.spot_amp`.
            If ``None``, all spots are assumed to have the same amplitude.
            Normalization is performed automatically; the user is not required to
            normalize.
        cameraslm : slmsuite.hardware.cameraslms.FourierSLM OR None
            If the ``"ij"`` basis is chosen, and/or if the user wants to make use of camera
            feedback, a cameraslm must be provided.
        null_vectors : array_like OR None
            Null position vectors with shape ``(2, N)`` in the style of
            :meth:`~slmsuite.holography.toolbox.format_2vectors()`.
            MRAF methods are forced zero around these points.
        null_radius : float OR None
            Radius to null around in the given ``basis``.
            Note that basis conversions are imperfect for anisotropic basis
            transformations. The radius will always be set to be circular in ``"knm"``
            space, and will attempt to match to the closest circle
            to the (potentially elliptical) projection into ``"knm"`` from the given ``basis``.
        null_region : array_like OR None
            Array of shape :attr:`shape`. Where ``True``, sets the background to zero
            instead of nan. If ``None``, has no effect.
        null_region_radius_frac : float OR None
            Helper function to set the ``null_region`` to zero for Fourier space radius fractions above
            ``null_region_radius_frac``. This is useful to prevent power being deflected
            to very high orders, which are unlikely to be properly represented in
            practice on a physical SLM.
        subpixel : bool
            If enabled, the :attr:`target` is set to a series of Gaussian spots with
            radius :attr:`subpixel_beamradius_knm` instead of a series of single pixel
            spots (the default for :class:`SpotHologram`). The major benefit here is:

            - **Greater resolution with limited padding.** With 2-3 orders of padding,
              the farfield has sufficient resolution to render a Gaussian positioned
              **inbetween** farfield pixels, allowing for greater resolutions without
              having to pad further. This is especially important when operating at the
              memory limits of a system.

            Defaults to ``False``. This is an experimental feature and should be used
            with caution. Currently, there are issues with the initial phase causing
            some spots to be permanently attenuated.
        **kwargs
            Passed to :meth:`.FeedbackHologram.__init__()`.
        """
        # Parse vectors.
        vectors = toolbox.format_2vectors(spot_vectors)

        if spot_amp is not None:
            assert np.shape(vectors)[1] == len(spot_amp.ravel()), \
                "spot_amp must have the same length as the provided spots."

        # Parse null_vectors
        if null_vectors is not None:
            null_vectors = toolbox.format_2vectors(null_vectors)
            assert np.all(np.shape(null_vectors) == np.shape(null_vectors)), \
                "spot_amp must have the same length as the provided spots."
        else:
            self.null_knm = None
            self.null_radius_knm = None
        self.null_region_knm = None

        # Interpret vectors depending upon the basis.
        if (basis is None or basis == "knm"):  # Computational Fourier space of SLM.
            self.spot_knm = vectors

            if cameraslm is not None:
                self.spot_kxy = toolbox.convert_blaze_vector(
                    self.spot_knm, "knm", "kxy", cameraslm.slm, shape
                )

                if cameraslm.fourier_calibration is not None:
                    self.spot_ij = cameraslm.kxyslm_to_ijcam(self.spot_kxy)
                else:
                    self.spot_ij = None
            else:
                self.spot_kxy = None
                self.spot_ij = None

            # Handle null parameters.
            self.null_knm = null_vectors
            self.null_radius_knm = null_radius
            self.null_region_knm = null_region
        elif basis == "kxy":                    # Normalized units.
            assert cameraslm is not None, "We need a cameraslm to interpret kxy."

            self.spot_kxy = vectors

            if hasattr(cameraslm, "fourier_calibration"):
                if cameraslm.fourier_calibration is not None:
                    self.spot_ij = cameraslm.kxyslm_to_ijcam(vectors)
                    # This is okay for non-feedback GS, so we don't error.
            else:
                self.spot_ij = None

            self.spot_knm = toolbox.convert_blaze_vector(
                self.spot_kxy, "kxy", "knm", cameraslm.slm, shape
            )
        elif basis == "ij":                     # Pixel on the camera.
            assert cameraslm is not None, "We need an cameraslm to interpret ij."
            assert cameraslm.fourier_calibration is not None, (
                "We need an cameraslm with "
                "fourier-calibrated kxyslm_to_ijcam and ijcam_to_kxyslm transforms "
                "to interpret ij."
            )

            self.spot_ij = vectors
            self.spot_kxy = cameraslm.ijcam_to_kxyslm(vectors)
            self.spot_knm = toolbox.convert_blaze_vector(
                vectors, "ij", "knm", cameraslm, shape
            )
        else:
            raise Exception("algorithms.py: Unrecognized basis for spots '{}'.".format(basis))

        # Handle null conversions in the ij or kxy cases.
        if basis == "ij" or basis == "kxy":
            if null_vectors is not None:
                # Convert the null vectors.
                self.null_knm = toolbox.convert_blaze_vector(
                    null_vectors, basis, "knm", cameraslm, shape
                )

                # Convert the null radius.
                if null_radius is not None:
                    self.null_radius_knm = toolbox.convert_blaze_radius(
                        null_radius, basis, "knm", cameraslm, shape
                    )
                else:
                    self.null_radius_knm = None
            else:
                self.null_knm = None
                self.null_radius_knm = None

            self.null_region_knm = null_region

        # Generate point spread functions (psf) for the knm and ij bases
        if cameraslm is not None:
            psf_kxy = cameraslm.slm.spot_radius_kxy()
            psf_knm = toolbox.convert_blaze_radius(psf_kxy, "kxy", "knm", cameraslm.slm, shape)
            psf_ij = toolbox.convert_blaze_radius(psf_kxy, "kxy", "ij", cameraslm, shape)
        else:
            psf_knm = 0
            psf_ij = np.nan

        if np.isnan(psf_knm):   psf_knm = 0
        if np.isnan(psf_ij):    psf_ij = 0

        if subpixel:
            warnings.warn(
                "algorithms.py: subpixel spot sampling is an experimental feature "
                "and should be used with caution."
            )
            if psf_knm > .5:
                self.subpixel_beamradius_knm = psf_knm
            else:
                raise ValueError(
                    "algorithms.py: nearfield amplitude is not sufficiently padded to have "
                    "appreciable size in the farfield. Consider padding more to use subpixel "
                    "features."
                )
        else:
            self.subpixel_beamradius_knm = None

        # Use semi-arbitrary values to determine integration widths. The default width is:
        #  - six times the psf,
        #  - but then clipped to be:
        #    + larger than 3 and
        #    + smaller than the minimum inf-norm distance between spots divided by 1.5
        #      (divided by 1 would correspond to the largest non-overlapping integration
        #      regions; 1.5 gives comfortable padding)
        #  - and finally forced to be an odd integer.
        min_psf = 3

        dist_knm = np.max([toolbox.smallest_distance(self.spot_knm) / 1.5, min_psf])
        self.spot_integration_width_knm = np.clip(6 * psf_knm, min_psf, dist_knm)
        self.spot_integration_width_knm = int(2 * np.floor(self.spot_integration_width_knm / 2) + 1)

        if self.spot_ij is not None:
            dist_ij = np.max([toolbox.smallest_distance(self.spot_ij) / 1.5, min_psf])
            self.spot_integration_width_ij = np.clip(6 * psf_ij, 3, dist_ij)
            self.spot_integration_width_ij =  int(2 * np.floor(self.spot_integration_width_ij / 2) + 1)
        else:
            self.spot_integration_width_ij = None

        # Check to make sure spots are within relevant camera and SLM shapes.
        if (
            np.any(self.spot_knm[0] < self.spot_integration_width_knm / 2) or
            np.any(self.spot_knm[1] < self.spot_integration_width_knm / 2) or
            np.any(self.spot_knm[0] >= shape[1] - self.spot_integration_width_knm / 2) or
            np.any(self.spot_knm[1] >= shape[0] - self.spot_integration_width_knm / 2)
        ):
            raise ValueError(
                "Spots outside SLM computational space bounds!\nSpots:\n{}\nBounds: {}".format(
                    self.spot_knm, shape
                )
            )

        if self.spot_ij is not None:
            cam_shape = cameraslm.cam.shape

            if (
                np.any(self.spot_ij[0] < self.spot_integration_width_ij / 2) or
                np.any(self.spot_ij[1] < self.spot_integration_width_ij / 2) or
                np.any(self.spot_ij[0] >= cam_shape[1] - self.spot_integration_width_ij / 2) or
                np.any(self.spot_ij[1] >= cam_shape[0] - self.spot_integration_width_ij / 2)
            ):
                raise ValueError(
                    "Spots outside camera bounds!\nSpots:\n{}\nBounds: {}".format(
                        self.spot_ij, cam_shape
                    )
                )

        # Parse spot_amp.
        if spot_amp is None:
            self.spot_amp = np.full(len(vectors[0]), 1.0 / np.sqrt(len(vectors[0])))
        else:
            self.spot_amp = np.ravel(spot_amp)

        # Set the external amp variable to be perfect by default.
        self.external_spot_amp = np.copy(self.spot_amp)

        # Decide the null_radius (if necessary)
        if self.null_knm is not None:
            if self.null_radius_knm is None:
                all_spots = np.hstack((self.null_knm, self.spot_knm))
                self.null_radius_knm = toolbox.smallest_distance(all_spots) / 4

            self.null_radius_knm = int(np.ceil(self.null_radius_knm))

        # Initialize target/etc.
        super().__init__(shape, target_ij=None, cameraslm=cameraslm, **kwargs)

        # Parse null_region after __init__
        if basis == "ij" and null_region is not None:
            # Transformation order of zero to prevent nan-blurring in MRAF cases.
            self.null_region_knm = self.ijcam_to_knmslm(null_region, out=self.null_region_knm, order=0) != 0

        # If we have an input for null_region_radius_frac, then force the null region to
        # exclude higher order k-vectors according to the desired exclusion fraction.
        if null_region_radius_frac is not None:
            # Build up the null region pattern if we have not already done the transform above.
            if self.null_region_knm is None:
                self.null_region_knm = cp.zeros(self.shape, dtype=bool)

            # Make a circle, outside of which the null_region is active.
            xl = cp.linspace(-1, 1, self.null_region_knm.shape[0])
            yl = cp.linspace(-1, 1, self.null_region_knm.shape[1])
            (xg, yg) = cp.meshgrid(xl, yl)
            mask = cp.square(xg) + cp.square(yg) > null_region_radius_frac ** 2
            self.null_region_knm[mask] = True

        # Fill the target with data.
        self.update_target(reset_weights=True)

    def __len__(self):
        """
        Overloads len() to return the number of spots in this :class:`SpotHologram`.

        Returns
        -------
        int
            The length of :attr:`spot_amp`.
        """
        return self.spot_knm.shape[1]

    @staticmethod
    def make_rectangular_array(
        shape,
        array_shape,
        array_pitch,
        array_center=None,
        basis="knm",
        orientation_check=False,
        **kwargs
    ):
        """
        Helper function to initialize a rectangular 2D array of spots, with certain size and pitch.

        Note
        ~~~~
        The array can be in SLM k-space coordinates or in camera pixel coordinates, depending upon
        the choice of ``basis``. For the ``"ij"`` basis, ``cameraslm`` must be included as one
        of the ``kwargs``. See :meth:`__init__()` for more ``basis`` information.

        Important
        ~~~~~~~~~
        Spot positions will be rounded to the grid of computational k-space ``"knm"``,
        to create the target image (of finite size) that algorithms optimize towards.
        Choose ``array_pitch`` and ``array_center`` carefully to avoid undesired pitch
        non-uniformity caused by this rounding.

        Parameters
        ----------
        shape : (int, int)
            Computational shape of the SLM in :mod:`numpy` `(h, w)` form. See :meth:`.SpotHologram.__init__()`.
        array_shape : (int, int) OR int
            The size of the rectangular array in number of spots ``(NX, NY)``.
            If a scalar N is given, assume ``(N, N)``.
        array_pitch : (float, float) OR float
            The spacing between spots in the x and y directions ``(pitchx, pitchy)``.
            If a single pitch is given, assume ``(pitch, pitch)``.
        array_center : (float, float) OR None
            The shift of the center of the spot array from the zeroth order.
            Uses ``(x, y)`` form in the chosen basis.
            If ``None``, defaults to the position of the zeroth order, converted into the
            relevant basis:

             - If ``"knm"``, this is ``(shape[1], shape[0])/2``.
             - If ``"kxy"``, this is ``(0,0)``.
             - If ``"ij"``, this is the pixel position of the zeroth order on the
               camera (calculated via Fourier calibration).

        basis : str
            See :meth:`__init__()`.
        orientation_check : bool
            Whether to delete the last two points to check for parity.
        **kwargs
            Any other arguments are passed to :meth:`__init__()`.
        """
        # Parse size and pitch.
        if isinstance(array_shape, REAL_TYPES):
            array_shape = (int(array_shape), int(array_shape))
        if isinstance(array_pitch, REAL_TYPES):
            array_pitch = (array_pitch, array_pitch)

        # Determine array_center default.
        if array_center is None:
            if basis == "knm":
                array_center = (shape[1] / 2.0, shape[0] / 2.0)
            elif basis == "kxy":
                array_center = (0, 0)
            elif basis == "ij":
                assert "cameraslm" in kwargs, "We need an cameraslm to interpret ij."
                cameraslm = kwargs["cameraslm"]
                assert cameraslm is not None, "We need an cameraslm to interpret ij."
                assert cameraslm.fourier_calibration is not None, (
                    "We need an cameraslm with "
                    "fourier-calibrated kxyslm_to_ijcam and ijcam_to_kxyslm transforms "
                    "to interpret ij."
                )

                array_center = toolbox.convert_blaze_vector(
                    (0, 0), "kxy", "ij", cameraslm
                )

        # Make the grid edges.
        x_edge = (np.arange(array_shape[0]) - (array_shape[0] - 1) / 2.0)
        x_edge = x_edge * array_pitch[0] + array_center[0]
        y_edge = (np.arange(array_shape[1]) - (array_shape[1] - 1) / 2.0)
        y_edge = y_edge * array_pitch[1] + array_center[1]

        # Make the grid lists.
        x_grid, y_grid = np.meshgrid(x_edge, y_edge, sparse=False, indexing="xy")
        x_list, y_list = x_grid.ravel(), y_grid.ravel()

        # Delete the last two points if desired and valid.
        if orientation_check and len(x_list) > 2:
            x_list = x_list[:-2]
            y_list = y_list[:-2]

        vectors = np.vstack((x_list, y_list))

        # Return a new SpotHologram.
        return SpotHologram(shape, vectors, basis=basis, spot_amp=None, **kwargs)

    def _update_target_spots(self, reset_weights=False, plot=False):
        """
        Wrapped by :meth:`SpotHologram.update_target()`.
        """
        # Round the spot points to the nearest integer coordinates in knm space.
        if self.subpixel_beamradius_knm is None:
            self.spot_knm_rounded = np.around(self.spot_knm).astype(int)
        else:
            # Don't round if we're doing subpixel stuff.
            self.spot_knm_rounded = self.spot_knm

        # Convert these to the other coordinate systems if possible.
        if self.cameraslm is not None:
            self.spot_kxy_rounded = toolbox.convert_blaze_vector(
                self.spot_knm_rounded,
                "knm",
                "kxy",
                self.cameraslm.slm,
                self.shape,
            )

            if self.cameraslm.fourier_calibration is not None:
                self.spot_ij_rounded = self.cameraslm.kxyslm_to_ijcam(
                    self.spot_kxy_rounded
                )
            else:
                self.spot_ij_rounded = None
        else:
            self.spot_kxy_rounded = None
            self.spot_ij_rounded = None

        # Erase previous target in-place.
        if self.null_knm is None:
            self.target.fill(0)
        else:
            # By default, everywhere is "amplitude free", denoted by nan.
            self.target.fill(np.nan)

            # Now we start setting areas where null is desired. First, zero the blanket region.
            if self.null_region_knm is not None:
                self.target[self.null_region_knm] = 0

            # Second, zero the regions around the "null points".
            if self.null_knm is not None:
                all_spots = np.hstack((self.null_knm, self.spot_knm))
                w = int(2*self.null_radius_knm + 1)

                for ii in range(all_spots.shape[1]):
                    toolbox.imprint(
                        self.target,
                        (np.around(all_spots[0, ii]), w, np.around(all_spots[1, ii]), w),
                        0,
                        centered=True,
                        circular=True
                    )

        # Set all the target pixels to the appropriate amplitude.
        if self.subpixel_beamradius_knm is None:
            self.target[
                self.spot_knm_rounded[1, :], self.spot_knm_rounded[0, :]
            ] = self.spot_amp
        else:   # Otherwise, make a target consisting of imprinted gaussians (subpixel enabled)
            grid = np.meshgrid(np.arange(self.target.shape[1]), np.arange(self.target.shape[0]))

            for spot_idx in range(len(self)):
                toolbox.imprint(
                    matrix=self.target,
                    window=(
                        self.spot_knm[0, spot_idx], 4*np.ceil(self.subpixel_beamradius_knm)+1,
                        self.spot_knm[1, spot_idx], 4*np.ceil(self.subpixel_beamradius_knm)+1
                    ),
                    function=gaussian2d,
                    grid=grid,
                    imprint_operation="replace",
                    centered=True,
                    circular=True,
                    clip=True,                      # End of imprint parameters
                    x0=self.spot_knm[0, spot_idx],  # Start of gaussian2d parameters
                    y0=self.spot_knm[1, spot_idx],
                    a=self.spot_amp[spot_idx],
                    c=0,
                    wx=self.subpixel_beamradius_knm,
                    wy=self.subpixel_beamradius_knm,
                )

        self.target /= Hologram._norm(self.target)

        if reset_weights:
            self.reset_weights()

        if plot:
            self.plot_farfield(self.target)

    def update_target(self, reset_weights=False, plot=False):
        """
        From the spot locations stored in :attr:`spot_knm`, update the target pattern.

        Note
        ~~~~
        If there's a cameraslm, updates the :attr:`spot_ij_rounded` attribute
        corresponding to where pixels in the k-space where actually placed (due to rounding
        to integers, stored in :attr:`spot_knm_rounded`), rather the
        idealized floats :attr:`spot_knm`.

        Note
        ~~~~
        The :attr:`target` and :attr:`weights` matrices are modified in-place for speed,
        unlike :class:`.Hologram` or :class:`.FeedbackHologram` which make new matrices.
        This is because spot positions are expected to be corrected using :meth:`correct_spots()`.

        Parameters
        ----------
        reset_weights : bool
            Whether to rest the :attr:`weights` to this new :attr:`target`.
        plot : bool
            Whether to enable debug plotting to see the positions of the spots relative to the
            shape of the camera and slm.
        """
        self._update_target_spots(reset_weights=reset_weights, plot=plot)

    def refine_offset(self, img=None, basis="kxy", force_affine=True, plot=False):
        """
        Hones the positions of the produced spots toward the desired targets to compensate for
        Fourier calibration imperfections. Works either by moving camera integration
        regions to the positions where the spots ended up (``basis="ij"``) or by moving
        the :math:`k`-space targets to target the desired camera pixels
        (``basis="knm"``/``basis="kxy"``). This should be run at the user's request
        inbetween :meth:`optimize` iterations.

        Parameters
        ----------
        img : numpy.ndarray OR None
            Image measured by the camera. If ``None``, defaults to :attr:`img_ij` via :meth:`measure()`.
        basis : str
            The correction can be in any of the following bases:

            - ``"ij"`` changes the pixel that the spot is expected at,
            - ``"kxy"``, ``"knm"`` changes the k-vector which the SLM targets.

            Defaults to ``"kxy"``. If basis is set to ``None``, no correction is applied
            to the data in the :class:`SpotHologram`.
        force_affine : bool
            Whether to force the offset refinement to behave as an affine transformation
            between the original and refined coordinate system. This helps to tame
            outliers. Defaults to ``True``.

        plot : bool
            Enables debug plots.

        Returns
        -------
        numpy.ndarray
            Spot shift in the ``"ij"`` basis for each spot.
        """
        # If no image was provided, get one from cache.
        if img is None:
            self.measure(basis="ij")
            img = self.img_ij

        # Take regions around each point from the given image.
        regions = analysis.take(
            img, self.spot_ij, self.spot_integration_width_ij, centered=True, integrate=False
        )

        # Fast version; have to iterate for accuracy.
        shift_vectors = analysis.image_positions(regions)
        shift_vectors = np.clip(
            shift_vectors,
            -self.spot_integration_width_ij/4,
            self.spot_integration_width_ij/4
        )

        # Store the shift vector before we force_affine.
        sv1 = self.spot_ij + shift_vectors

        if force_affine:
            affine = analysis.fit_affine(self.spot_ij, self.spot_ij + shift_vectors, plot=plot)
            shift_vectors = (np.matmul(affine["M"], self.spot_ij) + affine["b"]) - self.spot_ij

        # Record the shift vector after we force_affine.
        sv2 = self.spot_ij + shift_vectors

        # Plot the above if desired.
        if plot:
            mask = analysis.take(
                img, self.spot_ij, self.spot_integration_width_ij,
                centered=True, integrate=False, return_mask=True
            )

            plt.figure(figsize=(12, 12))
            plt.imshow(img * mask)
            plt.scatter(sv1[0,:], sv1[1,:], s=200, fc="none", ec="r")
            plt.scatter(sv2[0,:], sv2[1,:], s=300, fc="none", ec="b")
            plt.show()

        # Handle the feedback applied from this refinement.
        if basis is not None:
            if (basis == "kxy" or basis == "knm"):
                # Modify k-space targets. Don't modify any camera spots.
                self.spot_kxy = self.spot_kxy - (
                    self.cameraslm.ijcam_to_kxyslm(shift_vectors) -
                    self.cameraslm.ijcam_to_kxyslm((0,0))
                )
                self.spot_knm = toolbox.convert_blaze_vector(
                    self.spot_kxy, "kxy", "knm", self.cameraslm.slm, self.shape
                )
                self.update_target(reset_weights=True)
                self.reset_phase()
            elif basis == "ij":
                # Modify camera targets. Don't modify any k-vectors.
                self.spot_ij = self.spot_ij - shift_vectors
            else:
                raise Exception("Unrecognized basis '{}'.".format(basis))

        return shift_vectors

    def _update_weights(self):
        """
        Change :attr:`weights` to optimize towards the :attr:`target` using feedback from
        :attr:`amp_ff`, the computed farfield amplitude. This function also updates stats.
        """
        feedback = self.flags["feedback"]

        # If we're doing subpixel stuff, we can't use computational feedback, upgrade to computational_spot.
        if self.subpixel_beamradius_knm is not None and feedback == "computational":
            feedback = self.flags["feedback"] = "computational_spot"

        # Weighting strategy depends on the chosen feedback method.
        if feedback == "computational":
            # Pixel-by-pixel weighting
            self._update_weights_generic(self.weights, self.amp_ff, self.target, nan_checks=True)
        else:
            # Integrate a window around each spot, with feedback from respective sources.
            if feedback == "computational_spot":
                amp_feedback = cp.sqrt(analysis.take(
                    cp.square(self.amp_ff),
                    self.spot_knm_rounded,
                    self.spot_integration_width_knm,
                    centered=True,
                    integrate=True,
                    mp=cp
                ))
            elif feedback == "experimental_spot":
                self.measure(basis="ij")

                amp_feedback = np.sqrt(analysis.take(
                    np.square(np.array(self.img_ij, copy=False, dtype=self.dtype)),
                    self.spot_ij,
                    self.spot_integration_width_ij,
                    centered=True,
                    integrate=True
                ))
            elif feedback == "external_spot":
                amp_feedback = self.external_spot_amp
            else:
                raise ValueError("algorithms.py: Feedback '{}' not recognized.".format(feedback))

            if self.subpixel_beamradius_knm is None:
                # Default mode: no subpixel stuff. We update single pixels.
                self.weights[self.spot_knm_rounded[1, :], self.spot_knm_rounded[0, :]] = (
                    self._update_weights_generic(
                        self.weights[self.spot_knm_rounded[1, :], self.spot_knm_rounded[0, :]],
                        cp.array(amp_feedback, copy=False, dtype=self.dtype),
                        self.spot_amp,
                        nan_checks=True
                    )
                )
            else:
                # Complex mode: subpixel stuff. Update Gaussian patterns.
                if hasattr(amp_feedback, "get"):
                    amp_feedback = amp_feedback.get()

                # Figure out the multiplication factors on a dummy array.
                dummy_weights = (
                    self._update_weights_generic(
                        np.ones(len(self)),
                        amp_feedback,
                        self.spot_amp,
                        mp=np
                    )
                )

                # Update each Gaussian with each respective multiplication factor.
                for spot_idx in range(len(self)):
                    window = toolbox.window_slice(
                        window=(
                            self.spot_knm[0, spot_idx], 4*np.ceil(self.subpixel_beamradius_knm)+1,
                            self.spot_knm[1, spot_idx], 4*np.ceil(self.subpixel_beamradius_knm)+1
                        ),
                        shape=None,
                        centered=True,
                        circular=True
                    )
                    self.weights[window] *= dummy_weights[spot_idx]

    def _calculate_stats_spots(self, stats, stat_groups=[]):
        """
        Wrapped by :meth:`SpotHologram.update_stats()`.
        """

        if "computational_spot" in stat_groups:
            if self.shape == self.slm_shape:
                # Spot size is one pixel wide: no integration required.
                stats["computational_spot"] = self._calculate_stats(
                    self.amp_ff[self.spot_knm_rounded[1, :], self.spot_knm_rounded[0, :]],
                    self.spot_amp,
                    efficiency_compensation=False,
                    total=cp.sum(cp.square(self.amp_ff)),
                    raw="raw_stats" in self.flags and self.flags["raw_stats"]
                )
            else:
                # Spot size is wider than a pixel: integrate a window around each spot
                if cp != np:
                    pwr_ff = cp.square(self.amp_ff)
                    pwr_feedback = analysis.take(
                        pwr_ff,
                        self.spot_knm,
                        self.spot_integration_width_knm,
                        centered=True,
                        integrate=True,
                        mp=cp
                    )

                    stats["computational_spot"] = self._calculate_stats(
                        cp.sqrt(pwr_feedback),
                        self.spot_amp,
                        mp=cp,
                        efficiency_compensation=False,
                        total=cp.sum(pwr_ff),
                        raw="raw_stats" in self.flags and self.flags["raw_stats"]
                    )
                else:
                    pwr_ff = np.square(self.amp_ff)
                    pwr_feedback = analysis.take(
                        pwr_ff,
                        self.spot_knm,
                        self.spot_integration_width_knm,
                        centered=True,
                        integrate=True
                    )

                    stats["computational_spot"] = self._calculate_stats(
                        np.sqrt(pwr_feedback),
                        self.spot_amp,
                        mp=np,
                        efficiency_compensation=False,
                        total=np.sum(pwr_ff),
                        raw="raw_stats" in self.flags and self.flags["raw_stats"]
                    )

        if "experimental_spot" in stat_groups:
            self.measure(basis="ij")

            pwr_img = np.square(self.img_ij)

            pwr_feedback = analysis.take(
                pwr_img,
                self.spot_ij,
                self.spot_integration_width_ij,
                centered=True,
                integrate=True
            )

            stats["experimental_spot"] = self._calculate_stats(
                np.sqrt(pwr_feedback),
                self.spot_amp,
                mp=np,
                efficiency_compensation=False,
                total=np.sum(pwr_img),
                raw="raw_stats" in self.flags and self.flags["raw_stats"]
            )

        if "external_spot" in stat_groups:
            pwr_feedback = np.square(np.array(self.external_spot_amp, copy=False, dtype=self.dtype))
            stats["external_spot"] = self._calculate_stats(
                np.sqrt(pwr_feedback),
                self.spot_amp,
                mp=np,
                efficiency_compensation=False,
                total=np.sum(pwr_feedback),
                raw="raw_stats" in self.flags and self.flags["raw_stats"]
            )

    def update_stats(self, stat_groups=[]):
        """
        Calculate statistics corresponding to the desired ``stat_groups``.

        Parameters
        ----------
        stat_groups : list of str
            Which groups or types of statistics to analyze.
        """
        stats = {}

        self._calculate_stats_computational(stats, stat_groups)
        self._calculate_stats_experimental(stats, stat_groups)
        self._calculate_stats_spots(stats, stat_groups)

        self._update_stats_dictionary(stats)