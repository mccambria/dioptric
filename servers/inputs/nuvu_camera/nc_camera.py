# -*- coding: utf-8 -*-
"""
Python interface for Nuvu camera with calls to ctypes functions. Modified from file of same name
provided by Nuvu

Obtained on August 1st, 2023

@author: Nuvu/mccambria
"""

import time
from .nc_api import *
import numpy as np
import sys
import logging
from utils import tool_belt as tb


class NuvuException(Exception):
    """
        Classe: NuvuException, Hérite Exception
        permet le raise custom d'une erroeur provenant du sdk de Nuvu
    :attribut:  -error: numéro de l'erreur obtenue
    """

    def __init__(self, error):
        self.error = error

    def __str__(self):
        return repr(self.error)

    def value(self):
        """
            methode Value, class NuvuException
        :return: la valeur de l'erreur, définie dans le fichier erreur.h du sdk de nuvu
        """
        return self.error


class NcCamera:
    """
    Class nc_camera, Inherits: none

    Provides a Python-like interface to the Nuvu SDK.

    Attributes:
    - macAdress: Mac address of the camera being controlled.
    - ncCam: Pointer to the handle of the camera API.
    - ncImage: Pointer to the handle of the camera image.
    - readoutTime: c_double type readout time, initialized to -1.
    - WaitingTime: c_double type waiting time, initialized to -1.
    - ExposureTime: c_double type exposure time, initialized to -1.
    - shutterMode: c_int type camera shutter state (0=NOT SET, 1=open, 2=closed, 3=auto).
    - name: Save name of the image on the disk if using SDK functions.
    - comment: Comment in the metadata of the image saved with SDK functions.
    - width: Image width in pixels.
    - height: Image height in pixels.
    - inMemoryAccess: Boolean that determines if a pointer is allocated to an array for the image.
    - saveFormat: Determines the format of images saved by the SDK.
    - targetdetectorTempMin: Minimum target detector temperature.
    - targetdetectorTempMax: Maximum target detector temperature.
    """

    def __init__(self, MacAdress=None):
        self.macAdress = MacAdress
        self.ncCam = NCCAM()
        self.ncImage = NCIMAGE()
        self.nbBuff = 0
        self.ampliType = c_int(-2)
        self.vertFreq = c_int(0)
        self.horizFreq = c_int(0)
        self.readoutMode = c_int(-1)
        self.ampliString = "12345678"
        self.nbrReadoutMode = c_int(0)
        self.readoutTime = c_double(-1.0)
        self.waitingTime = c_double(-1.0)
        self.exposureTime = c_double(-1.0)
        self.shutterMode = c_int(0)
        self.name = "image1"
        self.comment = ""
        self.width = c_int(-1)
        self.height = c_int(-1)
        self.saveFormat = 1
        self.detectorTemp = c_double(100.0)
        self.controllerTemp = c_double(100.0)
        self.powerSupplyTemp = c_double(100.0)
        self.fpgaTemp = c_double(100.0)
        self.heatsinkTemp = c_double(100.0)
        self.targetDetectorTemp = c_double(100.0)
        self.targetDetectorTempMin = c_double(100.0)
        self.targetDetectorTempMax = c_double(100.0)
        self.rawEmGain = c_int(-1)
        self.rawEmGainRangeMin = c_int(-1)
        self.rawEmGainRangeMax = c_int(-1)
        self.calibratedEmGain = c_int(-1)
        self.calibratedEmGainMin = c_int(-1)
        self.calibratedEmGainMax = c_int(-1)
        self.calibratedEmGainTempMin = c_double(100.0)
        self.calibratedEmGainTempMax = c_double(100.0)
        self.binx = c_int(0)
        self.biny = c_int(0)
        tb.configure_logging(self)

    # region Dioptric functions
    # These functions are either ours or have been modified from the original file

    def connect(self, num_buffer=1):
        """
        Opens the connection with the camera (assumes there is only one available).
        If the class has been initialized with the camera's MAC address, the method will try to connect directly to that camera.
        :param nbBuff: Number of buffers initialized in the Nuvu API. (Number of images stored in pc memory at a time)
        :return: None
        """
        try:
            if self.macAdress is None:
                error = ncCamOpen(NC_AUTO_UNIT, NC_AUTO_CHANNEL, -1, byref(self.ncCam))
                if error:
                    raise NuvuException(error)
                self.nbBuff = num_buffer
            else:
                print("Still not implemented")

        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def disconnect(self, no_raise=False):
        """
        Function that closes the camera driver.
        :param no_raise: Internal parameter that allows not raising an error if the driver is already closed.
        :return: None
        """
        try:
            error = ncCamClose(self.ncCam)
            if error and not no_raise:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def _set_shutter_mode(self, shutter_mode, no_raise=False):
        """
        Méthode qui sélectionne le mode de l'obturateur
        :param mode: (int) mode de l'obturateur
        :return: None
        """
        try:
            error = ncCamSetShutterMode(self.ncCam, shutter_mode.value)
            if error and not no_raise:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def open_shutter(self):
        self._set_shutter_mode(ShutterMode.OPEN)

    def close_shutter(self, no_raise=False):
        self._set_shutter_mode(ShutterMode.CLOSE, no_raise)

    def set_readout_mode(self, readout_mode):
        """
        Set the camera's readout mode, including amplifier and vertical/horizontal frequencies.
        See camera_NUVU_hnu512gamma for more specificity
        """
        try:
            error = ncCamSetReadoutMode(self.ncCam, readout_mode)
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def get_readout_mode(self):
        """
        Méthode permettant de récupérer les informations du readout mode utilisé
        :return: None
        """
        try:
            error = ncCamGetCurrentReadoutMode(
                self.ncCam,
                byref(self.readoutMode),
                byref(self.ampliType),
                self.ampliString.encode(),
                byref(self.vertFreq),
                byref(self.horizFreq),
            )
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

        # return self.readoutMode.value
        ret_string = f"Mode: {self.readoutMode.value}; amplifier: {self.ampliType.value}; vertical frequency: {self.vertFreq.value}; horizontal frequency: {self.horizFreq.value}"
        return ret_string

    def get_num_readout_modes(self):
        """
        Récupère le nombre de readoutmode disponibles
        :return:
        """
        try:
            error = ncCamGetNbrReadoutModes(self.ncCam, byref(self.nbrReadoutMode))
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

        return self.nbrReadoutMode.value

    def set_trigger_mode(self, trigger_mode, num_images=0):
        try:
            error = ncCamSetTriggerMode(self.ncCam, trigger_mode.value, num_images)
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def set_timeout(self, timeout):
        """
        Method that allows selecting a timeout period representing the time before
        the driver declares an error if it waits for a new image to enter a buffer.
        :param timeout: (float) waiting time (in ms), -1 to turn off timeout
        :return: None
        """
        try:
            error = ncCamSetTimeout(self.ncCam, int(timeout))
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def set_target_detector_temp(self, temp):
        """
        Method for setting the target temperature of the detector for the camera.
        :param temp: (float) Target temperature
        :return: None
        """
        try:
            error = ncCamSetTargetDetectorTemp(self.ncCam, c_double(temp))
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def get_detector_temp(self):
        """
        Get the current temperature of the detector
        :return: float
        """
        self.getComponentTemp(0)
        return self.detectorTemp.value

    def get_size(self):
        """
        Method that retrieves the height and width of the images from the camera,
        setting these two values to the 'height' and 'width' attributes of the camera class.
        :return: None
        """
        try:
            error = ncCamGetSize(self.ncCam, byref(self.width), byref(self.height))
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def set_processing_type(self, processing_type, num_images_for_photon_counting=1):
        """
        Sets the processing type that will be applied on any future acquisition.
        A valid bias must be in place prior to the start of images acquisition with processing.
        :return: None
        """
        try:
            error = ncCamSetProcType(
                self.ncCam, processing_type.value, num_images_for_photon_counting
            )
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def update_bias(self, num_images=300, shutter_mode=ShutterMode.BIAS_DEFAULT):
        try:
            error = ncCamCreateBias(self.ncCam, num_images, shutter_mode.value)
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def start(self, num_images=0):
        """
        Method that starts image acquisition on the camera and sends them to the buffer.
        :param num_images: Determines the number of images the driver will take, if 0, then the acquisition is continuous.
        :return: None
        """
        try:
            error = ncCamStart(self.ncCam, num_images)
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def stop(self, no_raise=True):
        """
        Method that stops all image acquisition on the camera.
        :return: None
        """
        try:
            error = ncCamAbort(self.ncCam)
            if error and not no_raise:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def _read(self):
        """
        Method that reads the next image from the buffer and sends it to memory.
        :return: None
        """
        try:
            error = ncCamRead(self.ncCam, self.ncImage)
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def read(self):
        """
        Method that calls _read() and then casts the pointer to the image into a 16-bit array,
        which is copied to another part of memory.
        :return: None
        """
        start = time.time()
        self._read()
        stop = time.time()
        logging.info(f"_read: {stop - start}")
        start = time.time()
        np_img_array_pointer = np.ctypeslib.as_array(
            cast(self.ncImage, POINTER(c_uint16)),
            (self.width.value, self.height.value),
        )
        # return np_img_array_pointer.tobytes()
        img_str = np_img_array_pointer.tobytes()
        stop = time.time()
        logging.info(f"processing: {stop - start}")
        return img_str

    def set_heartbeat(self, heartbeat_ms):
        try:
            error = ncCamSetHeartbeat(self.ncCam, heartbeat_ms)
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def get_dynamic_buffer_count(self):
        num_buffer = c_int(-1)
        try:
            error = ncCamGetDynamicBufferCount(self.ncCam, byref(num_buffer))
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())
        return num_buffer.value

    def set_buffer_count(self, num_buffer):
        try:
            error = ncCamSetBufferCount(self.ncCam, num_buffer, 0)
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    # endregion

    # region Nuvu functions
    # These methods are unmodified from the original Nuvu file

    def errorHandling(self, error):
        """
        Method that ensures an appropriate reaction to errors. So far, the function crashes the program,
        closes the driver, and exits the software.
        :param error: error number returned by the SDK.
        :return: None
        """
        if error == 107:
            pass
            # print(error)
        # if error == 131:
        # Camera is started when it shouldn't be
        #     pass
        if error == 27:
            raise NuvuException("Error 27: Could not find camera")
        else:
            self.stop(no_raise=True)
            self.close_shutter(no_raise=True)
            # self.disconnect(no_raise=True)
            raise NuvuException(error)

    def getReadoutTime(self):
        """
        Method that makes a call to the camera to retrieve the readout time and stores the value in the attribute readoutTime.
        :return: None
        """
        try:
            error = ncCamGetReadoutTime(self.ncCam, byref(self.readoutTime))
            if error:
                raise NuvuException(error)

        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def setExposureTime(self, exposureTime):
        """
        Method to select the exposure time for the images.
        :param exposureTime: (float) the exposure time in milliseconds.
        :return: None
        """
        try:
            error = ncCamSetExposureTime(self.ncCam, exposureTime)
            if error:
                raise NuvuException(error)
            self.getExposureTime()
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def getExposureTime(self, cameraCall=1):
        """
        Method that retrieves the exposure time from the camera.
        :param cameraCall: Selects whether to check the value in the driver (0) or in the camera (1). Note: calling the camera takes time.
        :return: None
        """
        try:
            error = ncCamGetExposureTime(
                self.ncCam, cameraCall, byref(self.exposureTime)
            )
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def setWaitingTime(self, waitTime):
        """
        Méthode qui permet de sélectionner le temps d'attente entre deux acquisition
        :param waitTime: (float) temps d'attente en ms
        :return: None
        """
        try:
            error = ncCamSetWaitingTime(self.ncCam, waitTime)
            if error:
                raise NuvuException(error)
            self.getWaitingTime(cameraCall=0)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def getWaitingTime(self, cameraCall=1):
        """
        Méthode qui permet d'obtenir le temps d'attente entre deux acquisitions
        :param cameraCall: Sélectionne si on apelle le temps dans le driver (0) ou dans la caméra (1)
        A noter, un appel à la caméra prendra plus de temps.
        :return: None
        """
        try:
            error = ncCamGetWaitingTime(self.ncCam, cameraCall, byref(self.waitingTime))
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def getShutterMode(self, cameraCall=1):
        """
        Méthode qui récupère le mode de l'obturateur
        :param mode: (int) mode de l'obturateur
        :return: None
        """
        try:
            error = ncCamGetShutterMode(self.ncCam, cameraCall, byref(self.shutterMode))
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def saveImage(self, encode=0):
        """
        Méthode qui sauvegarde l'image stoquée dans ncImage et l'encode en tif (0) ou fits(1) sur le disque dur
        :param encode: (int) mode d'encodage
        :return: None
        """
        try:
            error = ncCamSaveImage(
                self.ncCam,
                self.ncImage,
                self.name.encode(),
                encode,
                self.comment.encode(),
                1,
            )
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def getTargetDetectorTempRange(self):
        """
        Permet de récupérer le range de températures demandées du détecteur
        :return: None
        """
        try:
            error = ncCamGetTargetDetectorTempRange(
                self.ncCam,
                byref(self.targetDetectorTempMin),
                byref(self.targetDetectorTempMax),
            )
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def getControllerTemp(self):
        """
        récupère la température actuelle du controleur et la store dans l'attribut controller temp
        :return: None
        """
        try:
            error = ncCamGetControllerTemp(self.ncCam, byref(self.controllerTemp))
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def getComponentTemp(self, comp):
        """
        Méthode qui récupère la température du composant spécifié et la stoque dans la valeur associée
        :param comp: identifiant du composant. 0=CCD, 1=controleur, 2=powerSupply, 3=FGPA, 4=heatSink
        :return: None
        """
        try:
            if comp == 0:
                temp = self.detectorTemp
            elif comp == 1:
                temp = self.controllerTemp
            elif comp == 2:
                temp = self.powerSupplyTemp
            elif comp == 3:
                temp = self.fpgaTemp
            elif comp == 4:
                temp = self.heatsinkTemp
            else:
                comp = -1

            error = ncCamGetComponentTemp(self.ncCam, c_int(comp), byref(temp))
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def getTargetDetectorTemp(self, cameraCall=1):
        """
        Méthode permettant de récupérer la température visée du détecteur et la stoquer dans
        l'attribut targetDetectorTemp
        :param cameraCall: Détermine si on apelle la caméra ou le driver, setté de base sur la camméra
        :return: None
        """
        try:
            error = ncCamGetTargetDetectorTemp(
                self.ncCam, cameraCall, byref(self.targetDetectorTemp)
            )
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def setRawEmGain(self, gain):
        """
        Méthode qui envoie une gain à la caméra
        :return: None
        """
        try:
            error = ncCamSetRawEmGain(self.ncCam, gain)
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def getRawEmGain(self, cameracall=1):
        """
        Méthode qui récupère la valeur rawEmGain de la caméra et la stoque dans l'attribut rawEmGain
        :return: None
        """
        try:
            error = ncCamGetRawEmGain(self.ncCam, cameracall, byref(self.rawEmGain))
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def getRawEmGainRange(self):
        """
        Méthode permettant de récupérer le range de Em gain disponible, stoque dans les attributs
        :return: None
        """
        try:
            error = ncCamGetRawEmGainRange(
                self.ncCam, byref(self.rawEmGainRangeMin), byref(self.rawEmGainRangeMax)
            )
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def setCalibratedEmGain(self, emGain):
        """
        Méthode permettant de sélectionner un EmGain calibré
        :param emGain: (int) Em gain
        :return: None
        """
        try:
            error = ncCamSetCalibratedEmGain(self.ncCam, emGain)
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def getCalibratedEmGain(self, cameraCall=1):
        """
        Méthode permettant de récupérer le Em Gain calibré sur la caméra
        :param cameraCall: Détermine si on accède a la caméra ou au driver
        :return: None
        """
        try:
            error = ncCamGetCalibratedEmGain(
                self.ncCam, cameraCall, byref(self.calibratedEmGain)
            )
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def getCalibratedEmGainRange(self):
        """
        Méthode permettant d'obtenir le range ou la calibration du emGain est valide
        :return: None
        """
        try:
            error = ncCamGetCalibratedEmGainRange(
                self.ncCam,
                byref(self.calibratedEmGainMin),
                byref(self.calibratedEmGainMax),
            )
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def getCalibratedEmGainTempRange(self):
        """
        Méthode permettant de connaitre l'intervalle en température de validité de la calibration du emGain
        :return: None
        """
        try:
            error = ncCamGetCalibratedEmGainTempRange(
                self.ncCam,
                byref(self.calibratedEmGainTempMin),
                byref(self.calibratedEmGainTempMax),
            )
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def camInit(self):
        """
        retrieve every info about cam
        :return: no Return
        """
        self.getNbrReadoutModes()
        self.getCurrentReadoutMode()
        self.getReadoutTime()
        self.getSize()
        self.getWaitingTime()
        self.getExposureTime()
        self.getComponentTemp(0)
        self.getComponentTemp(1)
        self.getComponentTemp(2)
        self.getComponentTemp(3)
        self.getShutterMode()
        self.getCalibratedEmGain()
        self.getCalibratedEmGainRange()
        self.getCalibratedEmGainTempRange()
        self.getRawEmGain()
        self.getRawEmGainRange()
        self.getTargetDetectorTemp()
        self.getTargetDetectorTempRange()

    def updateCam(self):
        """
        retrieve info about cam
        :return: Ǹo return
        """
        pass

    def purgeBuffer(self):
        for i in range(self.nbBuff):
            self.read()

    def setSquareBinning(self, bin):
        """
        sets binning mode and actualise data in cam class
        :param bin:
        :return:
        """
        try:
            error = ncCamSetBinningMode(self.ncCam, bin, bin)
            if error:
                raise NuvuException(error)
            error = ncCamGetBinningMode(self.ncCam, byref(self.binx), byref(self.biny))
            if error:
                raise NuvuException(error)
            self.getSize()

        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def flushReadQueue(self):
        """
        flushes all images acquired prior to this call
        :param self:
        :return:
        """
        try:
            error = ncCamFlushReadQueue(self.ncCam)
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    # endregion
