# -*- coding: utf-8 -*-
"""
Python interface for Nuvu camera with calls to ctypes functions. Modified from file of same name
provided by Nuvu

Obtained on August 1st, 2023

@author: Nuvu
"""

from .nc_api import *
import numpy as np
import sys


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

    def errorHandling(self, error):
        """
        Method that ensures an appropriate reaction to errors. So far, the function crashes the program, closes the driver, and exits the software.
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
            print(
                "Code d'erreur: "
                + str(error)
                + ". \n Se référer au fichier erreur.h du SDK de Nuvu."
            )
            self.close_cam(noRaise=True)
            sys.exit("Erreur d'exécution du driver Nuvu")

    def open_cam(self, num_buffer=1):
        """
        Opens the connection with the camera (assumes there is only one available).
        If the class has been initialized with the camera's MAC address, the method will try to connect directly to that camera.
        :param nbBuff: Number of buffers initialized in the Nuvu API. (Number of images stored in memory at a time, I think...)
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

    def close_cam(self, noRaise=False):
        """
        Function that closes the camera driver.
        :param noRaise: Internal parameter that allows not raising an error if the driver is already closed.
        :return: None
        """
        try:
            error = ncCamClose(self.ncCam)
            if error and not noRaise:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def setReadoutMode(self, mode):
        """
        Allows selecting the camera's readout mode.
        :param mode: (int) mode nothing=0, EM = 1, CONV = 2
        :return: None
        """
        try:
            error = ncCamSetReadoutMode(self.ncCam, mode)
            if error:
                raise NuvuException(error)

        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

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

    def setTimeout(self, timeout):
        """
        Méthode qui permet de sélectionner un temps de timeout qui représente le temps avant que le driver déclare une
        erreur si il attend qu'une nouvelle image entre dans un buffer.
        :param timeout: (float) temps d'attente (en ms)
        :return: None
        """
        try:
            error = ncCamSetTimeout(self.ncCam, int(timeout))
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def setShutterMode(self, mode):
        """
        Méthode qui sélectionne le mode de l'obturateur
        :param mode: (int) mode de l'obturateur
        :return: None
        """
        try:
            error = ncCamSetShutterMode(self.ncCam, mode)
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

    def getSize(self):
        """
        Méthode qui récupère la hauteur et la largeur des images de la caméra, ces deux valeurs aux attributs height
        et width de la classe caméra
        :return: None
        """
        try:
            error = ncCamGetSize(self.ncCam, byref(self.width), byref(self.height))
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def camStart(self, nbrImg=0):
        """
        Méthode qui démarre l'acquisition d'images sur la caméra et les envoie au buffer
        :param nbrImg: Détermine le nombre d'images que le driver prendra, si 0 alors l'acquisition est continue
        :return: None
        """
        try:
            error = ncCamStart(self.ncCam, nbrImg)
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def camAbort(self):
        """
        Méthode qui arrète toute acquisition sur la caméra
        :return: None
        """
        try:
            error = ncCamAbort(self.ncCam)
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def read(self):
        """
        Méthode qui lit la prochaine image dans le buffer et l'envoie en mémoire.
        :return: None
        """
        try:
            error = ncCamReadChronological(self.ncCam, self.ncImage, byref(c_int()))
            if error:
                raise NuvuException(error)
        except NuvuException as nuvuException:
            self.errorHandling(nuvuException.value())

    def get_img_array(self):
        """
        Méthode qui apelle read() puis cast le pointeur vers l'image en array de 16 bit que l'on copie vers une autre
        partie de la mémoire.
        :return: None
        """
        self.read()
        return np.copy(
            np.ctypeslib.as_array(
                cast(self.ncImage, POINTER(c_uint16)),
                (self.width.value, self.height.value),
            )
        )

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

    def setTargetDetectorTemp(self, temp):
        """
        Méthode permettant de donner à la caméra la température visée du détecteur
        :param temp: (float) Température visée
        :return: None
        """
        try:
            error = ncCamSetTargetDetectorTemp(self.ncCam, c_double(temp))
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

    def getCurrentReadoutMode(self):
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

    def getNbrReadoutModes(self):
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
