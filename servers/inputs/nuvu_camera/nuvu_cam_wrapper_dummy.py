# file: nuvu_cam_wrapper.py
# author: Guillaume Allain
# date: 28/04/18
# desc: Wrapper that emulates nuvu_cam_wrapper for testing

import numpy as np

class Nuvu_wrapper_error(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Nuvu_cam_wrapper():
    def __init__(self, targetDetectorTemp=-35, readoutMode=1, binning=1,
                 exposureTime=0.75, fps=250):
        self.__fps = fps
        # self.openCam(nbBuff=4)
        # self.setReadoutMode(readoutMode)
        # self.getCurrentReadoutMode()
        # self.setSquareBinning(binning)
        # self.camInit()
        # self.setTargetDetectorTemp(targetDetectorTemp)
        # self.getTargetDetectorTemp()
        self.set_target_detector_temp(targetDetectorTemp)
        self.readoutTime = 0.02
        self.exposureTime = exposureTime
        self.waitingTime = 0.3
        self.change_exposure_time(exposureTime)
        self.em_gain = 1
        self.isrunning = False
        # self.setTimeout(self.exposureTime.value + self.waitingTime.value
        #                 + self.readoutTime.value + 500.0)
    #Deactivated disconnect_if_error
    def disconnect_if_error(f):
        """Decorator that disconnect the camera if the method fails"""
        def func(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                # args[0].closeCam()
                # print('Successfull emergency disconnect of Nuvu Camera')
                raise e from None
        return func

    @property
    def real_fps(self):
        return self.__real_fps

    @disconnect_if_error
    def millisecond_to_fps(self,arg):
        return 1/(arg/1000)

    @disconnect_if_error
    def fps_to_millisecond(self,arg):
        return (1/arg)*1000

    @disconnect_if_error
    def set_fps(self,fps):
        current_exp_time = self.exposureTime
        current_read_time = self.readoutTime
        self.waitingTime = (self.fps_to_millisecond(fps)
                            - current_exp_time
                            - current_read_time)
        self.__real_fps = self.millisecond_to_fps(self.exposureTime
                                                  + self.waitingTime
                                                  + self.readoutTime)

    @disconnect_if_error
    def change_exposure_time(self, new_exposure_time):
        if new_exposure_time >= (self.fps_to_millisecond(self.__fps)
                                -self.readoutTime):
            raise Nuvu_wrapper_error("Exposure time too large, change FPS first")
        self.exposureTime = new_exposure_time
        self.set_fps(self.__fps)

    @disconnect_if_error
    def set_calibrated_em_gain(self, new_em_gain):
        if new_em_gain<1 or new_em_gain>5000:
            raise Nuvu_wrapper_error("Em gain should be smaller than 5000 and larger than 0")
        print('Nuvu128 em gain set to: '+ str(new_em_gain))
        self.em_gain = new_em_gain

    @disconnect_if_error
    def change_fps(self, new_fps):
        if self.fps_to_millisecond(new_fps) <= (self.exposureTime
                                               + self.readoutTime):
            raise Nuvu_wrapper_error("FPS to small, change exposure time first")
        self.set_fps(new_fps)

    @disconnect_if_error
    def get_image64(self):
        #get directly 64bit image
        return self.get_image().astype(np.float64)

    @disconnect_if_error
    def get_image(self):
        #get a uint16 image
        if self.isrunning == False:
            raise(Error('Camera needs to be running'))
        return np.ones((128,128)).astype(np.uint16)

    @disconnect_if_error
    def get_bias(self):
        return self.get_image()

    @disconnect_if_error
    def get_bias64(self):
        return self.get_bias().astype(np.float64)

    @disconnect_if_error
    def camStart(self):
        print('Nuvu_128 started')
        self.isrunning = True

    @disconnect_if_error
    def camStop(self):
        print('Nuvu_128 stopped')
        self.isrunning = False

    @disconnect_if_error
    def get_calibrated_em_gain(self):
        return self.em_gain

    def closeCam(self):
        self.isrunning = False
        pass

    @disconnect_if_error
    def get_component_temp(self,component):
        """
        Méthode qui récupère la température du composant spécifié et la
        stoque dans la valeur associée
        :param comp: identifiant du composant. 0=CCD, 1=controleur,
        2=powerSupply, 3=FGPA, 4=heatSink
        :return: None
        """
        list = [-35.0, -10.0,
                20.0, 30.0,
                40.0]
        return list[component]

    @disconnect_if_error
    def get_ccd_temp(self):
        '''Récupère et retourne la valeur de température du CCD'''
        return self.get_component_temp(0)

    @disconnect_if_error
    def set_target_detector_temp(self,target):
        self.targetDetectorTemp=target
        print('target detector temp'+str(self.targetDetectorTemp))

    def get_target_detector_temp(self):
        return self.targetDetectorTemp

if __name__ == '__main__':
    cam= Nuvu_cam_wrapper_dummy()
    cam.set_calibrated_em_gain(2)
    cam.get_image()
