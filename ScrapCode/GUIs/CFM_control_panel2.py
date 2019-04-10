# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 15:01:38 2018

@author: mccambria
"""

######################### Hardcoded parameters here #########################

# The IP address adopted by the PulseStreamer is hardcoded. See the lab wiki
# for information on how to change it
PULSER_IP = "128.104.160.11"
# The name of the DAQ is assigned in MAX
DAQ_NAME = "dev1"

########################### Import statements here ###########################

import sys
import os
# By default, python looks for modules in ...installDirectoy\Lib\site-packages
# We can tell it to additionally look elsewhere by appending a path to sys.path
# pulse_streamer_grpc does not live in the default directory so we need to add
# that path before we import the library
sys.path.append(os.getcwd() + '/PulseStreamerExamples/python/lib')
import kivy
kivy.require("1.0.7")
import NV_utils
import galvo_sweep
from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.properties import StringProperty

######################### Simple custom widgets #########################

class PanelSection(BoxLayout):
    pass

class PanelSectionHeading(BoxLayout):
    pass

class PanelSubsection(BoxLayout):
    pass

class ShortButton(Button):
    pass

class ParameterGrid(GridLayout):
    pass

############################## Main widget ##############################

class ControlPanel2(TabbedPanel):
    
    daqGalvoXAOStr = StringProperty("checkity check", rebind=True)
    
    def open_error_popup(message):
        titleMessage = "Error: press escape or click outside popup to close"
        popup = Popup(title=titleMessage, 
                      content=Label(text=message), 
                      size_hint=(0.5, 0.5), 
                      auto_dismiss=True)
        popup.open()
    
    def get_wiring(self):
        pass
        
    
    def galvo_sweep(self):
        
        print(self.daqGalvoXAOStr)
        return
        
        # Make sure we have a pulser
        pulser = NV_utils.get_pulser(PULSER_IP)
        if pulser == None:
            self.open_error_popup("Couldn't get PulseStreamer at" + PULSER_IP)
            return
        
        galvo_sweep.main(pulser, DAQ_NAME, 
                         samplesPerDim = 100, 
                         resolution = 0.01, 
                         period = 0.25, 
                         offset = [-0.5, -0.5],
                         initial = [0.0, 0.0])
    
    def pulser_all_zero(self):
        
        # Make sure we have a pulser
        pulser = NV_utils.get_pulser(PULSER_IP)
        if pulser == None:
            self.open_error_popup("Couldn't get PulseStreamer at" + PULSER_IP)
            return
        
        NV_utils.pulser_all_zero(pulser)
        
    def pulser_square_wave(self):
        
        # Make sure we have a pulser
        pulser = NV_utils.get_pulser(PULSER_IP)
        if pulser == None:
            self.open_error_popup("Couldn't get PulseStreamer at" + PULSER_IP)
            return
        
        # Get the period
        periodInput = self.ids.periodInput
        period = float(periodInput.text)
        periodNano = int(period * (10**9)) # integer period in ns
        
        # Get the channels
        toggleList = []
        digitalToggles = self.ids.digitalTogglesBox.children
        for toggle in digitalToggles:
            if toggle.state == "down":
                toggleList.append("ch" + toggle.text)
        
        # Run the sequence indefinitely
        NV_utils.pulser_square_wave(pulser, periodNano, toggleList, -1)

############################## App definition ##############################
    
class ControlPanel2App(App):

    def build(self):
        return ControlPanel2()

################################### Main ###################################

if __name__ == "__main__":
    ControlPanel2App().run()