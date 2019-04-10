# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 15:01:38 2018

@author: mccambria
"""

######################### Hardcoded parameters here #########################

# The IP address adopted by the PulseStreamer is hardcoded. See the lab wiki
# for information on how to change it
PULSE_STREAMER_IP = "128.104.160.11"
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
from pulse_streamer_grpc import PulseStreamer
import nidaqmx
import NV_utils
import sweep_galvo
from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.label import Label

############################## Functions ##############################

def open_error_popup(message):
    titleMessage = "Error: press escape or click outside the popup to close"
    popup = Popup(title=titleMessage, 
                  content=Label(text=message), 
                  size_hint=(0.5, 0.5), 
                  auto_dismiss=True)
    popup.open()

def get_pulser(self):
    global PULSER
    try:
        return PULSER
    except:
        try:
            PULSER = PulseStreamer(PULSE_STREAMER_IP)
            PULSER.isRunning()
            return PULSER
        except:
            del PULSER
            open_error_popup("No PulseStreamer found at IP-address: " + \
                             PULSE_STREAMER_IP)
            return None

######################### Simple custom widgets #########################

class Panel(BoxLayout):
    pass

class PanelHeading(BoxLayout):
    pass

class PanelSection(BoxLayout):
    pass

class ShortButton(Button):
    pass

class ParameterGrid(GridLayout):
    pass

############################## Panel widgets ##############################

class DAQPanel(Panel):
    pass

class PulseStreamerPanel(Panel):
    
    def pulser_square_wave(self):
        
        # Make sure we have a pulser
        pulser = get_pulser()
        if pulser == None:
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
        
    def pulser_all_zero(self):
        
        # Make sure we have a pulser
        pulser = get_pulser()
        if pulser == None:
            return
        
        NV_utils.pulser_all_zero(pulser)
        
class MainPanel(TabbedPanel):
    
    def sweep_galvo(self):
        
        # Make sure we have a pulser
        pulser = get_pulser()
        if pulser == None:
            return
        
        sweep_galvo.main(pulser, DAQ_NAME, 
                         samplesPerDim = 100, 
                         resolution = 0.01, 
                         period = 0.25, 
                         offset = [-0.5, -0.5],
                         initial = [0.0, 0.0])

############################## Main widget ##############################

class ControlPanel(BoxLayout):
    pass

############################## App definition ##############################
    
class ControlPanelApp(App):

    def build(self):
        return ControlPanel()

################################### Main ###################################

if __name__ == "__main__":
    ControlPanelApp().run()