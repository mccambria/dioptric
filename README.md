# KolkowitzNVExperiment
This repository contains the code and data for the Kolkowitz lab's experiments with NV centers. I recommend using Spyder as an IDE. In order to run the code you have to add your local repository's directory to PYTHONPATH so that Python knows to look there for imports. To do this in Spyder go to Tools => PYTHONPATH manager and add in the directory. The directory on my computer is C:\Users\Matt\GitHub\KolkowitzNVExperiment. To properly show Matplotlib figures in Spyder, go to Tools => Preferences => IPython console => Graphics and set the backend to automatic.

You should be able to perform most necessary tasks with cfm_control_panel.

Here are some things that I think people working on this in the future should be aware of.

1. Don't work on a GUI until the thing you're making the GUI for is just about set in stone. If you make subsequent changes to more than just the GUI, then chances are you'll have to change the GUI as well, which means double work. I've found it best to abstract everything as far as possible in the IDE itself, like what I did with cfm_control_panel. Creating a GUI wrapper around that file will be relatively simple when the time comes.
2. For streaming pulses, use Swabian's super useful Sequence class. It allows you to construct individual sequences for each channel and then combine them into one unified sequence that the PulseStreamer can actually read. 
3. When using the Sequence class, be sure to specify durations in numpy's int64 type. This is what Sequence assumes you will be using. If you use something different (for example Python's built-in int type), then you can run into overflows pretty quickly since the unit is nanoseconds. These overflows don't throw errors; the sequence will just be missing any pulse that has an absolute timestamp past the overflow. 
4. The PulseStreamer has a memory of 10^6 pulses. It's good to check that you haven't exceeded this once you get the unified sequence with GetSequence().
