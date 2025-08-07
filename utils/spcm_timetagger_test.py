# import time

# import matplotlib.pyplot as plt
# from TimeTagger import Correlation, createTimeTagger

# # --- TimeTagger setup ---
# tagger = createTimeTagger()
# tagger.setTriggerLevel(2, 0.25)
# tagger.setInputDelay(3, 123)

# # Create Correlation object
# corr = Correlation(tagger, 2, 3, binwidth=10, n_bins=1000)  # 10 ps bins

# # Run for 1 second
# corr.startFor(int(10e12), clear=True)  # 1e12 ps = 1 s
# corr.waitUntilFinished()


# # --- Get and visualize data ---
# data = corr.getData()
# # binwidth = corr.getBinwidth()  # in ps
# binwidth = 10  # in ps
# n_bins = len(data)

# # Generate time axis (in ns for readability)
# time_ps = [(i - n_bins // 2) * binwidth for i in range(n_bins)]
# time_ns = [t * 1e-3 for t in time_ps]  # convert to ns

# # --- Plot ---
# plt.figure(figsize=(8, 4))
# plt.plot(time_ns, data, drawstyle="steps-mid")
# plt.xlabel("Time delay (ns)")
# plt.ylabel("Coincidence counts")
# plt.title("g(2) Correlation Histogram (ch2 vs ch3)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Connect to the Time Tagger
# tt = TimeTagger.createTimeTagger()

# # Set the input channel connected to the SPCM (e.g., channel 1)
# channel = 1

# # Count photons for a short time
# counter = TimeTagger.Counter(tt, channel, binwidth=1000000, n_values=1)  # 1 ms bin


# time.sleep(1)

# # Read the count value
# print("Photon count in 1 ms:", counter.getData())


import matplotlib.pyplot as plt
from pulsestreamer import PulseStreamer
from TimeTagger import Correlation, createTimeTagger

# Setup Pulse Streamer
ps = PulseStreamer()
ps.reset()

# PulseStreamer Ch0 and Ch1 output TTL pulses (e.g., 100 ns delay between them)
ps.setOutput(0, 1)  # Ch0 HIGH
ps.delay(100)  # 100 ns
ps.setOutput(0, 0)  # Ch0 LOW

ps.setOutput(1, 1)  # Ch1 HIGH
ps.delay(100)  # 100 ns
ps.setOutput(1, 0)  # Ch1 LOW

ps.sendLoop()  # Continuously repeat this pattern

# Connect to Time Tagger
tagger = createTimeTagger()
tagger.setTriggerLevel(2, 1.0)
tagger.setTriggerLevel(3, 1.0)

# Correlation between channels 2 and 3
corr = Correlation(tagger, 2, 3, binwidth=1000, n_bins=1000)  # 1 ns bins

# Collect data
corr.startFor(int(1e12))  # 1 s
corr.waitUntilFinished()
data = corr.getData()
binwidth = corr.getBinwidth()  # in ps
n_bins = len(data)

# Plot
x = [(i - n_bins // 2) * binwidth * 1e-3 for i in range(n_bins)]  # ns
plt.plot(x, data)
plt.xlabel("Delay (ns)")
plt.ylabel("Coincidence count")
plt.title("g2 correlation test with Pulse Streamer")
plt.grid()
plt.show()
