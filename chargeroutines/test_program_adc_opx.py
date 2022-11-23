import time
from qm import SimulationConfig
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from opx_configuration_file import *
import matplotlib.pyplot as plt
import numpy as np
import numpy
from qualang_tools.units import unit
u = unit()


readout_time = 5000000
config_opx['pulses']['readout_pulse']['length'] = readout_time

#readout pulse length is 200

with program() as seq:

    times1 = declare(int, size=1000)
    times2 = declare(int, size=1000)
    counts1 = declare(int)
    counts2 = declare(int)
    counts_st = declare_stream()
    times_st = declare_stream()
    # update_frequency('Yb', int(0))
    i = declare(int)
    adc_st1 = declare_stream(adc_trace=True)
    adc_st2 = declare_stream(adc_trace=True)
    
    laseron = int(readout_time // 4)
    
    play("laser_ON_DIGITAL","cobolt_515",duration=laseron)
    
    measure('readout', 'do_apd_0_gate', adc_st1, time_tagging.analog(times1, readout_time, counts1))
    # measure('readout', 'do_apd_1_gate', adc_st2, time_tagging.analog(times2, readout_time, counts2))
    save(counts1, counts_st)
    # with for_(i, 0, i < counts, i + 1):
    #     save(times[i], times_st)
    

    with stream_processing():
        counts_st.save_all('counts')
        # times_st.save_all('times')
        adc_st1.input1().save('adc1')
        # adc_st2.input2().save('adc2')
        
        
qmm = QuantumMachinesManager(qop_ip)
qm = qmm.open_qm(config_opx)
job = qm.execute(seq)

# simulation_duration = 1200 // 4 # clock cycle units - 4ns
# job_sim = qm.simulate(seq, SimulationConfig(simulation_duration))

# job_sim.get_simulated_samples().con1.plot()
# job_sim.get_simulated_samples().adc_trace.plot()

res_handles = job.result_handles
res_handles.wait_for_all_values()
counts = res_handles.get("counts").fetch_all()
# times = res_handles.get("times").fetch_all()
adc1_single_run = u.raw2volts(res_handles.get("adc1").fetch_all())
# adc2_single_run = u.raw2volts(res_handles.get("adc2").fetch_all())
print('')
print(counts)
# print('')
# print(times)
plt.figure()
plt.title("Single run")
plt.plot(adc1_single_run, label="Input 1")
# plt.plot(adc2_single_run, label="Input 2")
plt.xlabel("Time [ns]")
plt.ylabel("Signal amplitude [V]")
plt.legend()
plt.xlim(0,readout_time)

