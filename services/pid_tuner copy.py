# -*- coding: utf-8 -*-
"""
Created on July 3rd, 2025

@author: Saroj B Chand

Live plot for laser power data from NI DAQ logger
"""

import time
import math
import logging
from datetime import datetime

# Optional libraries for specific functionalities
try:
    import numpy as np
except ImportError:
    np = None  # numpy is used for calculations; the script will handle if not available
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import requests  # for TempStick API calls
except ImportError:
    requests = None
try:
    import labrad  # for LabRAD connection if available
except ImportError:
    labrad = None
try:
    import telnetlib
except ImportError:
    telnetlib = None
try:
    import serial
except ImportError:
    serial = None

# Configuration Section
USE_LABRAD = False  # Set True to use LabRAD connection, False for direct connection (Telnet or serial)
LABRAD_SERVER_NAME = "ptc10"  # Name of the LabRAD server for PTC10, if applicable
PTC10_HOST = "192.168.1.50"   # IP address of the PTC10 (for Telnet/Ethernet connection)
PTC10_PORT = 23              # Telnet port (default 23)
SERIAL_PORT = "COM3"         # Serial port name if using RS-232 (e.g., "COM3" on Windows or "/dev/ttyUSB0" on Linux)
SERIAL_BAUD = 9600           # Baud rate for serial (if RS-232, likely 115200 or 9600 based on PTC10 settings)
OUTPUT_CHANNEL = "Out1"      # PTC10 output channel name to tune (e.g., "Out1" corresponds to first output)
INPUT_CHANNEL = "4A"         # PTC10 input channel name (sensor) that the above output controls (e.g., "4A")
SETPOINT = 50.0              # Desired temperature setpoint in °C (adjust as needed for your test)
TOLERANCE = 0.5              # Settling tolerance (acceptable deviation from setpoint in °C)
SETTLING_TIME_WINDOW = 30.0  # Time (s) that temperature must remain within tolerance to be considered settled
MAX_RUN_TIME = 600.0         # Max time (s) to wait for settling before aborting a trial (e.g., 10 minutes)
COOLDOWN_TIME = 300.0        # Time (s) to cool down between trials (or time to wait after turning off output)
COOLDOWN_TEMP = None         # Optional: temperature to cool down to before next trial (if None, use fixed time)
DATA_LOG_FILE = "pid_tuning_results.csv"  # File to log summary of results
LOG_DATA_DIR = "pid_logs/"   # Directory to save detailed logs of each trial (if needed)
ENABLE_PLOTTING = True       # Whether to generate plots at the end of the tuning run

# PID sweep ranges (define either lists of values or use range generation)
P_VALUES = [1.0, 2.0, 5.0, 10.0]      # Example P gains to test
I_VALUES = [0.0, 0.1, 0.5, 1.0]       # Example I gains to test (0.0 to test P-only control)
D_VALUES = [0.0, 0.1, 0.5]           # Example D gains to test

# Auto-tuning mode configuration
AUTO_TUNE_MODE = None  # Set to "ZN" for Ziegler-Nichols, "relay" for Åström-Hägglund relay tuning, or None to skip
P_INITIAL = 0.1        # Starting P for ZN method (small value)
P_STEP = 0.1           # Increment for P in ZN search
ZN_MAX_P = 100.0       # Maximum P to try for ZN before giving up (to avoid infinite loop)
RELAY_HIGH_OUTPUT = 100.0  # High output level (%) for relay tuning (if using normalized 0-100 scale)
RELAY_LOW_OUTPUT = 0.0     # Low output level (%) for relay tuning (or negative if cooling available)
RELAY_PERIOD_ESTIMATE = 60.0  # Expected oscillation period in seconds (used for timeout in relay test)

# External TempStick API configuration
USE_TEMPSTICK = True
TEMPSTICK_API_URL = "https://api.tempstick.com/latest"  # Placeholder URL for current reading
TEMPSTICK_API_KEY = None  # API key or credentials if required by the Temp Stick API

# Set up logging to console and file
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Utility class for communicating with PTC10
class PTC10Interface:
    def __init__(self, use_labrad=False):
        """Initialize the interface to PTC10 via LabRAD or direct connection (Ethernet or serial)."""
        self.use_labrad = use_labrad
        self.telnet = None
        self.serial = None
        self.labrad_server = None
        if use_labrad:
            if labrad is None:
                raise RuntimeError("pylabrad not installed or not available, cannot use LabRAD interface.")
            # Connect to LabRAD and get the PTC10 server
            try:
                self.cxn = labrad.connect()
                self.labrad_server = self.cxn[LABRAD_SERVER_NAME]
                logger.info("Connected to PTC10 via LabRAD server '%s'.", LABRAD_SERVER_NAME)
            except Exception as e:
                raise RuntimeError(f"Failed to connect to LabRAD server '{LABRAD_SERVER_NAME}': {e}")
        else:
            # Try Ethernet (Telnet) first if host is provided, else fallback to serial
            if PTC10_HOST:
                if telnetlib is None:
                    raise RuntimeError("telnetlib not available for Telnet connection.")
                try:
                    self.telnet = telnetlib.Telnet(PTC10_HOST, PTC10_PORT, timeout=10)
                    logger.info("Opened Telnet connection to PTC10 at %s:%d.", PTC10_HOST, PTC10_PORT)
                    # The PTC10 might not send a prompt; it's a simple line interface. We can proceed to send commands.
                except Exception as e:
                    raise RuntimeError(f"Failed to open Telnet connection to {PTC10_HOST}:{PTC10_PORT}: {e}")
            else:
                # Use serial if host not specified
                if serial is None:
                    raise RuntimeError("pySerial not available for serial connection.")
                try:
                    self.serial = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
                    logger.info("Opened serial connection to PTC10 on port %s.", SERIAL_PORT)
                except Exception as e:
                    raise RuntimeError(f"Failed to open serial port {SERIAL_PORT}: {e}")

    def send_command(self, cmd):
        """Send a command string to the PTC10 and return the response (if any)."""
        cmd_str = cmd.strip()
        if not cmd_str.endswith('?'):
            # It's a set command or action command (no immediate response expected)
            terminator = "\r\n"
        else:
            # It's a query command (expects a response)
            terminator = "\r\n"
        full_cmd = cmd_str + terminator
        logger.debug("Sending command: %s", cmd_str)
        try:
            if self.use_labrad:
                # If LabRAD server provides a generic write/read interface:
                # We assume the LabRAD server for PTC10 might have methods like query() or write() implemented.
                # If not, we'd need to adjust according to actual server implementation.
                if cmd_str.endswith('?'):
                    # query expecting response
                    resp = self.labrad_server.query(cmd_str)
                    # LabRAD query might return a byte string or already a str
                    resp_str = resp if isinstance(resp, str) else resp.decode('utf-8')
                    logger.debug("Response: %s", resp_str.strip())
                    return resp_str.strip()
                else:
                    # set command
                    self.labrad_server.write(cmd_str)
                    return None
            elif self.telnet:
                # Telnet mode: send bytes and read response if query
                self.telnet.write(full_cmd.encode('ASCII'))
                if cmd_str.endswith('?'):
                    # Read until newline (the PTC10 typically returns a line termination)
                    resp_bytes = self.telnet.read_until(b"\r\n", timeout=1)
                    resp = resp_bytes.decode('ASCII').strip()
                    logger.debug("Response: %s", resp)
                    return resp
                else:
                    # For set commands, the PTC10 might not send any confirmation (silent on success).
                    # We can optionally read any echoed text or just proceed.
                    return None
            elif self.serial:
                # Serial mode
                self.serial.write(full_cmd.encode('ASCII'))
                if cmd_str.endswith('?'):
                    resp_bytes = self.serial.readline()  # read one line from serial
                    resp = resp_bytes.decode('ASCII').strip()
                    logger.debug("Response: %s", resp)
                    return resp
                else:
                    return None
            else:
                raise RuntimeError("No connection available to send command.")
        except Exception as e:
            logger.error("Error sending command '%s': %s", cmd_str, e)
            raise

    def close(self):
        """Close the connection to PTC10."""
        if self.telnet:
            try:
                self.telnet.close()
            except Exception:
                pass
        if self.serial:
            try:
                self.serial.close()
            except Exception:
                pass
        if self.use_labrad:
            try:
                self.cxn.disconnect()
            except Exception:
                pass

# Utility function for TempStick external sensor
def get_tempstick_temperature():
    """Fetch the current temperature from the external Temp Stick sensor via its API."""
    if not USE_TEMPSTICK or TEMPSTICK_API_URL is None:
        return None
    if requests is None:
        logger.warning("Requests library not available, cannot query TempStick API.")
        return None
    try:
        # For example, if the API requires an API key or auth, you might use:
        # resp = requests.get(TEMPSTICK_API_URL, params={'api_key': TEMPSTICK_API_KEY}, timeout=5)
        resp = requests.get(TEMPSTICK_API_URL, timeout=5)
        resp.raise_for_status()
        data = resp.json()  # assuming the API returns JSON
        # Extract temperature from JSON. The exact key will depend on the API format.
        # Here we assume data contains something like {"temperature": 25.3}
        temp = None
        if isinstance(data, dict):
            # Attempt common keys
            for key in ['temperature', 'temp', 'Temperature', 'Temp']:
                if key in data:
                    temp = float(data[key])
                    break
        else:
            # If data is not a dict, handle accordingly (maybe data itself is the value)
            try:
                temp = float(data)
            except:
                temp = None
        logger.debug("TempStick temperature: %s °C", temp)
        return temp
    except Exception as e:
        logger.error("Failed to get temperature from TempStick API: %s", e)
        return None

# Performance evaluation helper
def analyze_performance(time_points, temp_points, setpoint):
    """Compute performance metrics (settling time, steady-state std dev) from a temperature time series."""
    settling_idx = None
    settling_time = None
    steady_state_std = None
    # Determine when the temperature enters and stays in the tolerance band around setpoint
    lower_bound = setpoint - TOLERANCE
    upper_bound = setpoint + TOLERANCE
    # Find the first index where temperature is within bounds
    for i, T in enumerate(temp_points):
        if lower_bound <= T <= upper_bound:
            # Check if it stays within bounds for the next SETTLING_TIME_WINDOW seconds
            t_enter = time_points[i]
            # Find where time >= t_enter + window
            t_end_required = t_enter + SETTLING_TIME_WINDOW
            # Assume time_points is sorted and fairly continuous
            # Check all points from i to end of array that fall within the window
            within = True
            for j in range(i, len(temp_points)):
                if time_points[j] > t_end_required:
                    break
                if not (lower_bound <= temp_points[j] <= upper_bound):
                    within = False
                    break
            if within:
                settling_idx = i
                settling_time = t_enter - time_points[0]
                break
    # If settled, compute std dev of last part of data (steady state)
    if settling_idx is not None:
        # Take data from settling_idx to end (or maybe a fixed duration after settling_idx)
        steady_state_temps = [T for j, T in enumerate(temp_points) if j >= settling_idx]
        if steady_state_temps:
            if np:
                steady_state_std = float(np.std(steady_state_temps))
            else:
                # Manual std dev calculation if numpy not available
                mean_val = sum(steady_state_temps) / len(steady_state_temps)
                variance = sum((x-mean_val)**2 for x in steady_state_temps) / len(steady_state_temps)
                steady_state_std = math.sqrt(variance)
    else:
        # Not settled within run
        settling_time = None
        # We can still compute std dev of entire run as an indicator of oscillation magnitude
        if temp_points:
            if np:
                steady_state_std = float(np.std(temp_points))
            else:
                mean_val = sum(temp_points) / len(temp_points)
                variance = sum((x-mean_val)**2 for x in temp_points) / len(temp_points)
                steady_state_std = math.sqrt(variance)
    return settling_time, steady_state_std

# Optional: Auto-tuning functions for ZN and relay methods
def autotune_ziegler_nichols(ptc: PTC10Interface):
    """
    Perform Ziegler-Nichols tuning: find the ultimate gain Ku and period Tu by increasing P until sustained oscillations.
    Returns (P_recommended, I_recommended, D_recommended) using classic Z-N formula.
    """
    logger.info("Starting Ziegler-Nichols auto-tuning...")
    # Set I = 0, D = 0 for P-only test
    ptc.send_command(f'{OUTPUT_CHANNEL}.PID.I 0')
    ptc.send_command(f'{OUTPUT_CHANNEL}.PID.D 0')
    # Ensure loop is on
    ptc.send_command(f'{OUTPUT_CHANNEL}.PID.Mode On')
    Ku = None
    Tu = None
    current_p = P_INITIAL
    oscillation_times = []  # to record times of oscillation peaks or zero-crossings
    prev_temp = None
    increasing = None  # track if temperature is going up or down for peak detection
    start_time = time.time()
    last_flip_time = start_time
    # Increase P gradually until oscillation sustained
    while current_p <= ZN_MAX_P:
        ptc.send_command(f'{OUTPUT_CHANNEL}.PID.P {current_p}')
        logger.info("Testing P = %.3f for oscillation...", current_p)
        oscillation_times.clear()
        prev_temp = None
        increasing = None
        # Monitor for a few oscillation cycles or a timeout
        oscillation_detected = False
        t0 = time.time()
        while time.time() - t0 < 2 * RELAY_PERIOD_ESTIMATE:  # monitor for up to two expected periods
            temp = None
            try:
                resp = ptc.send_command(f'{INPUT_CHANNEL}?')  # query current temperature
                if resp is not None:
                    temp = float(resp)
                else:
                    # If no direct response (some PTC10 commands might not return with '?'),
                    # we can use getOutput or other mechanism if needed.
                    pass
            except Exception as e:
                logger.error("Error reading temperature during ZN tuning: %s", e)
                break
            if temp is None:
                continue
            # Detect peaks: check sign changes in the derivative
            if prev_temp is not None:
                if temp > prev_temp:
                    # temperature rising
                    if increasing is False:
                        # was decreasing, now increasing -> trough point
                        oscillation_times.append(time.time())
                    increasing = True
                elif temp < prev_temp:
                    # temperature falling
                    if increasing is True:
                        # was rising, now falling -> peak point
                        oscillation_times.append(time.time())
                    increasing = False
            prev_temp = temp
            time.sleep(0.5)  # small delay between reads (2 Hz sampling)
            # If enough peak/trough timestamps collected to judge sustained oscillation:
            if len(oscillation_times) >= 6:  # e.g. 3 full cycles (peak-trough-peak counts as 2)
                oscillation_detected = True
                break
        if oscillation_detected:
            # Estimate oscillation period from collected times (e.g. average difference between alternate peaks)
            if len(oscillation_times) >= 6:
                # Take difference between every two consecutive peaks (or troughs)
                periods = []
                for k in range(2, len(oscillation_times)):
                    period = oscillation_times[k] - oscillation_times[k-2]  # difference between every other timestamp
                    periods.append(period)
                if periods:
                    Tu = sum(periods) / len(periods)
            Ku = current_p
            logger.info("Sustained oscillation detected at P = %.3f. Measured oscillation period ~ %.2f s.", Ku, Tu or 0)
            break
        else:
            current_p += P_STEP
    # Stop PID for safety
    ptc.send_command(f'{OUTPUT_CHANNEL}.PID.Mode Off')
    if Ku is None or Tu is None or Tu == 0:
        logger.warning("Ziegler-Nichols tuning failed to find sustained oscillation within range.")
        return None
    # Apply classic Ziegler-Nichols formula for PID
    P_recommended = 0.6 * Ku
    I_recommended = 2 * P_recommended / Tu   # Ki = Kp / Ti, Ti = Tu/2 -> Ki = 2*Kp/Tu
    D_recommended = 0.125 * Tu * P_recommended  # Kd = Kp * (Tu/8)
    logger.info("Z-N recommended gains: P = %.3f, I = %.3f, D = %.3f", P_recommended, I_recommended, D_recommended)
    return P_recommended, I_recommended, D_recommended

def autotune_relay(ptc: PTC10Interface):
    """
    Perform Åström–Hägglund relay auto-tuning to estimate Ku and Tu.
    Returns (P_recommended, I_recommended, D_recommended) using classic Z-N formula.
    """
    logger.info("Starting relay auto-tuning...")
    # Turn off PID and use manual control for relay test (set output in open-loop)
    ptc.send_command(f'{OUTPUT_CHANNEL}.PID.Mode Off')
    # We assume we can control the output manually via OutX.Value if needed. For simplicity,
    # we'll simulate relay by adjusting the PID setpoint around the actual setpoint to force on/off.
    # If direct manual heater control is possible via OutX.Value, that would be ideal.
    # Here, we'll adjust setpoint to simulate full heating and cooling.
    high_sp = SETPOINT + 10.0  # a setpoint above actual to force heating
    low_sp = SETPOINT - 10.0   # a setpoint below actual to force cooling/off
    oscillation_times = []
    temp_amplitudes = []
    output_state = True  # True = heating (high output), False = low output
    ptc.send_command(f'{OUTPUT_CHANNEL}.PID.P {0}')   # No PID effect, we'll just use setpoint to toggle output
    ptc.send_command(f'{OUTPUT_CHANNEL}.PID.I {0}')
    ptc.send_command(f'{OUTPUT_CHANNEL}.PID.D {0}')
    ptc.send_command(f'{OUTPUT_CHANNEL}.PID.Input {INPUT_CHANNEL}')
    ptc.send_command(f'{OUTPUT_CHANNEL}.PID.Mode On')  # enable loop (though P=I=D=0, it will just follow setpoint)
    # Use the PTC10 in "Follow" mode if available, or trick by switching setpoint:
    # Actually, since P=I=D=0, the output will not respond at all (stuck), so using PID Mode On with zero gains won't toggle.
    # Instead, if the PTC10 has a manual output control, we should use that. Alternatively, use Follow mode:
    # In Follow mode, output = (Input - ZeroPoint)*Gain. Not useful for relay either.
    # To do a proper relay test, it's better to directly control output in a loop:
    ptc.send_command(f'{OUTPUT_CHANNEL}.PID.Mode Off')  # go to manual, which holds output at last value
    # We'll manually toggle output by setting a value on analog output or using Out.Value if available.
    # Note: On many controllers, one can set output via OutX.Value when PID is off (manual mode).
    # We assume the output is power in Watts or percentage.
    high_output_value = RELAY_HIGH_OUTPUT
    low_output_value = RELAY_LOW_OUTPUT
    ptc.send_command(f'"{OUTPUT_CHANNEL}.Value" {high_output_value}')  # set high output to start
    t_start = time.time()
    last_switch_time = t_start
    last_temp = None
    period_measured = None
    oscillating = False
    try:
        while time.time() - t_start < 3 * RELAY_PERIOD_ESTIMATE:
            # Read current temperature
            resp = ptc.send_command(f'{INPUT_CHANNEL}?')
            current_temp = None
            if resp is not None:
                try:
                    current_temp = float(resp)
                except:
                    current_temp = None
            # If we have a previous temperature reading to compare and see crossing of setpoint
            if last_temp is not None and current_temp is not None:
                # Check if temperature has crossed the setpoint (which would trigger a relay switch in on-off control)
                if output_state and current_temp >= SETPOINT:  # was heating, now exceeded setpoint
                    # Switch to low output (turn off heater)
                    ptc.send_command(f'"{OUTPUT_CHANNEL}.Value" {low_output_value}')
                    output_state = False
                    switch_time = time.time()
                    oscillation_times.append(switch_time)
                    logger.debug("Relay switch: turned OFF at t=%.1f s, T=%.2f°C", switch_time - t_start, current_temp)
                elif (not output_state) and current_temp <= SETPOINT:  # was cooling/off, now dropped below setpoint
                    # Switch to high output (turn on heater)
                    ptc.send_command(f'"{OUTPUT_CHANNEL}.Value" {high_output_value}')
                    output_state = True
                    switch_time = time.time()
                    oscillation_times.append(switch_time)
                    logger.debug("Relay switch: turned ON at t=%.1f s, T=%.2f°C", switch_time - t_start, current_temp)
            last_temp = current_temp
            time.sleep(1.0)  # 1 Hz sampling for relay test
            # If we have enough switch points to measure a period (two switches = one half-period, four switches = full cycle)
            if len(oscillation_times) >= 4:
                # Calculate period as time between two ON (or OFF) switches (i.e., every 2nd switch)
                full_cycles = []
                for k in range(2, len(oscillation_times), 2):
                    period = oscillation_times[k] - oscillation_times[k-2]
                    full_cycles.append(period)
                if full_cycles:
                    period_measured = sum(full_cycles) / len(full_cycles)
                    oscillating = True
                    break
        # After relay test
    finally:
        # Turn off output after relay test for safety
        ptc.send_command(f'"{OUTPUT_CHANNEL}.Value" 0')
        ptc.send_command(f'{OUTPUT_CHANNEL}.PID.Mode Off')
    if not oscillating or period_measured is None:
        logger.warning("Relay tuning did not induce clear oscillations.")
        return None
    # Estimate ultimate gain Ku from relay results.
    # A rough estimate: Ku ~ (4/π) * (output_amp / process_amp)
    # output_amp = (high_output_value - low_output_value) / 2
    output_amp = (high_output_value - low_output_value) / 2.0
    # process_amp: we need the resulting oscillation amplitude in temperature. We can estimate it as half the difference
    # between max and min temperature achieved during oscillation (if we had logged those).
    # For simplicity, let's assume a small oscillation around setpoint of amplitude ~ TOLERANCE or so (this is a rough guess).
    process_amp = TOLERANCE if TOLERANCE > 0 else 1.0
    Ku_est = (4.0 / math.pi) * (output_amp / process_amp)
    # Use Z-N formula with Ku_est and Tu = period_measured
    P_recommended = 0.6 * Ku_est
    I_recommended = 2 * P_recommended / period_measured
    D_recommended = 0.125 * period_measured * P_recommended
    logger.info("Relay tuning estimated Ku ≈ %.3f, Tu ≈ %.1f s.", Ku_est, period_measured)
    logger.info("Recommended initial PID gains from relay method: P = %.3f, I = %.3f, D = %.3f",
                P_recommended, I_recommended, D_recommended)
    return P_recommended, I_recommended, D_recommended

# Main tuning procedure
def run_pid_tuning():
    # Connect to PTC10
    ptc = PTC10Interface(use_labrad=USE_LABRAD)
    logger.info("Starting PID tuning procedure on %s (controlling %s).", OUTPUT_CHANNEL, INPUT_CHANNEL)
    try:
        # Initial setup: select PID input channel and set setpoint
        ptc.send_command(f'{OUTPUT_CHANNEL}.PID.Input "{INPUT_CHANNEL}"')
        ptc.send_command(f'{OUTPUT_CHANNEL}.PID.Setpoint {SETPOINT}')
        # Ensure PID mode is off before starting (manual mode)
        ptc.send_command(f'{OUTPUT_CHANNEL}.PID.Mode Off')
        # Optionally, ensure output is at 0 to start
        ptc.send_command(f'"{OUTPUT_CHANNEL}.Value" 0')
    except Exception as e:
        logger.error("Failed initial setup commands: %s", e)
        ptc.close()
        return

    # If auto-tune mode is requested, run it to get initial guess or directly use it
    initial_guess = None
    if AUTO_TUNE_MODE:
        try:
            if AUTO_TUNE_MODE.lower() == "zn":
                initial_guess = autotune_ziegler_nichols(ptc)
            elif AUTO_TUNE_MODE.lower() == "relay":
                initial_guess = autotune_relay(ptc)
            else:
                logger.warning("Unknown AUTO_TUNE_MODE '%s'; skipping auto-tuning.", AUTO_TUNE_MODE)
        except Exception as e:
            logger.error("Auto-tuning (%s) failed: %s", AUTO_TUNE_MODE, e)
        # If we got an initial guess, we can prepend it to the test list (or use it exclusively)
        if initial_guess:
            P_init, I_init, D_init = initial_guess
            # Ensure the guess is included in the sweep lists (round to avoid duplicates due to float precision)
            P_val = round(P_init, 4); I_val = round(I_init, 4); D_val = round(D_init, 4)
            if P_val not in P_VALUES or I_val not in I_VALUES or D_val not in D_VALUES:
                logger.info("Including auto-tuned initial guess (P=%.3f,I=%.3f,D=%.3f) in test sweep.", P_val, I_val, D_val)
                P_VALUES.insert(0, P_val)
                I_VALUES.insert(0, I_val)
                D_VALUES.insert(0, D_val)

    # Prepare results log file
    with open(DATA_LOG_FILE, 'w') as f:
        f.write("P,I,D,settling_time(s),steady_state_std(C)\n")
    # Loop over PID gain combinations
    for P in P_VALUES:
        for I in I_VALUES:
            for D in D_VALUES:
                # Apply PID gains
                try:
                    ptc.send_command(f'{OUTPUT_CHANNEL}.PID.P {P}')
                    ptc.send_command(f'{OUTPUT_CHANNEL}.PID.I {I}')
                    ptc.send_command(f'{OUTPUT_CHANNEL}.PID.D {D}')
                except Exception as e:
                    logger.error("Error setting PID gains P=%.3f,I=%.3f,D=%.3f: %s", P, I, D, e)
                    continue
                logger.info("Testing PID gains: P=%.3f, I=%.3f, D=%.3f", P, I, D)
                # Turn PID on to start controlling
                try:
                    ptc.send_command(f'{OUTPUT_CHANNEL}.PID.Mode On')
                except Exception as e:
                    logger.error("Failed to enable PID mode for P=%.3f,I=%.3f,D=%.3f: %s", P, I, D, e)
                    continue
                # Data collection loop
                trial_time = []
                trial_temp_ptc = []
                trial_temp_ext = []
                t_start = time.time()
                settled = False
                settling_time = None
                # Run until settled or max time reached
                while time.time() - t_start < MAX_RUN_TIME:
                    t_now = time.time()
                    # Query PTC10 temperature
                    temp_ptc = None
                    try:
                        resp = ptc.send_command(f'{INPUT_CHANNEL}?')
                        if resp is not None:
                            temp_ptc = float(resp)
                    except Exception as e:
                        logger.warning("Temperature query failed: %s", e)
                    # Query external TempStick temperature
                    temp_ext = get_tempstick_temperature() if USE_TEMPSTICK else None
                    # Log data
                    trial_time.append(t_now - t_start)
                    trial_temp_ptc.append(temp_ptc if temp_ptc is not None else float('nan'))
                    trial_temp_ext.append(temp_ext if temp_ext is not None else float('nan'))
                    # Check settling condition
                    if not settled:
                        if temp_ptc is not None:
                            if abs(temp_ptc - SETPOINT) <= TOLERANCE:
                                # Potentially settled, but ensure it stays within tolerance for required window
                                # (The analyze_performance function will double-check after collecting data)
                                # We can optimistically mark settled here, but it's safer to continue gathering data.
                                settled = True
                                settling_time = t_now - t_start
                    # End condition: if we already flagged as settled and also have collected enough post-settle data
                    if settled and (time.time() - t_start) >= settling_time + SETTLING_TIME_WINDOW:
                        # stayed within tolerance window for the required time
                        break
                    time.sleep(1.0)  # 1 second interval logging
                # Stop PID control for this trial
                try:
                    ptc.send_command(f'{OUTPUT_CHANNEL}.PID.Mode Off')
                except Exception as e:
                    logger.error("Error disabling PID after trial: %s", e)
                # Ensure output off
                try:
                    ptc.send_command(f'"{OUTPUT_CHANNEL}.Value" 0')
                except Exception:
                    pass
                # Analyze performance metrics for this trial
                st_time, ss_std = analyze_performance(trial_time, trial_temp_ptc, SETPOINT)
                logger.info("Results for P=%.3f,I=%.3f,D=%.3f: settling_time=%s, steady_state_std=%.4f",
                            P, I, D,
                            ("%.1f s" % st_time) if st_time is not None else "None",
                            ss_std if ss_std is not None else float('nan'))
                # Save summary to CSV
                with open(DATA_LOG_FILE, 'a') as f:
                    f.write(f"{P},{I},{D},{('%.3f' % st_time) if st_time is not None else ''},{('%.4f' % ss_std) if ss_std is not None else ''}\n")
                # Optionally save detailed log of this trial
                # We can save time and temperature arrays to a file for later analysis if needed.
                # For brevity, we skip writing individual trial logs here or only do so for certain trials.
                # Cooldown before next trial
                if COOLDOWN_TIME > 0:
                    logger.info("Cooling down for %.1f seconds before next trial...", COOLDOWN_TIME)
                    time.sleep(COOLDOWN_TIME)
                if COOLDOWN_TEMP is not None:
                    # Wait until temperature falls below cooldown threshold (or a timeout to avoid stalling too long)
                    cooldown_timeout = 300  # e.g., 5 minutes max for cooldown
                    t_cool_start = time.time()
                    while time.time() - t_cool_start < cooldown_timeout:
                        try:
                            resp = ptc.send_command(f'{INPUT_CHANNEL}?')
                            current_temp = float(resp) if resp is not None else None
                        except:
                            current_temp = None
                        if current_temp is not None and current_temp <= COOLDOWN_TEMP:
                            break
                        time.sleep(5)
    finally:
        # Close connection to PTC10
        ptc.close()
        logger.info("PID tuning procedure completed. Connection to PTC10 closed.")
        # If enabled, generate summary plots from the results CSV.
        if ENABLE_PLOTTING and plt is not None and np is not None:
            try:
                import csv
                P_list, I_list, D_list, settle_list, std_list = [], [], [], [], []
                with open(DATA_LOG_FILE, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            P_list.append(float(row['P']))
                            I_list.append(float(row['I']))
                            D_list.append(float(row['D']))
                            settle_list.append(float(row['settling_time(s)']) if row['settling_time(s)'] != '' else None)
                            std_list.append(float(row['steady_state_std(C)']) if row['steady_state_std(C)'] != '' else None)
                        except:
                            continue
                # Plot settling time heatmap for P vs I (for a fixed D or averaged over D)
                # Here, as an example, we plot settling time vs P for each I (taking the minimum over D values).
                # This will produce multiple lines for different I values.
                if settle_list:
                    plt.figure()
                    # Gather unique I values
                    unique_I = sorted(set(I_list))
                    for Ival in unique_I:
                        P_vals = []
                        settle_vals = []
                        for Pval, Ival2, Dval, st in zip(P_list, I_list, D_list, settle_list):
                            if Ival2 == Ival:
                                P_vals.append(Pval)
                                settle_vals.append(st if st is not None else MAX_RUN_TIME)
                        # Sort by P for nice plotting
                        sorted_pairs = sorted(zip(P_vals, settle_vals))
                        if not sorted_pairs:
                            continue
                        P_vals_sorted, settle_vals_sorted = zip(*sorted_pairs)
                        plt.plot(P_vals_sorted, settle_vals_sorted, marker='o', label=f"I={Ival}")
                    plt.xlabel("P Gain")
                    plt.ylabel("Settling Time (s)")
                    plt.title("Settling Time vs P Gain for different I values")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig("settling_time_vs_P.png")
                    # We could also show the plot or create 3D plots for (P,I,D) if needed.
                # Plot standard deviation vs D for example (averaging over P perhaps)
                if std_list:
                    plt.figure()
                    unique_D = sorted(set(D_list))
                    # Compute average std for each D across tests (for simplicity of illustration)
                    avg_std_per_D = {}
                    for Dval in unique_D:
                        vals = [std for d, std in zip(D_list, std_list) if d == Dval and std is not None]
                        if vals:
                            avg_std_per_D[Dval] = sum(vals)/len(vals)
                    D_vals = sorted(avg_std_per_D.keys())
                    std_vals = [avg_std_per_D[d] for d in D_vals]
                    plt.plot(D_vals, std_vals, marker='o')
                    plt.xlabel("D Gain")
                    plt.ylabel("Average Steady-State Std Dev (°C)")
                    plt.title("Steady-State Temperature Variability vs D Gain")
                    plt.grid(True)
                    plt.savefig("std_vs_D.png")
                logger.info("Plots saved to disk (PNG files).")
            except Exception as e:
                logger.error("Error while generating plots: %s", e)

# If run as a script, execute the tuning
if __name__ == "__main__":
    run_pid_tuning()
