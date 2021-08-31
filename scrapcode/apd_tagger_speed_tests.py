import time
import numpy

tagger_di_clock = 1
stream_apd_indices = [0, 1]
tagger_di_apd = [2, 3]
tagger_di_gate = [4, 4]


def read_raw_stream():
    """Returns dummy time tag stream"""
    global tagger_di_clock, stream_apd_indices, tagger_di_apd, tagger_di_gate
    num_active_apd_chans = len(stream_apd_indices)
    active_apd_chans = [tagger_di_apd[el] for el in stream_apd_indices]
    num_reps = int(1e4)
    # Say our sequence has the laser on constantly, then 2 us polarization
    # time before a 350 ns readout. Say the count rate is 1000 kcps
    count_rate = 1e6
    pol_count_avg = count_rate * 2e-6
    readout_count_avg = count_rate * 350e-9
    # For each rep, we'll insert a random number of counts split between the
    # APDs according to the Poissonian statistics characterized by the average
    # numbers of counts.
    sample_window = []
    gate_chan = tagger_di_gate[0]
    for ind in range(num_reps):
        gate_window = []
        pol_gate_window = []
        for chan in active_apd_chans:
            pol_counts = numpy.random.poisson(
                pol_count_avg / num_active_apd_chans
            )
            pol_gate_window.extend([chan] * pol_counts)
        numpy.random.shuffle(pol_gate_window)
        gate_window.extend(pol_gate_window)
        gate_window.append(gate_chan)
        readout_gate_window = []
        for chan in active_apd_chans:
            readout_counts = numpy.random.poisson(
                readout_count_avg / num_active_apd_chans
            )
            readout_gate_window.extend([chan] * readout_counts)
        numpy.random.shuffle(readout_gate_window)
        gate_window.extend(readout_gate_window)
        gate_window.append(-gate_chan)
        sample_window.extend(gate_window)
    sample_window.append(tagger_di_clock)
    return numpy.array(sample_window, dtype=int)


def read_counter_internal(channels):

    global tagger_di_clock, stream_apd_indices, tagger_di_apd, tagger_di_gate

    # Find clock clicks (sample breaks)
    result = numpy.nonzero(channels == tagger_di_clock)
    clock_click_inds = result[0].tolist()

    previous_sample_end_ind = None
    sample_end_ind = None

    # Counts will be a list of lists - the first dimension will divide
    # samples and the second will divide gatings within samples
    return_counts = []
    return_counts_append = return_counts.append

    # If all APDs are running off the same gate, we can make things faster
    single_gate = all(
        tagger_di_gate[el] == tagger_di_gate[0] for el in stream_apd_indices
    )

    for clock_click_ind in clock_click_inds:

        # Clock clicks end samples, so they should be included with the
        # sample itself
        sample_end_ind = clock_click_ind + 1

        sample_channels = channels[previous_sample_end_ind:sample_end_ind]

        # Make sure we've got an array for comparison to find click indices
        # and a list for operations that necessarily scale linearly and
        # need to be fast.
        sample_channels_arr = numpy.array(sample_channels)
        sample_channels_list = sample_channels.tolist()

        sample_counts = []
        sample_counts_append = sample_counts.append

        # Loop through the APDs
        for apd_index in stream_apd_indices:

            apd_channel = tagger_di_apd[apd_index]
            gate_open_channel = tagger_di_gate
            gate_close_channel = -gate_open_channel

            # Find gate open clicks
            result = numpy.nonzero(sample_channels_arr == gate_open_channel)
            gate_open_click_inds = result[0].tolist()

            # Find gate close clicks
            # Gate close channel is negative of gate open channel,
            # signifying the falling edge
            result = numpy.nonzero(sample_channels_arr == gate_close_channel)
            gate_close_click_inds = result[0].tolist()

            gate_zip = zip(gate_open_click_inds, gate_close_click_inds)
            channel_counts = [
                sample_channels_list[open_ind:close_ind].count(apd_channel)
                for open_ind, close_ind in gate_zip
            ]

            sample_counts_append(channel_counts)
        return_counts_append(sample_counts)
        previous_sample_end_ind = sample_end_ind

    return return_counts


if __name__ == "__main__":

    # Timestamps are irrelevant for a counter
    channels = read_raw_stream()
    start = time.time()
    res = read_counter_internal(channels)
    return_counts = [
        numpy.sum(sample, 0, dtype=int).tolist() for sample in res
    ]
    stop = time.time()
    print(stop - start)
    # print(return_counts)
