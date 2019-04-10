import keysight.command_expert_py3 as kt #import keysight library

test = "C:\\Users\\kolkowitz\\Desktop\\Waveforms\\" #define local directory where commend sequences are saved
kt.run_sequence(test + "SquaresAndTriangles.iseqx") #excuate sequences in Keysight Commend Expert
