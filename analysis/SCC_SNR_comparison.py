import numpy

# SNR_C = 0.025
# SNR_SCC = 0.1
# t_SCC = 50000 #us
# t_C = 0.3 #us


# SNR_C = 0.025
# SNR_SCC = 0.5
# t_SCC = 50000 #us
# t_C = 0.3 #us

SNR_C = 0.02
SNR_SCC = 0.1
t_SCC = 5000 #us
t_C = 0.3 #us


T=(SNR_C**2 * t_SCC - SNR_SCC * t_C ) / (SNR_SCC**2 - SNR_C**2)

print('{} us or longer'.format(T))