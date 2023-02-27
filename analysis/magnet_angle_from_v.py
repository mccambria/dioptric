import numpy as np
from scipy.optimize import fsolve

# https://www.nature.com/articles/nature07278 methods


# def solve_B_theta(D, v1, v2, z):
#    theta = z[1]
#    B = z[0]

#    F = np.empty((2))
   
#    num=7*D**3 + 2*(v1+v2)*(2*(v1**2+v2**2)-5*v1*v2)-3*D*(v1**2+v2**2-v1*v2)
#    den = 9*(v1**2+v2**2-v1*v2-D**2)
#    F[1] = num/den - D*np.cos(2*theta)
#    F[0] = (v1**2+v2**2-v1*v2-D**2)/3 - B**2
#    return F

def solve_B_theta(D, E, v1, v2, z):
   theta = z[1]
   B = z[0]

   F = np.empty((2))
   
   num=7*D**3 + 2*(v1+v2)*(2*(v1**2+v2**2)-5*v1*v2 - 9*E**2)-3*D*(v1**2+v2**2-v1*v2+9*E**2)
   den = 9*(v1**2+v2**2-v1*v2-D**2-3*E**2)
   F[1] = num/den - ( D*np.cos(2*theta) )#+ 2*E*np.sin(theta)**2 )
   F[0] = (v1**2+v2**2-v1*v2-D**2)/3 - E**2 - B**2
   return F

# %% NV0
v1=2.86608
v2=2.87750

D =2.87039 #(3)
E = 0
solve_B_theta_func = lambda z: solve_B_theta(D,E, v1, v2, z)
zGuess = np.array([0.06,( 20 )*np.pi/180])
# zGuess = np.array([0.06,1.9])
z = fsolve(solve_B_theta_func,zGuess)

print('NV0')
print('{:.1f} G'.format(z[0]/0.0028))
print('{:.2f} deg'.format(z[1]*180/np.pi))

# %% NV1
v1= 2.83967
v2= 2.90306

D=2.8704
E = 0.00
solve_B_theta_func = lambda z: solve_B_theta(D,E, v1, v2, z)
zGuess = np.array([0.06,1.9])
z = fsolve(solve_B_theta_func,zGuess)

print('NV1')
print('{:.1f} G'.format(z[0]/0.0028))
print('{:.2f} deg'.format(z[1]*180/np.pi))

# %% NV2
v1=2.81980
v2=2.92137

D=2.8704
E = 0
solve_B_theta_func = lambda z: solve_B_theta(D, E,v1, v2, z)
# zGuess = np.array([0.06,1.9])
zGuess = np.array([0.06,( 5 )*np.pi/180])
z = fsolve(solve_B_theta_func,zGuess)

print('NV2')
print('{:.1f} G'.format(z[0]/0.0028))
print('{:.2f} deg'.format(z[1]*180/np.pi))