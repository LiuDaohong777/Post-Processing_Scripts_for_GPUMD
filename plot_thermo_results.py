import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import argparse

# ========== Parameters ==========
parser = argparse.ArgumentParser(description='Plot GPUMD thermo.out with integral average')
parser.add_argument('-dump', type=int, required=True, help='thermo output interval (steps between output)')
parser.add_argument('-dt', type=float, required=True, help='time step (e.g. 0.001 for ps)')
args = parser.parse_args()

# ========== Time setting ==========
dump = args.dump
dt = args.dt
thermo_interval = dump * dt

# ========== Read file ==========
data = np.loadtxt('thermo.out')
time = np.arange(len(data)) * thermo_interval * 0.001    # ps
t = np.arange(1, len(data) + 1) * thermo_interval * 0.001

# ========== Get thermal data ==========
T = data[:, 0]
K = data[:, 1]
U = data[:, 2]
E = K + U
P = (data[:, 3] + data[:, 4] + data[:, 5]) / 3
a_len = np.linalg.norm(data[:, 9:12], axis=1)
b_len = np.linalg.norm(data[:, 12:15], axis=1)
c_len = np.linalg.norm(data[:, 15:18], axis=1)

# ========== Define function ==========
def integral_average(y, t):
    integral = cumulative_trapezoid(y, t, initial=0)
    avg = np.zeros_like(integral)
    avg[1:] = integral[1:] / t[1:]
    avg[0] = y[0]
    return avg

# ========== Compute integral average ==========
T_avg = integral_average(T, time)
E_avg = integral_average(E, time)
P_avg = integral_average(P, time)
a_avg = integral_average(a_len, time)
b_avg = integral_average(b_len, time)
c_avg = integral_average(c_len, time)

# ========== Save average data ==========
output_file = 'thermo_averaged_data.dat'
output_data = np.column_stack((t, T_avg, E_avg, P_avg, a_avg, b_avg, c_avg))
header = "time T_avg E_avg P_avg a_avg b_avg c_avg"
np.savetxt(output_file, output_data, header=header, fmt='%.6f')

# ========== Plot ==========
plt.figure(figsize=(14, 10))

# Temperature
plt.subplot(2, 2, 1)
plt.plot(t, T, label='T', color='orange', alpha=0.4)
plt.plot(t, T_avg, label='⟨T⟩', color='orange', linestyle='--')
#plt.annotate(f'⟨T⟩ = {T_avg[-1]:.2f}', xy=(time[-1], T_avg[-1]), xytext=(time[-1]*0.85, T_avg[-1]*1.05), fontsize=10, color='orange')
plt.xlabel('Time (ps)')
plt.ylabel('Temperature (K)')
plt.title('Temperature vs Time')
plt.legend()
plt.grid(True)

# Energy
plt.subplot(2, 2, 2)
plt.plot(t, E, label='E', color='green', alpha=0.4)
plt.plot(t, E_avg, label='⟨E⟩', color='green', linestyle='--')
#plt.annotate(f'⟨E⟩ = {E_avg[-1]:.2f}', xy=(time[-1], E_avg[-1]), xytext=(time[-1]*0.7, E_avg[-1]*1.03), fontsize=10, color='green')
plt.xlabel('Time (ps)')
plt.ylabel('Total Energy')
plt.title('Total Energy vs Time')
plt.legend()
plt.grid(True)

# Pressure
plt.subplot(2, 2, 3)
plt.plot(t, P, label='P', color='blue', alpha=0.4)
plt.plot(t, P_avg, label='⟨P⟩', color='blue', linestyle='--')
#plt.annotate(f'⟨P⟩ = {P_avg[-1]:.2f}', xy=(time[-1], P_avg[-1]), xytext=(time[-1]*0.7, P_avg[-1]*1.03), fontsize=10, color='blue')
plt.xlabel('Time (ps)')
plt.ylabel('Pressure')
plt.title('Pressure vs Time')
plt.legend()
plt.grid(True)

# Lattice constants
plt.subplot(2, 2, 4)
plt.plot(t, a_len, label='|a|', color='red', alpha=0.3)
plt.plot(t, b_len, label='|b|', color='purple', alpha=0.3)
plt.plot(t, c_len, label='|c|', color='brown', alpha=0.3)
plt.plot(time, a_avg, label='⟨|a|⟩', color='red', linestyle='--')
plt.plot(time, b_avg, label='⟨|b|⟩', color='purple', linestyle='--')
plt.plot(time, c_avg, label='⟨|c|⟩', color='brown', linestyle='--')
#plt.annotate(f'⟨|a|⟩ = {a_avg[-1]:.3f}', xy=(time[-1], a_avg[-1]), xytext=(time[-1]*0.7, a_avg[-1]*1.01), fontsize=9, color='red')
#plt.annotate(f'⟨|b|⟩ = {b_avg[-1]:.3f}', xy=(time[-1], b_avg[-1]), xytext=(time[-1]*0.7, b_avg[-1]*1.01), fontsize=9, color='purple')
#plt.annotate(f'⟨|c|⟩ = {c_avg[-1]:.3f}', xy=(time[-1], c_avg[-1]), xytext=(time[-1]*0.7, c_avg[-1]*1.01), fontsize=9, color='brown')
plt.xlabel('Time (ps)')
plt.ylabel('Lattice Constant')
plt.title('Lattice Constants vs Time')
plt.legend()
plt.grid(True)

plt.savefig('thermo.png', dpi=300, bbox_inches='tight')
