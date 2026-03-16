#!/usr/bin/env python3
"""
Create an educational Bode diagram for a second-order system
System parameters:
- Natural frequency (wn): 10 rad/s
- Damping factor (zeta): 0.5
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import json
from datetime import datetime

# System parameters
wn = 10  # Natural frequency in rad/s
zeta = 0.5  # Damping factor

# Create transfer function for second-order system: wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
numerator = [wn**2]
denominator = [1, 2*zeta*wn, wn**2]
system = signal.TransferFunction(numerator, denominator)

# Frequency range for Bode plot (logarithmic scale)
w = np.logspace(0, 3, 1000)  # From 1 to 1000 rad/s

# Calculate frequency response
w_out, mag, phase = signal.bode(system, w)

# Convert magnitude to dB (already done by signal.bode)
# Convert phase to degrees (already done by signal.bode)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Configure the figure style
fig.suptitle('Diagrama de Bode - Sistema de Segundo Orden',
             fontsize=16, fontweight='bold', y=0.995)

# Plot magnitude response
ax1.semilogx(w_out, mag, 'b-', linewidth=2.5, label='Magnitud')
ax1.grid(True, which='both', linestyle='--', alpha=0.7)
ax1.set_ylabel('Magnitud (dB)', fontsize=12, fontweight='bold')
ax1.set_title(f'Diagrama de Magnitud\n(ωn = {wn} rad/s, ζ = {zeta})',
              fontsize=12, pad=10)
ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax1.axvline(x=wn, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label=f'ωn = {wn} rad/s')

# Add -3dB line
ax1.axhline(y=-3, color='g', linestyle=':', linewidth=1.5, alpha=0.7, label='-3 dB')
ax1.legend(loc='upper right', fontsize=10)
ax1.set_xlim([w_out[0], w_out[-1]])

# Plot phase response
ax2.semilogx(w_out, phase, 'r-', linewidth=2.5, label='Fase')
ax2.grid(True, which='both', linestyle='--', alpha=0.7)
ax2.set_xlabel('Frecuencia Angular (rad/s)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Fase (grados)', fontsize=12, fontweight='bold')
ax2.set_title('Diagrama de Fase', fontsize=12, pad=10)
ax2.axvline(x=wn, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label=f'ωn = {wn} rad/s')
ax2.axhline(y=-90, color='g', linestyle=':', linewidth=1.5, alpha=0.7, label='-90°')
ax2.legend(loc='lower left', fontsize=10)
ax2.set_xlim([w_out[0], w_out[-1]])

# Adjust layout
plt.tight_layout()

# Save the figure
output_path = '/sessions/vibrant-compassionate-dijkstra/mnt/TC3002B-2026/imagen-didactica-workspace/iteration-1/eval-bode-diagram/without_skill/outputs/bode_diagram.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Bode diagram saved to: {output_path}")

# Save metrics
metrics = {
    "timestamp": datetime.now().isoformat(),
    "task_description": "Generate educational Bode diagram for second-order system",
    "system_parameters": {
        "natural_frequency_rad_s": wn,
        "damping_factor": zeta,
        "transfer_function": "H(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2)"
    },
    "output_file": output_path,
    "output_dpi": 300,
    "figure_size": [12, 10],
    "frequency_range": {
        "min_rad_s": float(w_out[0]),
        "max_rad_s": float(w_out[-1]),
        "points": len(w_out)
    }
}

metrics_path = '/sessions/vibrant-compassionate-dijkstra/mnt/TC3002B-2026/imagen-didactica-workspace/iteration-1/eval-bode-diagram/without_skill/outputs/diagram_info.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Diagram information saved to: {metrics_path}")

# Also create a detailed analysis
analysis = {
    "dc_gain_db": float(mag[0]),
    "magnitude_at_wn": float(np.interp(wn, w_out, mag)),
    "phase_at_wn": float(np.interp(wn, w_out, phase)),
    "resonance_frequency": float(wn * np.sqrt(1 - 2*zeta**2)) if zeta < 1/np.sqrt(2) else wn,
    "quality_factor_q": 1 / (2 * zeta)
}

analysis_path = '/sessions/vibrant-compassionate-dijkstra/mnt/TC3002B-2026/imagen-didactica-workspace/iteration-1/eval-bode-diagram/without_skill/outputs/system_analysis.json'
with open(analysis_path, 'w') as f:
    json.dump(analysis, f, indent=2)
print(f"System analysis saved to: {analysis_path}")

print("\nSystem characteristics:")
print(f"  DC Gain: {analysis['dc_gain_db']:.2f} dB")
print(f"  Magnitude at ωn: {analysis['magnitude_at_wn']:.2f} dB")
print(f"  Phase at ωn: {analysis['phase_at_wn']:.2f}°")
print(f"  Quality Factor (Q): {analysis['quality_factor_q']:.2f}")

plt.close()
