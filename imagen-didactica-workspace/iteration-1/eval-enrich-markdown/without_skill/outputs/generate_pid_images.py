#!/usr/bin/env python3
"""
Generate pedagogical diagrams for PID Control lecture.
Creates: 1) Block diagram of PID controller
          2) Step response comparison (P, PI, PID)
          3) Industrial control panel
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

OUTPUT_DIR = "/sessions/vibrant-compassionate-dijkstra/mnt/TC3002B-2026/imagen-didactica-workspace/iteration-1/eval-enrich-markdown/without_skill/outputs/"

def create_pid_block_diagram():
    """
    Create a block diagram of PID controller showing:
    - Signal flow: r(t) -> error e(t) -> P/I/D -> sum -> G(s) -> y(t)
    - Feedback loop
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(7, 5.5, 'Diagrama de Bloques del Sistema de Control PID',
            fontsize=16, fontweight='bold', ha='center')

    # Input reference r(t)
    ax.arrow(0.5, 3, 0.8, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
    ax.text(0.3, 3.3, 'r(t)', fontsize=11, fontweight='bold')

    # Sum junction (circle)
    circle_sum1 = Circle((1.8, 3), 0.2, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(circle_sum1)
    ax.text(1.8, 3, '+', fontsize=14, ha='center', va='center', fontweight='bold')

    # Error signal
    ax.arrow(1.3, 3, 0.4, 0, head_width=0.1, head_length=0.08, fc='black', ec='black')
    ax.arrow(2, 3, 0.5, 0, head_width=0.1, head_length=0.08, fc='black', ec='black')
    ax.text(2.2, 3.3, 'e(t)', fontsize=11, fontweight='bold')

    # Three control blocks: P, I, D
    block_y = [4.2, 3, 1.8]
    block_labels = ['Kp', 'Ki', 'Kd']
    block_colors = ['#FFB6C6', '#B6D7FF', '#D7FFB6']

    for i, (y, label, color) in enumerate(zip(block_y, block_labels, block_colors)):
        # Block
        rect = FancyBboxPatch((2.8, y-0.3), 0.8, 0.6,
                             boxstyle="round,pad=0.05",
                             edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(3.2, y, label, fontsize=11, fontweight='bold', ha='center', va='center')

        # Input arrow
        ax.arrow(2.55, y, 0.15, 0, head_width=0.08, head_length=0.05, fc='black', ec='black')

        # Output arrow to sum
        ax.arrow(3.7, y, 0.8, 0, head_width=0.08, head_length=0.05, fc='black', ec='black')

    # Sum all control actions
    circle_sum2 = Circle((4.8, 3), 0.25, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(circle_sum2)
    ax.text(4.8, 3, '+', fontsize=14, ha='center', va='center', fontweight='bold')

    # Signals entering sum from P, I, D
    for y in block_y:
        if y != 3:
            ax.plot([3.7, 4.55], [y, 3], 'k-', linewidth=1.5)

    # Plant transfer function G(s)
    rect_plant = FancyBboxPatch((6, 2.7), 1.2, 0.6,
                               boxstyle="round,pad=0.05",
                               edgecolor='black', facecolor='#FFE6B6', linewidth=2)
    ax.add_patch(rect_plant)
    ax.text(6.6, 3, 'G(s)', fontsize=12, fontweight='bold', ha='center', va='center')

    # Arrow from sum to plant
    ax.arrow(5.1, 3, 0.75, 0, head_width=0.1, head_length=0.08, fc='black', ec='black')

    # Output y(t)
    ax.arrow(7.3, 3, 1, 0, head_width=0.1, head_length=0.08, fc='black', ec='black')
    ax.text(8.5, 3.3, 'y(t)', fontsize=11, fontweight='bold')

    # Feedback path
    ax.plot([8.3, 8.3, 1.8, 1.8], [3, 0.5, 0.5, 2.8], 'k-', linewidth=1.5)

    # Feedback sum junction
    circle_fback = Circle((1.8, 2.8), 0.15, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(circle_fback)
    ax.text(1.55, 2.55, '−', fontsize=12, fontweight='bold')

    # Add legend for control actions
    legend_text = "P: Acción Proporcional    I: Acción Integral    D: Acción Derivativa"
    ax.text(7, 0.8, legend_text, fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}01_pid_block_diagram.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated: 01_pid_block_diagram.png")

def create_step_response_comparison():
    """
    Create a graph comparing step responses:
    - P only (oscillatory with offset)
    - PI (no offset but with overshoot)
    - PID (optimal with minimal overshoot)
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Time vector
    t = np.linspace(0, 5, 500)

    # Reference (unit step)
    reference = np.ones_like(t)

    # P control (proportional only) - oscillatory with offset
    # Underdamped response with steady-state error
    wn_p = 1.2
    zeta_p = 0.3
    y_p = 1 - np.exp(-zeta_p*wn_p*t) * (np.cos(np.sqrt(1-zeta_p**2)*wn_p*t) +
                                         (zeta_p/np.sqrt(1-zeta_p**2))*np.sin(np.sqrt(1-zeta_p**2)*wn_p*t))
    y_p = y_p * 0.85  # Steady-state error for P-only

    # PI control - critically damped/slightly underdamped
    wn_pi = 1.8
    zeta_pi = 0.8
    y_pi = 1 - np.exp(-zeta_pi*wn_pi*t) * (np.cos(np.sqrt(1-zeta_pi**2)*wn_pi*t) +
                                            (zeta_pi/np.sqrt(1-zeta_pi**2))*np.sin(np.sqrt(1-zeta_pi**2)*wn_pi*t))
    # Add slight overshoot
    y_pi = np.minimum(y_pi, 1.3)

    # PID control - well-damped response
    wn_pid = 2.0
    zeta_pid = 0.95
    y_pid = 1 - np.exp(-zeta_pid*wn_pid*t) * (np.cos(np.sqrt(1-zeta_pid**2)*wn_pid*t) +
                                               (zeta_pid/np.sqrt(1-zeta_pid**2))*np.sin(np.sqrt(1-zeta_pid**2)*wn_pid*t))
    y_pid = np.minimum(y_pid, 1.12)

    # Plot
    ax.plot(t, reference, 'k--', linewidth=2.5, label='Referencia r(t)', alpha=0.7)
    ax.plot(t, y_p, color='#FF6B6B', linewidth=2.5, label='Solo P (error de estado estacionario)', marker='o',
            markevery=25, markersize=6)
    ax.plot(t, y_pi, color='#4ECDC4', linewidth=2.5, label='PI (sin error, con sobrepaso)', marker='s',
            markevery=25, markersize=6)
    ax.plot(t, y_pid, color='#95E77D', linewidth=2.5, label='PID (respuesta óptima)', marker='^',
            markevery=25, markersize=6)

    # Grid and labels
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('Tiempo (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Amplitud', fontsize=12, fontweight='bold')
    ax.set_title('Respuesta al Escalón: Comparación P, PI y PID', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='right', framealpha=0.95)
    ax.set_xlim(0, 5)
    ax.set_ylim(0.6, 1.4)

    # Add annotations
    ax.annotate('Offset\nestacionario', xy=(4.5, y_p[-1]), xytext=(3.5, 0.75),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    ax.annotate('Sobrepaso\nPI', xy=(1.5, np.max(y_pi)), xytext=(1.8, 1.35),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='#4ECDC4', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))

    ax.annotate('Respuesta\nóptima', xy=(2.5, y_pid[250]), xytext=(3, 1.25),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='#95E77D', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}02_step_response_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated: 02_step_response_comparison.png")

def create_industrial_control_panel():
    """
    Create a realistic industrial control panel visualization showing HMI displays
    with PID control loops in real-time for a chemical processing plant.
    """
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Panel background
    panel = Rectangle((0.2, 0.2), 9.6, 9.6, linewidth=3, edgecolor='#333333',
                     facecolor='#E8E8E8')
    ax.add_patch(panel)

    # Header
    ax.text(5, 9.3, 'Panel de Control - Planta de Procesamiento Químico',
            fontsize=15, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='#1F4788',
                     edgecolor='black', linewidth=2, alpha=0.9),
            color='white')

    # Three HMI display sections
    displays = [
        {'title': 'Loop 1: Control\nde Temperatura', 'x': 1, 'y': 6.5, 'setpoint': 85, 'actual': 84.2, 'color': '#FF6B6B'},
        {'title': 'Loop 2: Control\nde Presión', 'x': 3.7, 'y': 6.5, 'setpoint': 5.0, 'actual': 4.98, 'color': '#4ECDC4'},
        {'title': 'Loop 3: Control\nde Flujo', 'x': 6.4, 'y': 6.5, 'setpoint': 100, 'actual': 99.5, 'color': '#FFE66D'},
    ]

    for i, display in enumerate(displays):
        # Display frame
        frame = FancyBboxPatch((display['x'], display['y']-2), 2.2, 2.2,
                              boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='#F0F0F0',
                              linewidth=2)
        ax.add_patch(frame)

        # Title
        ax.text(display['x'] + 1.1, display['y'] - 0.1, display['title'],
                fontsize=10, fontweight='bold', ha='center', va='top')

        # Setpoint display
        ax.text(display['x'] + 0.3, display['y'] - 0.9, 'Setpoint:', fontsize=9, fontweight='bold')
        ax.text(display['x'] + 1.8, display['y'] - 0.9, f"{display['setpoint']}",
                fontsize=10, ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

        # Actual value display
        ax.text(display['x'] + 0.3, display['y'] - 1.4, 'Actual:', fontsize=9, fontweight='bold')
        ax.text(display['x'] + 1.8, display['y'] - 1.4, f"{display['actual']}",
                fontsize=10, ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFFCC', edgecolor='gray'))

        # Status bar (simulating control action)
        status_width = (display['actual'] / display['setpoint']) * 1.6
        status_bar = Rectangle((display['x'] + 0.3, display['y'] - 1.9), status_width, 0.3,
                              facecolor=display['color'], edgecolor='black', linewidth=1)
        ax.add_patch(status_bar)
        ax.text(display['x'] + 1.1, display['y'] - 2.05, 'Control', fontsize=8, ha='center', va='center')

    # Lower section - System status
    status_frame = FancyBboxPatch((0.5, 2.8), 9, 3,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='black', facecolor='#F5F5F5',
                                 linewidth=2)
    ax.add_patch(status_frame)

    ax.text(5, 5.5, 'Estado del Sistema en Tiempo Real', fontsize=12, fontweight='bold', ha='center')

    # System information
    info_text = [
        '● Planta activa | Velocidad de muestreo: 100 ms | Comunicación: ModBus',
        '● Sintonización PID | Modo: Automático | Alarmas: 0',
        '● Última actualización: 2024-03-14 14:32:15 UTC',
        '● Temperatura Proceso: 84.2°C | Presión: 4.98 bar | Flujo: 99.5 L/min'
    ]

    y_pos = 5.0
    for line in info_text:
        ax.text(5, y_pos, line, fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='lightgray', alpha=0.8))
        y_pos -= 0.6

    # Indicators at bottom
    indicators = [
        {'label': 'Alimentación', 'status': 'OK', 'color': '#95E77D', 'x': 1.5},
        {'label': 'Red', 'status': 'OK', 'color': '#95E77D', 'x': 4},
        {'label': 'Sensores', 'status': 'OK', 'color': '#95E77D', 'x': 6.5},
        {'label': 'Actuadores', 'status': 'OK', 'color': '#95E77D', 'x': 9},
    ]

    ax.text(5, 2.2, 'Indicadores de Estado', fontsize=10, fontweight='bold', ha='center')

    for ind in indicators:
        circle = Circle((ind['x'], 1.5), 0.25, facecolor=ind['color'],
                       edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(ind['x'], 0.9, ind['label'], fontsize=8, ha='center', fontweight='bold')
        ax.text(ind['x'], 0.5, ind['status'], fontsize=7, ha='center', style='italic',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.6))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}03_industrial_control_panel.png', dpi=300, bbox_inches='tight',
                facecolor='#CCCCCC', edgecolor='none')
    plt.close()
    print("✓ Generated: 03_industrial_control_panel.png")

if __name__ == '__main__':
    print("Generating PID control pedagogical images...")
    create_pid_block_diagram()
    create_step_response_comparison()
    create_industrial_control_panel()
    print("\nAll images generated successfully!")
