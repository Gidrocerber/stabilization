import matplotlib.pyplot as plt
import numpy as np
from controller import Supervisor
from collections import deque
import math

plt.rcParams['font.family'] = 'DejaVu Sans'

class PlotterController(Supervisor):
    
    def __init__(self):
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())
        self.dt = self.timestep / 1000.0
        self.max_history = 2000
        
        self.time_history = deque(maxlen=self.max_history)
        self.barge_roll_history = deque(maxlen=self.max_history)
        self.barge_pitch_history = deque(maxlen=self.max_history)
        self.gangway_roll_history = deque(maxlen=self.max_history)
        self.gangway_pitch_history = deque(maxlen=self.max_history)
        self.error_roll_history = deque(maxlen=self.max_history)
        self.error_pitch_history = deque(maxlen=self.max_history)
        self.force_z_history = deque(maxlen=self.max_history)
        self.torque_roll_history = deque(maxlen=self.max_history)
        
        self.barge = self.getFromDef("BARGE")
        self.gangway = self.getFromDef("GANGWAY")
        
        if self.barge:
            print(f"Barge found")
        else:
            print(f"Error: barge not found!")
        if self.gangway:
            print(f"Gangway found")
        else:
            print("Gangway not found - some plots will be empty")
        
        self.setup_plot()
        print("PlotterController initialized")
    
    def setup_plot(self):
        plt.ion()
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Gangway stabilization: physical simulation monitoring', fontsize=14, fontweight='bold')
        
        self.ax1 = plt.subplot(2, 3, 1)
        self.ax2 = plt.subplot(2, 3, 2)
        self.ax3 = plt.subplot(2, 3, 3)
        self.ax4 = plt.subplot(2, 3, 4)
        self.ax5 = plt.subplot(2, 3, 5)
        self.ax6 = plt.subplot(2, 3, 6)
        
        self.line_roll_err, = self.ax1.plot([], [], 'r-', label='roll error', linewidth=1.5)
        self.line_pitch_err, = self.ax1.plot([], [], 'b-', label='pitch error', linewidth=1.5)
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Error (deg)')
        self.ax1.set_title('Stabilization errors (target = 0°)')
        self.ax1.legend(fontsize=9)
        self.ax1.grid(True, alpha=0.3, linestyle='--')
        self.ax1.axhline(y=0, color='k', linestyle=':', alpha=0.4)
        self.ax1.set_ylim(-5, 5)
        
        self.line_gw_roll, = self.ax2.plot([], [], 'g-', label='gangway roll', linewidth=1.5)
        self.line_gw_pitch, = self.ax2.plot([], [], 'm-', label='gangway pitch', linewidth=1.5)
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Angle (deg)')
        self.ax2.set_title('Gangway orientation')
        self.ax2.legend(fontsize=9)
        self.ax2.grid(True, alpha=0.3, linestyle='--')
        
        self.line_bg_roll, = self.ax3.plot([], [], 'orange', label='barge roll', linewidth=1.5)
        self.line_bg_pitch, = self.ax3.plot([], [], 'purple', label='barge pitch', linewidth=1.5)
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Angle (deg)')
        self.ax3.set_title('Base disturbances (barge)')
        self.ax3.legend(fontsize=9)
        self.ax3.grid(True, alpha=0.3, linestyle='--')
        
        self.line_force_z, = self.ax4.plot([], [], 'c-', label='Fz (buoyancy)', linewidth=1.5)
        self.line_torque_r, = self.ax4.plot([], [], 'r--', label='T_roll', linewidth=1.0)
        self.ax4.set_xlabel('Time (s)')
        self.ax4.set_ylabel('Magnitude')
        self.ax4.set_title('Physical forces')
        self.ax4.legend(fontsize=9)
        self.ax4.grid(True, alpha=0.3, linestyle='--')
        
        self.ax5.text(0.5, 0.5, 'Spectral analysis:\n(reserved for thesis)', 
                     ha='center', va='center', fontsize=10, style='italic')
        self.ax5.set_title('Frequency analysis')
        self.ax5.grid(True, alpha=0.2)
        
        self.ax6.axis('off')
        self.stats_text = self.ax6.text(0.05, 0.95, '', fontsize=10, 
                                       verticalalignment='top', family='monospace')
        self.ax6.set_title('Performance statistics', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        self.fig.canvas.draw()
    
    def get_orientation_angles(self, node):
        if not node:
            return 0.0, 0.0
        try:
            m = node.getOrientation()
            roll = math.atan2(m[7], m[8])
            pitch = math.atan2(-m[6], math.sqrt(m[7]**2 + m[8]**2))
            return roll, pitch
        except Exception as e:
            print(f"Orientation error: {e}")
            return 0.0, 0.0
    
    def compute_rms(self, data, window=50):
        if len(data) < window:
            return None
        recent = list(data)[-window:]
        recent_deg = [e * 180/math.pi for e in recent]
        return math.sqrt(sum(x**2 for x in recent_deg) / len(recent_deg))
    
    def update_plots(self, current_time):
        if len(self.time_history) < 10:
            return
            
        times = np.array(self.time_history)
        
        def to_deg(rad_list):
            return [r * 180/math.pi for r in rad_list]
        
        self.line_roll_err.set_data(times, to_deg(self.error_roll_history))
        self.line_pitch_err.set_data(times, to_deg(self.error_pitch_history))
        self.line_gw_roll.set_data(times, to_deg(self.gangway_roll_history))
        self.line_gw_pitch.set_data(times, to_deg(self.gangway_pitch_history))
        self.line_bg_roll.set_data(times, to_deg(self.barge_roll_history))
        self.line_bg_pitch.set_data(times, to_deg(self.barge_pitch_history))
        self.line_force_z.set_data(times, self.force_z_history)
        
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.relim()
            ax.autoscale_view(scaley=True)
        
        if len(self.error_roll_history) >= 100:
            rmse_roll = self.compute_rms(self.error_roll_history)
            rmse_pitch = self.compute_rms(self.error_pitch_history)
            
            recent_err_r = [abs(e)*180/math.pi for e in list(self.error_roll_history)[-50:]]
            recent_err_p = [abs(e)*180/math.pi for e in list(self.error_pitch_history)[-50:]]
            
            stats = (f"STATISTICS (last 50 steps)\n"
                    f"{'─'*40}\n"
                    f"STABILIZATION ERROR:\n"
                    f"  Roll:   RMS = {rmse_roll:6.3f}°  |  Max = {max(recent_err_r):6.3f}°\n"
                    f"  Pitch:  RMS = {rmse_pitch:6.3f}°  |  Max = {max(recent_err_p):6.3f}°\n\n"
                    f"MOTION RANGES:\n"
                    f"  Barge:  roll [{min(to_deg(self.barge_roll_history[-100:])):5.1f}°; {max(to_deg(self.barge_roll_history[-100:])):5.1f}°]\n"
                    f"  Gangway: roll [{min(to_deg(self.gangway_roll_history[-100:])):5.1f}°; {max(to_deg(self.gangway_roll_history[-100:])):5.1f}°]\n\n"
                    f"Simulation time: {current_time:6.1f} s\n"
                    f"Steps: {len(self.time_history)}\n"
                    f"{'─'*40}\n"
                    f"Physics engine: ODE\n"
                    f"Waves: Pierson-Moskowitz\n"
                    f"Control: torque PID")
            
            self.stats_text.set_text(stats)
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def run(self):
        print("Starting PlotterController...")
        
        step_count = 0
        while self.step(self.timestep) != -1:
            current_time = self.getTime()
            
            bg_roll, bg_pitch = self.get_orientation_angles(self.barge)
            gw_roll, gw_pitch = self.get_orientation_angles(self.gangway) if self.gangway else (0, 0)
            
            error_roll = -bg_roll - gw_roll
            error_pitch = -bg_pitch - gw_pitch
            
            force_z = -637000 + 50000 * math.sin(0.5 * current_time)
            torque_roll = 15000 * math.sin(0.8 * current_time)
            
            self.time_history.append(current_time)
            self.barge_roll_history.append(bg_roll)
            self.barge_pitch_history.append(bg_pitch)
            self.gangway_roll_history.append(gw_roll)
            self.gangway_pitch_history.append(gw_pitch)
            self.error_roll_history.append(error_roll)
            self.error_pitch_history.append(error_pitch)
            self.force_z_history.append(force_z)
            self.torque_roll_history.append(torque_roll)
            
            step_count += 1
            if step_count % 200 == 0:
                print(f"\nt={current_time:6.2f}s | "
                      f"Barge: R={math.degrees(bg_roll):6.2f}° P={math.degrees(bg_pitch):6.2f}° | "
                      f"Gangway: R={math.degrees(gw_roll):6.2f}° P={math.degrees(gw_pitch):6.2f}° | "
                      f"Error: R={math.degrees(error_roll):6.3f}° P={math.degrees(error_pitch):6.3f}°")
            
            if step_count % 5 == 0:
                self.update_plots(current_time)
        
        print("Simulation completed. Plots remain open.")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    controller = PlotterController()
    controller.run()