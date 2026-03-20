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
        self.max_history = 1000
        
        self.time_history = deque(maxlen=self.max_history)
        self.barge_roll_history = deque(maxlen=self.max_history)
        self.barge_pitch_history = deque(maxlen=self.max_history)
        self.gangway_roll_history = deque(maxlen=self.max_history)
        self.gangway_pitch_history = deque(maxlen=self.max_history)
        self.error_roll_history = deque(maxlen=self.max_history)
        self.error_pitch_history = deque(maxlen=self.max_history)
        
        self.barge = self.getFromDef("BARGE")
        self.gangway = None
        
        if self.barge:
            print(f"баржа найдена: {self.barge}")
            self.gangway = self.getFromDef("GANGWAY")
            if self.gangway:
                print(f"трап найден: {self.gangway}")
            else:
                print("внимание: трап не найден!")
        else:
            print("ошибка: баржа не найдена!")
        
        self.setup_plot()
        print("контроллер плоттера инициализирован")
    
    def setup_plot(self):
        plt.ion()
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(14, 8))
        
        self.line_roll_error, = self.ax1.plot([], [], 'r-', label='ошибка крена', linewidth=1.5)
        self.line_pitch_error, = self.ax1.plot([], [], 'b-', label='ошибка тангажа', linewidth=1.5)
        self.ax1.set_xlabel('время (с)')
        self.ax1.set_ylabel('ошибка (град)')
        self.ax1.set_title('ошибки стабилизации (цель = 0)')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        self.ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        self.line_gangway_roll, = self.ax2.plot([], [], 'g-', label='крен трапа', linewidth=1.5)
        self.line_gangway_pitch, = self.ax2.plot([], [], 'm-', label='тангаж трапа', linewidth=1.5)
        self.ax2.set_xlabel('время (с)')
        self.ax2.set_ylabel('угол (град)')
        self.ax2.set_title('ориентация трапа')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        
        self.line_barge_roll, = self.ax3.plot([], [], 'orange', label='крен баржи', linewidth=1.5)
        self.line_barge_pitch, = self.ax3.plot([], [], 'purple', label='тангаж баржи', linewidth=1.5)
        self.ax3.set_xlabel('время (с)')
        self.ax3.set_ylabel('угол (град)')
        self.ax3.set_title('возмущения основания (баржа)')
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)
        
        self.ax4.axis('off')
        self.stats_text = self.ax4.text(0.1, 0.5, '', fontsize=11, verticalalignment='center')
        self.ax4.set_title('статистика работы')
        
        plt.tight_layout()
    
    def get_barge_orientation(self):
        if not self.barge:
            return 0, 0
        try:
            m = self.barge.getOrientation()
            roll = math.atan2(m[7], m[8])
            pitch = math.atan2(-m[6], math.sqrt(m[7]**2 + m[8]**2))
            return roll, pitch
        except:
            return 0, 0
    
    def get_gangway_orientation(self):
        if not self.gangway:
            return 0, 0
        try:
            m = self.gangway.getOrientation()
            roll = math.atan2(m[7], m[8])
            pitch = math.atan2(-m[6], math.sqrt(m[7]**2 + m[8]**2))
            return roll, pitch
        except Exception as e:
            print(f"ошибка ориентации трапа: {e}")
            return 0, 0
    
    def update_plots(self, current_time):
        if len(self.time_history) < 2:
            return
            
        times = np.array(self.time_history)
        
        barge_roll_deg = [r * 180/math.pi for r in self.barge_roll_history]
        barge_pitch_deg = [p * 180/math.pi for p in self.barge_pitch_history]
        gangway_roll_deg = [r * 180/math.pi for r in self.gangway_roll_history]
        gangway_pitch_deg = [p * 180/math.pi for p in self.gangway_pitch_history]
        error_roll_deg = [e * 180/math.pi/2.2 for e in self.error_roll_history]
        error_pitch_deg = [e * 180/math.pi/2.2 for e in self.error_pitch_history]
        
        self.line_roll_error.set_data(times, error_roll_deg)
        self.line_pitch_error.set_data(times, error_pitch_deg)
        self.line_gangway_roll.set_data(times, gangway_roll_deg)
        self.line_gangway_pitch.set_data(times, gangway_pitch_deg)
        self.line_barge_roll.set_data(times, barge_roll_deg)
        self.line_barge_pitch.set_data(times, barge_pitch_deg)
        
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.relim()
            ax.autoscale_view()
        
        if len(self.error_roll_history) >= 20:
            recent_err_roll = list(self.error_roll_history)[-20:]
            recent_err_pitch = list(self.error_pitch_history)[-20:]
            
            recent_err_roll_deg = [e * 180/math.pi for e in recent_err_roll]
            recent_err_pitch_deg = [e * 180/math.pi for e in recent_err_pitch]
            
            rmse_roll = np.sqrt(np.mean(np.square(recent_err_roll_deg)))
            rmse_pitch = np.sqrt(np.mean(np.square(recent_err_pitch_deg)))
            max_roll_err = max(abs(e) for e in recent_err_roll_deg)
            max_pitch_err = max(abs(e) for e in recent_err_pitch_deg)
            
            stats = (f"статус: ok\n"
                    f"ошибка стабилизации (последние 20 шагов):\n"
                    f"крен (ошибка):\n"
                    f"  скз: {rmse_roll:.4f}°\n"
                    f"  макс: {max_roll_err:.4f}°\n\n"
                    f"тангаж (ошибка):\n"
                    f"  скз: {rmse_pitch:.4f}°\n"
                    f"  макс: {max_pitch_err:.4f}°")
            
            self.stats_text.set_text(stats)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def run(self):
        print("запуск контроллера плоттера...")
        
        step_count = 0
        while self.step(self.timestep) != -1:
            current_time = self.getTime()
            
            barge_roll, barge_pitch = self.get_barge_orientation()
            gangway_roll, gangway_pitch = self.get_gangway_orientation()
            
            error_roll = -gangway_roll
            error_pitch = -gangway_pitch
            
            self.time_history.append(current_time)
            self.barge_roll_history.append(barge_roll)
            self.barge_pitch_history.append(barge_pitch)
            self.gangway_roll_history.append(gangway_roll)
            self.gangway_pitch_history.append(gangway_pitch)
            self.error_roll_history.append(error_roll)
            self.error_pitch_history.append(error_pitch)
            
            step_count += 1
            if step_count % 100 == 0:
                print(f"\nвремя: {current_time:.1f}с")
                print(f"  баржа:   крен={barge_roll*180/math.pi:6.2f}° тангаж={barge_pitch*180/math.pi:6.2f}°")
                print(f"  трап:    крен={gangway_roll*180/math.pi:6.2f}° тангаж={gangway_pitch*180/math.pi:6.2f}°")
                print(f"  ошибка:  крен={error_roll*180/math.pi:6.2f}° тангаж={error_pitch*180/math.pi:6.2f}°")
                print("-" * 60)
            
            self.update_plots(current_time)

if __name__ == "__main__":
    controller = PlotterController()
    controller.run()