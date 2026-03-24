import math
import random
from controller import Supervisor

class SpectralWaveGenerator:
    def __init__(self, grid_size=128, spacing=18.0, significant_height=0.3, peak_period=4.0):
        self.N = grid_size
        self.dx = spacing
        self.dy = spacing
        self.Hs = significant_height
        self.Tp = peak_period
        self.omega_p = 2 * math.pi / self.Tp
        self.num_components = 24
        
        random.seed(42)
        self.components = []
        for i in range(self.num_components):
            omega = self.omega_p * (0.4 + i * 2.0 / self.num_components)
            k = omega**2 / 9.81
            S_omega = (8.1e-3 * 9.81**2) / (omega**5) * math.exp(-1.25 * (self.omega_p/omega)**4)
            amp = math.sqrt(2 * S_omega * (self.omega_p / self.num_components)) * self.Hs / 2.5
            phase = random.uniform(0, 2*math.pi)
            direction = random.uniform(-math.pi/4, math.pi/4)
            self.components.append((k, omega, amp, phase, direction))
    
    def compute_heights(self, time):
        heights = []
        half_x = self.N * self.dx / 2
        half_y = self.N * self.dy / 2
        
        for jy in range(self.N):
            y = -half_y + jy * self.dy
            for ix in range(self.N):
                x = -half_x + ix * self.dx
                elevation = 0.0
                for k, omega, amp, phase, direction in self.components:
                    kx = k * math.cos(direction)
                    ky = k * math.sin(direction)
                    elevation += amp * math.cos(kx * x + ky * y - omega * time + phase)
                swell = 0.05 * math.sin(0.2 * x - 0.6 * time) * math.cos(0.15 * y + 0.4 * time)
                heights.append(elevation + swell)
        return heights


class WaterController(Supervisor):
    def __init__(self):
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())
        self.dt = self.timestep / 1000.0
        
        self.water = self.getFromDef("WATER")
        if not self.water:
            print("WATER not found")
            return
        
        children = self.water.getChildren()
        if not children:
            print("No children in WATER")
            return
        self.elevation_grid = children[0].getGeometry()
        
        self.wave_gen = SpectralWaveGenerator(
            grid_size=128, spacing=18.0, significant_height=0.3, peak_period=4.0
        )
        
        self.time = 0.0
        self.update_interval = 0.08
        self.update_counter = 0
        print("WaterController ready")

    def run(self):
        while self.step(self.timestep) != -1:
            self.time += self.dt
            self.update_counter += 1
            
            if self.update_counter >= int(self.update_interval / self.dt):
                self.update_counter = 0
                heights = self.wave_gen.compute_heights(self.time)
                try:
                    self.elevation_grid.setHeight(heights)
                except:
                    pass
            self.step(self.timestep)


if __name__ == "__main__":
    WaterController().run()