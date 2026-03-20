import math
from controller import Supervisor

class WaterController(Supervisor):
    def __init__(self):
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())
        self.water = self.getFromDef("WATER")
        if not self.water:
            print("Ошибка: WATER не найден")
            return
        self.time = 0.0

    def run(self):
        while self.step(self.timestep) != -1:
            dt = self.timestep / 1000.0
            self.time += dt

            heave = 0.3 * math.sin(2.0 * math.pi * 0.2 * self.time)

            roll = 0.05 * math.sin(2.0 * math.pi * 0.15 * self.time + 1.0)
            pitch = 0.05 * math.cos(2.0 * math.pi * 0.18 * self.time)

            self.water.setPosition([0, 0, -0.5 + heave])
            self.water.setRotation([1, 0, 0, roll])

if __name__ == "__main__":
    controller = WaterController()
    controller.run()