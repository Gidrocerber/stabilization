from controller import Supervisor
import math
import random

class WaveSpectrum:
    def __init__(self):
        self.sea_states = [
            {"Hs": 0.15, "Tp": 3.5},
            {"Hs": 0.175, "Tp": 3.75},
            {"Hs": 0.2, "Tp": 4.0},
            {"Hs": 0.225, "Tp": 4.25},
        ]

        self.current_state = 0
        self.last_switch = 0
        self.Hs = self.sea_states[0]["Hs"]
        self.Tp = self.sea_states[0]["Tp"]
        self.omega_p = 2 * math.pi / self.Tp
        
        self.g = 9.81

        random.seed(42)

        self.N = 512 
        self.components = []
        self._generate_spectrum()

        self.rogue_time = -100
        self.rogue_amp = 0
        self.rogue_x = 0
        self.rogue_y = 0
        self.rogue_vx = 2.0

    def _generate_spectrum(self):
        self.components = []
        
        omega_min = self.omega_p * 0.5
        omega_max = self.omega_p * 2.5
        
        for i in range(self.N):
            omega = omega_min + (omega_max - omega_min) * (i / self.N)
            
            pm_factor = (omega / self.omega_p) ** (-5) * math.exp(-1.25 * (self.omega_p / omega) ** 4)
            
            d_omega = (omega_max - omega_min) / self.N
            base_amp = math.sqrt(2 * pm_factor * d_omega) * self.Hs * 0.5
            
            phase = random.uniform(0, 2 * math.pi)
            direction = random.gauss(0, 0.3)
            
            phase2 = random.uniform(0, 2 * math.pi)
            amp2 = base_amp * 0.15
            
            self.components.append({
                'omega': omega,
                'phase': phase,
                'direction': direction,
                'amp': base_amp,
                'amp2': amp2,
                'phase2': phase2
            })

    def set_state(self, state):
        self.Hs = state["Hs"]
        self.Tp = state["Tp"]
        self.omega_p = 2 * math.pi / self.Tp
        self._generate_spectrum()

    def update(self, t):
        if t - self.last_switch > 60:
            self.current_state = (self.current_state + 1) % len(self.sea_states)
            self.set_state(self.sea_states[self.current_state])
            self.last_switch = t

        if random.random() < 0.0005:
            self.rogue_time = t
            self.rogue_amp = min(self.Hs * random.uniform(2.2, 2.8), 1.5)
            self.rogue_x = random.uniform(-50, 50)
            self.rogue_y = random.uniform(-50, 50)

    def get(self, x, y, t):
        h = 0
        k_factor = 1.0 / self.g

        for c in self.components:
            omega = c['omega']
            k = (omega ** 2) * k_factor
            
            arg1 = k * (x * math.cos(c['direction']) + y * math.sin(c['direction'])) - omega * t + c['phase']
            h += c['amp'] * math.cos(arg1)
            
            arg2 = 2 * k * (x * math.cos(c['direction']) + y * math.sin(c['direction'])) - 2 * omega * t + c['phase2']
            h += c['amp2'] * math.cos(arg2)

        dt = t - self.rogue_time
        if 0 < dt < 10:
            cx = self.rogue_x + self.rogue_vx * dt
            cy = self.rogue_y
            
            dist_sq = (x - cx)**2 + (y - cy)**2
            
            spatial_env = math.exp(-dist_sq / 150.0)
            temporal_env = math.exp(-((dt - 4.0)**2) / 4.0)
            
            h += self.rogue_amp * spatial_env * temporal_env * math.cos(2 * math.pi * dt / 3.0)

        return h

class BargePhysicsController(Supervisor):

    def __init__(self):
        super().__init__()
        self.dt = self.getBasicTimeStep() / 1000.0
        self.barge = self.getSelf()

        self.mass = 65000
        self.g = 9.81
        self.rho = 1025

        self.Awp = 230.0
        self.draft = 1.2
        self.hull_offset = 1.0
        self.L = 40
        self.B = 10

        self.GM_roll = 1.5
        self.GM_pitch = 2.0

        self.heave_damping_linear = 50000
        self.heave_damping_quad = 15000
        
        self.roll_damping_linear = 200000
        self.roll_damping_quad = 80000
        
        self.pitch_damping_linear = 250000
        self.pitch_damping_quad = 100000

        self.Cd = 1.2
        self.area_x = 40
        self.area_y = 80
        self.area_z = 230

        self.added_mass_heave = 0.5 * self.mass
        self.added_mass_roll = 0.3 * self.mass
        self.added_mass_pitch = 0.3 * self.mass

        self.wave = WaveSpectrum()
        self.water_level = -1.5

        self.anchor_point = [-34.44, 0]
        self.anchor_k = 25000
        self.anchor_d = 12000

        self.mooring_points = [(-10, -4), (10, -4), (-10, 4), (10, 4)]
        self.mooring_k = 15000
        self.mooring_d = 8000
        self.mooring_rest = 5.0

        self.float_points = [
            (-8,-3), (8,-3), (-8,3), (8,3),
            (-4,-2), (4,-2), (-4,2), (4,2)
        ]

        self.barge.getField("translation").setSFVec3f([-34.44, 0, -1])
        self.barge.setVelocity([0,0,0,0,0,0])
        
        self.prev_roll = 0
        self.prev_pitch = 0
        self.prev_t = 0

    def get_angles(self):
        m = self.barge.getOrientation()
        roll = math.atan2(m[7], m[8])
        pitch = math.atan2(-m[6], math.sqrt(m[7]**2 + m[8]**2))
        return roll, pitch

    def run(self):
        while self.step(int(self.dt * 1000)) != -1:
            pos = self.barge.getPosition()
            vel = self.barge.getVelocity()
            vx, vy, vz = vel[:3]
            wx, wy, wz = vel[3:]

            roll, pitch = self.get_angles()
            t = self.getTime()

            self.wave.update(t)

            total_force = [0, 0, 0]
            total_torque = [0, 0, 0]

            area_per_point = self.Awp / len(self.float_points)

            for px, py in self.float_points:
                wx_pos = pos[0] + px
                wy_pos = pos[1] + py

                water_z = self.water_level + self.wave.get(wx_pos, wy_pos, t)
                
                hull_z = pos[2] - self.hull_offset

                sub = max(0.0, min(water_z - hull_z, self.draft))

                volume = area_per_point * sub
                fz = self.rho * self.g * volume

                total_force[2] += fz
                total_torque[0] += py * fz
                total_torque[1] -= px * fz

            effective_mass = self.mass + self.added_mass_heave
            total_force[2] -= effective_mass * self.g

            total_force[2] -= self.heave_damping_linear * vz
            total_force[2] -= self.heave_damping_quad * vz * abs(vz)
            
            total_torque[0] += -self.GM_roll * self.mass * self.g * roll
            total_torque[0] -= self.roll_damping_linear * wx
            total_torque[0] -= self.roll_damping_quad * wx * abs(wx)
            
            total_torque[1] += -self.GM_pitch * self.mass * self.g * pitch
            total_torque[1] -= self.pitch_damping_linear * wy
            total_torque[1] -= self.pitch_damping_quad * wy * abs(wy)

            dx = pos[0] - self.anchor_point[0]
            dy = pos[1] - self.anchor_point[1]
            total_force[0] += -self.anchor_k * dx - self.anchor_d * vx
            total_force[1] += -self.anchor_k * dy - self.anchor_d * vy

            for mx, my in self.mooring_points:
                dx = pos[0] + mx - self.anchor_point[0]
                dy = pos[1] + my - self.anchor_point[1]
                dist = math.sqrt(dx*dx + dy*dy)

                if dist > self.mooring_rest:
                    stretch = dist - self.mooring_rest
                    k_eff = self.mooring_k * (1.0 + 0.5 * stretch / self.mooring_rest)
                    
                    Fx = -k_eff * stretch * (dx/dist)
                    Fy = -k_eff * stretch * (dy/dist)
                    
                    Fx -= self.mooring_d * vx
                    Fy -= self.mooring_d * vy

                    total_force[0] += Fx
                    total_force[1] += Fy

            total_force[0] += -0.5 * self.rho * self.Cd * self.area_x * vx * abs(vx)
            total_force[1] += -0.5 * self.rho * self.Cd * self.area_y * vy * abs(vy)
            total_force[2] += -0.5 * self.rho * self.Cd * self.area_z * vz * abs(vz)

            contact_x = 30
            dx_wall = pos[0] - contact_x
            if dx_wall > 0:
                k_wall = 300000
                d_wall = 20000
                total_force[0] += -k_wall * dx_wall - d_wall * vx

            self.barge.addForce(total_force, True)
            self.barge.addTorque(total_torque, True)

            if int(t * 10) % 20 == 0:
                print(f"t={t:.1f} z={pos[2]:.2f} roll={math.degrees(roll):.1f}")

if __name__ == "__main__":
    BargePhysicsController().run()