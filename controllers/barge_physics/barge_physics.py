from controller import Supervisor
import math

class BargePhysicsController(Supervisor):

    def __init__(self):
        super().__init__()

        self.timestep = int(self.getBasicTimeStep())
        self.barge = self.getSelf()

        self.initial_position = self.barge.getPosition()
        m = self.barge.getOrientation()
        self.initial_roll = math.atan2(m[7], m[8])
        self.initial_pitch = math.atan2(-m[6], math.sqrt(m[7]**2 + m[8]**2))
        self.initial_yaw = math.atan2(m[3], m[0])

        self.t = 0.0
        self.mass = 65000.0

        mass_factor = 50000.0 / self.mass
        self.heave_amp = 0.25 * mass_factor
        self.roll_amp = 0.10 * mass_factor
        self.pitch_amp = 0.07 * mass_factor

        freq_factor = math.sqrt(50000.0 / self.mass)
        self.heave_freq = 0.9 * freq_factor
        self.roll_freq = 0.8 * freq_factor
        self.pitch_freq = 1.0 * freq_factor

        self.roll_phase = 0.7
        self.pitch_phase = 1.4

        self.kp_angle = 10.0
        self.ki_angle = 2.0
        self.kd_angle = 1.0

        self.integral_roll = 0.0
        self.integral_pitch = 0.0
        self.integral_yaw = 0.0

        self.inertia = min(0.6, self.mass / 100000.0)
        self.damping = max(0.15, 0.3 - (self.mass / 200000.0))
        self.prev_heave_vel = 0.0

        self.prev_roll = self.initial_roll
        self.prev_pitch = self.initial_pitch
        self.prev_yaw = self.initial_yaw

        self.debug_timer = 0

    def get_orientation_angles(self):
        m = self.barge.getOrientation()
        roll = math.atan2(m[7], m[8])
        pitch = math.atan2(-m[6], math.sqrt(m[7]**2 + m[8]**2))
        yaw = math.atan2(m[3], m[0])
        return roll, pitch, yaw

    def local_to_world_angular(self, roll_vel, pitch_vel, yaw_vel):
        m = self.barge.getOrientation()
        wx = m[0] * roll_vel + m[1] * pitch_vel + m[2] * yaw_vel
        wy = m[3] * roll_vel + m[4] * pitch_vel + m[5] * yaw_vel
        wz = m[6] * roll_vel + m[7] * pitch_vel + m[8] * yaw_vel
        return wx, wy, wz

    def run(self):
        while self.step(self.timestep) != -1:
            dt = self.timestep / 1000.0
            self.t += dt
            self.debug_timer += dt

            target_roll = self.initial_roll + self.roll_amp * math.sin(self.roll_freq * self.t + self.roll_phase)
            target_pitch = self.initial_pitch + self.pitch_amp * math.sin(self.pitch_freq * self.t + self.pitch_phase)
            target_yaw = self.initial_yaw + 0.01 * math.sin(self.t * 0.3)

            current_roll, current_pitch, current_yaw = self.get_orientation_angles()

            error_roll = target_roll - current_roll
            error_pitch = target_pitch - current_pitch
            error_yaw = target_yaw - current_yaw

            self.integral_roll += error_roll * dt
            self.integral_pitch += error_pitch * dt
            self.integral_yaw += error_yaw * dt

            max_integral = 0.5
            self.integral_roll = max(-max_integral, min(max_integral, self.integral_roll))
            self.integral_pitch = max(-max_integral, min(max_integral, self.integral_pitch))
            self.integral_yaw = max(-max_integral, min(max_integral, self.integral_yaw))

            roll_rate = (current_roll - self.prev_roll) / dt
            pitch_rate = (current_pitch - self.prev_pitch) / dt
            yaw_rate = (current_yaw - self.prev_yaw) / dt

            self.prev_roll = current_roll
            self.prev_pitch = current_pitch
            self.prev_yaw = current_yaw

            ctrl_roll_vel = (self.kp_angle * error_roll + 
                            self.ki_angle * self.integral_roll - 
                            self.kd_angle * roll_rate)
            ctrl_pitch_vel = (self.kp_angle * error_pitch + 
                             self.ki_angle * self.integral_pitch - 
                             self.kd_angle * pitch_rate)
            ctrl_yaw_vel = (self.kp_angle * error_yaw + 
                           self.ki_angle * self.integral_yaw - 
                           self.kd_angle * yaw_rate)

            max_ang_vel = 2.0
            ctrl_roll_vel = max(-max_ang_vel, min(max_ang_vel, ctrl_roll_vel))
            ctrl_pitch_vel = max(-max_ang_vel, min(max_ang_vel, ctrl_pitch_vel))
            ctrl_yaw_vel = max(-max_ang_vel, min(max_ang_vel, ctrl_yaw_vel))

            heave_vel_raw = self.heave_amp * self.heave_freq * math.cos(self.heave_freq * self.t)
            heave_vel = heave_vel_raw * (1 - self.inertia) + self.prev_heave_vel * self.inertia
            heave_vel = heave_vel * (1 - self.damping) + self.prev_heave_vel * self.damping
            self.prev_heave_vel = heave_vel

            wx, wy, wz = self.local_to_world_angular(ctrl_roll_vel, ctrl_pitch_vel, ctrl_yaw_vel)

            vx = 0.0
            vy = 0.0
            vz = heave_vel

            self.barge.setVelocity([vx, vy, vz, wx, wy, wz])

            if self.debug_timer > 0.5:
                self.debug_timer = 0


if __name__ == "__main__":
    controller = BargePhysicsController()
    controller.run()