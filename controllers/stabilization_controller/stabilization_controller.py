from controller import Supervisor
import math
import random

class KinematicModel:
    def __init__(self):
        self.k_pitch = 0.15
        self.k_roll = 0.15
        
        self.linear_limit = math.radians(10.0)
        self.nonlinear_warning = False
        
        self.c2_pitch = 0.015
        self.c2_roll = 0.015
        
        self.max_stroke = 0.15
        self.max_compensable_angle = math.radians(15.4)

    def get_absolute_angles(self, platform_roll, platform_pitch, base_roll, base_pitch):
        abs_roll = platform_roll + base_roll
        abs_pitch = platform_pitch + base_pitch
        return abs_roll, abs_pitch

    def get_required_compensation(self, base_roll, base_pitch, use_nonlinear=True):
        if use_nonlinear and (abs(base_roll) > self.linear_limit or abs(base_pitch) > self.linear_limit):
            target_roll = self._nonlinear_compensation(base_roll, 'roll')
            target_pitch = self._nonlinear_compensation(base_pitch, 'pitch')
        else:
            target_roll = -base_roll
            target_pitch = -base_pitch
        return target_roll, target_pitch

    def _nonlinear_compensation(self, base_angle, axis):
        sign = -1 if base_angle >= 0 else 1
        abs_angle = abs(base_angle)
        
        if abs_angle > self.linear_limit:
            excess = abs_angle - self.linear_limit
            correction_factor = 1.0 + self.c2_pitch * excess * 10.0
        else:
            correction_factor = 1.0
            
        return sign * abs_angle * correction_factor

    def check_linearization(self, angle):
        if abs(angle) > self.linear_limit and not self.nonlinear_warning:
            print(f"WARNING: Angle {math.degrees(angle):.1f}° > 10°. Nonlinear correction enabled.")
            self.nonlinear_warning = True
        elif abs(angle) <= self.linear_limit:
            self.nonlinear_warning = False

    def angle_to_stroke(self, angle, axis='pitch'):
        k = self.k_pitch if axis == 'pitch' else self.k_roll
        return angle * k

    def check_stroke_limit(self, stroke):
        return abs(stroke) <= self.max_stroke

    def get_stroke_margin(self, stroke):
        margin = 1.0 - abs(stroke) / self.max_stroke
        return max(0.0, min(1.0, margin))

class ActuatorDynamics:
    def __init__(self, dt, time_constant=0.05):
        self.dt = dt
        self.T_eq = time_constant
        
        self.a1 = math.exp(-dt / time_constant)
        self.b0 = 1.0 - self.a1
        
        self.state = 0.0
        self.prev_input = 0.0
        
        self.saturation_count = 0
        
    def update(self, input_signal):
        self.state = self.a1 * self.state + self.b0 * input_signal
        self.prev_input = input_signal
        return self.state

    def reset(self):
        self.state = 0.0
        self.prev_input = 0.0
        self.saturation_count = 0

class PID:
    def __init__(self, kp=1.0, ki=0.1, kd=0.5, 
                 max_integral=1.0, output_limit=10.0,
                 derivative_filter=0.1, conditional_integration=True):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.output_limit = output_limit
        self.derivative_filter = derivative_filter
        self.conditional_integration = conditional_integration
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        
        self.stats = {
            'integral_activations': 0,
            'output_saturations': 0,
            'updates': 0
        }
        
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        for key in self.stats:
            self.stats[key] = 0
        
    def update(self, error, dt, error_threshold=0.01, stroke_margin=1.0):
        if dt <= 0: dt = 0.001
        self.stats['updates'] += 1
            
        p_term = self.kp * error
        
        if self.conditional_integration:
            if abs(error) > error_threshold and stroke_margin > 0.1:
                self.integral += error * dt
                self.stats['integral_activations'] += 1
            elif stroke_margin <= 0.1:
                self.integral *= 0.5
        else:
            self.integral += error * dt
            
        self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
        i_term = self.ki * self.integral
        
        raw_derivative = (error - self.prev_error) / dt
        self.filtered_derivative = (self.derivative_filter * raw_derivative + 
                                   (1 - self.derivative_filter) * self.filtered_derivative)
        d_term = self.kd * self.filtered_derivative
        
        self.prev_error = error
        
        output = p_term + i_term + d_term
        
        if abs(output) >= self.output_limit:
            self.stats['output_saturations'] += 1
            
        return max(-self.output_limit, min(self.output_limit, output))
    
    def get_stats(self):
        return self.stats.copy()

class QuantizationNoise:
    def __init__(self, bits=16, voltage_range=10.0, enabled=False):
        self.bits = bits
        self.voltage_range = voltage_range
        self.enabled = enabled
        
        self.q_step = (2.0 * voltage_range) / (2 ** bits)
        self.angle_error_equiv = 0.0003 * math.pi / 180.0
        
    def apply(self, value):
        if not self.enabled:
            return value
        noise = random.uniform(-self.q_step, self.q_step)
        return value + noise

class StabilizationController(Supervisor):
    def __init__(self):
        super().__init__()

        self.timestep = int(self.getBasicTimeStep())
        self.dt = self.timestep / 1000.0
        
        if self.dt > 0.005:
            print(f"WARNING: Sampling step {self.dt*1000:.1f}ms > 2ms. ZOH accuracy may be <98%.")
        else:
            print(f"Sampling step {self.dt*1000:.1f}ms meets recommendations")

        self.barge = self.getFromDef("BARGE")
        
        if self.barge is None:
            print("Error: barge not found!")
            return

        self.roll_motor = self.getDevice("gangway_roll_motor")
        self.pitch_motor = self.getDevice("gangway_pitch_motor")
        self.roll_sensor = self.getDevice("gangway_roll_sensor")
        self.pitch_sensor = self.getDevice("gangway_pitch_sensor")

        self.devices_ok = all([self.roll_motor, self.pitch_motor, self.roll_sensor, self.pitch_sensor])
        
        if not self.devices_ok:
            print("Error: gangway devices not found!")
            return

        self.roll_sensor.enable(self.timestep)
        self.pitch_sensor.enable(self.timestep)

        self.kinematics = KinematicModel()
        
        self.actuator_roll = ActuatorDynamics(self.dt, time_constant=0.05)
        self.actuator_pitch = ActuatorDynamics(self.dt, time_constant=0.05)

        self.roll_pid = PID(
            kp=15000.0, ki=800.0, kd=2500.0,
            max_integral=5000.0, output_limit=45000.0,
            derivative_filter=0.12, conditional_integration=True
        )
        self.pitch_pid = PID(
            kp=18000.0, ki=1000.0, kd=3000.0,
            max_integral=6000.0, output_limit=45000.0,
            derivative_filter=0.12, conditional_integration=True
        )

        self.deadzone = math.radians(0.1)
        
        self.quant_noise = QuantizationNoise(bits=16, enabled=False)
        
        self.stroke_roll = 0.0
        self.stroke_pitch = 0.0
        self.stroke_warning = False
        
        self.warmup_time = 2.0
        self.start_time = 0.0
        self.control_active = False
        
        self.alpha_filter = 0.2
        self.filt_base_roll = 0.0
        self.filt_base_pitch = 0.0
        
        self.stats_timer = 0.0

    def get_barge_orientation(self):
        if self.barge is None:
            return 0.0, 0.0
        m = self.barge.getOrientation()
        roll = math.atan2(m[7], m[8])
        pitch = math.atan2(-m[6], math.sqrt(m[7]**2 + m[8]**2))
        return roll, pitch

    def get_gangway_angles(self):
        roll = self.roll_sensor.getValue()
        pitch = self.pitch_sensor.getValue()
        
        roll = self.quant_noise.apply(roll)
        pitch = self.quant_noise.apply(pitch)
        
        return roll, pitch

    def _update_stroke_estimate(self, plat_roll, plat_pitch):
        self.stroke_roll = self.kinematics.angle_to_stroke(plat_roll, 'roll')
        self.stroke_pitch = self.kinematics.angle_to_stroke(plat_pitch, 'pitch')
        
        roll_margin = self.kinematics.get_stroke_margin(self.stroke_roll)
        pitch_margin = self.kinematics.get_stroke_margin(self.stroke_pitch)
        
        if (roll_margin < 0.2 or pitch_margin < 0.2) and not self.stroke_warning:
            print(f"WARNING: Stroke margin <20%! R={roll_margin:.1f}, P={pitch_margin:.1f}")
            self.stroke_warning = True
        elif roll_margin >= 0.2 and pitch_margin >= 0.2:
            self.stroke_warning = False
            
        return roll_margin, pitch_margin

    def run(self):
        if not self.devices_ok:
            return
            
        for _ in range(50):
            self.step(self.timestep)
            
        rb, pb = self.get_barge_orientation()
        self.filt_base_roll = rb
        self.filt_base_pitch = pb
        
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.actuator_roll.reset()
        self.actuator_pitch.reset()
        
        self.start_time = self.getTime()
        print("Stabilization controller started")
        print(f"Kinematics: linear + nonlinear correction >10°")
        print(f"Actuator: ZOH discretization dt={self.dt*1000:.1f}ms")
        print(f"Stroke: max {self.kinematics.max_stroke*1000:.0f}mm")
        print(f"Deadzone: {math.degrees(self.deadzone):.2f}°")

        while self.step(self.timestep) != -1:
            current_time = self.getTime()
            elapsed = current_time - self.start_time
            
            if elapsed < self.warmup_time and not self.control_active:
                if elapsed >= self.warmup_time:
                    self.control_active = True
                    print(f"[{elapsed:.2f}s] Stabilization activated")
                continue
            
            base_roll, base_pitch = self.get_barge_orientation()
            
            self.filt_base_roll = self.alpha_filter * base_roll + (1 - self.alpha_filter) * self.filt_base_roll
            self.filt_base_pitch = self.alpha_filter * base_pitch + (1 - self.alpha_filter) * self.filt_base_pitch
            
            plat_roll, plat_pitch = self.get_gangway_angles()

            abs_roll, abs_pitch = self.kinematics.get_absolute_angles(
                plat_roll, plat_pitch, self.filt_base_roll, self.filt_base_pitch
            )
            
            self.kinematics.check_linearization(abs_roll)
            self.kinematics.check_linearization(abs_pitch)

            error_roll = -abs_roll 
            error_pitch = -abs_pitch

            if abs(error_roll) < self.deadzone:
                error_roll = 0.0
            if abs(error_pitch) < self.deadzone:
                error_pitch = 0.0

            roll_margin, pitch_margin = self._update_stroke_estimate(plat_roll, plat_pitch)

            raw_torque_roll = self.roll_pid.update(error_roll, self.dt, stroke_margin=roll_margin)
            raw_torque_pitch = self.pitch_pid.update(error_pitch, self.dt, stroke_margin=pitch_margin)

            final_torque_roll = self.actuator_roll.update(raw_torque_roll)
            final_torque_pitch = self.actuator_pitch.update(raw_torque_pitch)

            max_torque = 45000.0
            final_torque_roll = max(-max_torque, min(max_torque, final_torque_roll))
            final_torque_pitch = max(-max_torque, min(max_torque, final_torque_pitch))

            self.roll_motor.setTorque(final_torque_roll)
            self.pitch_motor.setTorque(final_torque_pitch)

            self.stats_timer += self.dt
            if self.stats_timer >= 1.0:
                self.stats_timer = 0.0
                roll_stats = self.roll_pid.get_stats()
                print(f"t={current_time:5.1f}s | AbsErr: R={math.degrees(abs_roll):6.2f}° P={math.degrees(abs_pitch):6.2f}° | "
                      f"Torque: R={final_torque_roll/1000:.1f}kNm P={final_torque_pitch/1000:.1f}kNm | "
                      f"Stroke: R={self.stroke_roll*1000:.0f}mm P={self.stroke_pitch*1000:.0f}mm | "
                      f"PID updates: {roll_stats['updates']}")


if __name__ == "__main__":
    controller = StabilizationController()
    if hasattr(controller, 'devices_ok') and controller.devices_ok:
        controller.run()