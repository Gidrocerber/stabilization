from controller import Supervisor
import math
import random

class GeometryParams:
    L1_PITCH = 0.45
    D10_PITCH = 0.62
    H1_PITCH = math.sqrt(max(0.0, D10_PITCH**2 - L1_PITCH**2)) - D10_PITCH
    
    L2_ROLL = 0.45
    D20_ROLL = 0.62
    H2_ROLL = math.sqrt(max(0.0, D20_ROLL**2 - L2_ROLL**2)) - D20_ROLL
    
    DELTA_D_MAX = 0.15
    MAX_FORCE = 50000
    MAX_TORQUE = 45000

class ActuatorKinematics:
    def __init__(self, l=0.45, h=None, d0=0.62, max_stroke=0.15, max_force=50000):
        self.l = l
        self.d0 = d0
        self.h = h if h is not None else (math.sqrt(max(0.0, d0**2 - l**2)) - d0)
        self.max_stroke = max_stroke
        self.max_force = max_force
        
        self.k_p = (self.l * (self.h + self.d0)) / self.d0
        self.k_p2 = -(self.l**2 * (self.h + self.d0)**2) / (self.d0**3)
        
        h_d0 = self.h + self.d0
        self.d_neutral = math.sqrt(max(0.0, h_d0**2 + self.l**2))
        
        h_expected = math.sqrt(max(0.0, self.d0**2 - self.l**2)) - self.d0
        self.geom_valid = abs(self.h - h_expected) < 0.01
        
    def angle_to_stroke_nonlinear(self, angle):
        h_d0 = self.h + self.d0
        term = h_d0**2 + self.l**2 - 2 * self.l * h_d0 * math.sin(angle)
        term = max(0.0, term)
        d_current = math.sqrt(term)
        stroke = -self.d0 + d_current
        stroke = max(-self.max_stroke, min(self.max_stroke, stroke))
        return stroke
    
    def angle_to_stroke_linear(self, angle):
        stroke = -self.k_p * angle
        return max(-self.max_stroke, min(self.max_stroke, stroke))
    
    def get_linearization_error(self, angle):
        x_nonlinear = self.angle_to_stroke_nonlinear(angle)
        x_linear = self.angle_to_stroke_linear(angle)
        if abs(x_nonlinear) < 1e-6:
            return 0.0
        return abs((x_nonlinear - x_linear) / x_nonlinear) * 100.0
    
    def force_to_torque(self, force, angle):
        k_eff = self.k_p * math.cos(angle)
        return force * k_eff
    
    def get_stroke_margin(self, stroke):
        return max(0.0, min(1.0, 1.0 - abs(stroke) / self.max_stroke))
    
    def get_max_compensatable_angle(self):
        return self.max_stroke / abs(self.k_p) if abs(self.k_p) > 0 else 0.0

class HydraulicActuatorDynamics:
    def __init__(self, dt, max_stroke=0.15, max_force=50000.0):
        self.dt = dt
        self.max_stroke = max_stroke
        self.max_force = max_force
        
        mass = 500.0
        area = 0.025
        beta_e = 1.4e9
        volume = 2.5e-4
        leakage = 2.5e-12
        damping = 1500.0
        
        self.a1 = (damping * 4 * beta_e * leakage / volume + 
                   4 * beta_e * area**2 / volume)
        
        self.max_stroke_rate = 0.5
        self.stroke = 0.0
        self.stroke_rate = 0.0
        
    def update(self, target_force, current_stroke):
        target_stroke_rate = target_force / (self.a1 * 0.01) if self.a1 > 0 else 0.0
        target_stroke_rate = max(-self.max_stroke_rate, min(self.max_stroke_rate, target_stroke_rate))
        
        self.stroke_rate += (target_stroke_rate - self.stroke_rate) * self.dt * 10
        self.stroke += self.stroke_rate * self.dt
        self.stroke = max(-self.max_stroke, min(self.max_stroke, self.stroke))
        
        actual_force = target_force * 0.98
        actual_force = max(-self.max_force, min(self.max_force, actual_force))
        return actual_force
    
    def reset(self):
        self.stroke = 0.0
        self.stroke_rate = 0.0

class DiscretePID:
    def __init__(self, kp=1.0, ki=0.1, kd=0.5, dt=0.002,
                 max_integral=1.0, output_limit=10.0,
                 derivative_filter=0.1, conditional_integration=False,
                 backlash=0.001):
        
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.max_integral = max_integral
        self.output_limit = output_limit
        self.derivative_filter = derivative_filter
        self.conditional_integration = conditional_integration
        self.backlash = backlash
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        
        self.stats = {
            'integral_activations': 0,
            'output_saturations': 0,
            'backlash_activations': 0,
            'updates': 0
        }
    
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        for key in self.stats:
            self.stats[key] = 0
    
    def apply_backlash(self, error):
        if abs(error) <= self.backlash:
            self.stats['backlash_activations'] += 1
            return 0.0
        return error - self.backlash if error > 0 else error + self.backlash
    
    def update(self, error, dt, margin=1.0, error_threshold=0.001):
        if dt <= 0:
            dt = 0.002
        
        self.stats['updates'] += 1
        error_eff = self.apply_backlash(error)
        
        p_term = self.kp * error_eff
        
        if self.conditional_integration:
            if abs(error_eff) > error_threshold and margin > 0.1:
                self.integral += error_eff * dt
                self.stats['integral_activations'] += 1
            elif margin <= 0.1:
                self.integral *= 0.5
        else:
            self.integral += error_eff * dt
        
        self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
        i_term = self.ki * self.integral
        
        raw_derivative = (error_eff - self.prev_error) / dt
        self.filtered_derivative = (self.derivative_filter * raw_derivative + 
                                     (1 - self.derivative_filter) * self.filtered_derivative)
        d_term = self.kd * self.filtered_derivative
        
        self.prev_error = error_eff
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
        
        k_x = 1.2e-5
        k_amp = 5.0
        k_p1 = 0.562
        if k_x * k_amp > 0:
            self.pos_error_deg = math.degrees((self.q_step * k_p1) / (k_x * k_amp))
        else:
            self.pos_error_deg = 0.0
    
    def apply(self, value):
        if not self.enabled:
            return value
        return value + random.uniform(-self.q_step, self.q_step)
    
    def get_quantization_error_deg(self):
        return self.pos_error_deg

class StabilizationController(Supervisor):
    def __init__(self):
        super().__init__()

        self.timestep = int(self.getBasicTimeStep())
        self.dt = self.timestep / 1000.0

        self.barge = self.getFromDef("BARGE")
        if self.barge is None:
            print("Error: barge not found!")
            return

        self.roll_motor = self.getDevice("gangway_roll_motor")
        self.pitch_motor = self.getDevice("gangway_pitch_motor")
        self.roll_sensor = self.getDevice("gangway_roll_sensor")
        self.pitch_sensor = self.getDevice("gangway_pitch_sensor")

        self.devices_ok = all([self.roll_motor, self.pitch_motor, 
                               self.roll_sensor, self.pitch_sensor])
        if not self.devices_ok:
            print("Error: gangway devices not found!")
            return

        self.roll_sensor.enable(self.timestep)
        self.pitch_sensor.enable(self.timestep)

        self.geom_roll = ActuatorKinematics(
            l=GeometryParams.L2_ROLL,
            h=GeometryParams.H2_ROLL,
            d0=GeometryParams.D20_ROLL,
            max_stroke=GeometryParams.DELTA_D_MAX,
            max_force=GeometryParams.MAX_FORCE
        )
        self.geom_pitch = ActuatorKinematics(
            l=GeometryParams.L1_PITCH,
            h=GeometryParams.H1_PITCH,
            d0=GeometryParams.D10_PITCH,
            max_stroke=GeometryParams.DELTA_D_MAX,
            max_force=GeometryParams.MAX_FORCE
        )

        self.act_roll = HydraulicActuatorDynamics(
            self.dt,
            max_stroke=GeometryParams.DELTA_D_MAX,
            max_force=GeometryParams.MAX_FORCE
        )
        self.act_pitch = HydraulicActuatorDynamics(
            self.dt,
            max_stroke=GeometryParams.DELTA_D_MAX,
            max_force=GeometryParams.MAX_FORCE
        )

        backlash_rad = math.radians(0.03)
        
        self.roll_pid = DiscretePID(
            kp=250000.0,
            ki=10000.0,
            kd=8000.0,
            dt=self.dt,
            max_integral=20000.0,
            output_limit=45000.0,
            derivative_filter=0.12,
            conditional_integration=False,
            backlash=backlash_rad
        )
        self.pitch_pid = DiscretePID(
            kp=280000.0,
            ki=12000.0,
            kd=10000.0,
            dt=self.dt,
            max_integral=25000.0,
            output_limit=45000.0,
            derivative_filter=0.12,
            conditional_integration=False,
            backlash=backlash_rad
        )

        self.deadzone = math.radians(0.02)
        
        self.quant_noise = QuantizationNoise(bits=16, voltage_range=10.0, enabled=False)
        
        self.stroke_roll = 0.0
        self.stroke_pitch = 0.0
        self.stroke_warning = False
        
        self.warmup_time = 2.0
        self.start_time = 0.0
        self.control_active = False
        
        self.alpha_filter = 0.25
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
        roll = self.quant_noise.apply(self.roll_sensor.getValue())
        pitch = self.quant_noise.apply(self.pitch_sensor.getValue())
        return roll, pitch

    def _update_stroke_estimate(self, plat_roll, plat_pitch):
        self.stroke_roll = self.geom_roll.angle_to_stroke_nonlinear(plat_roll)
        self.stroke_pitch = self.geom_pitch.angle_to_stroke_nonlinear(plat_pitch)
        
        roll_margin = self.geom_roll.get_stroke_margin(self.stroke_roll)
        pitch_margin = self.geom_pitch.get_stroke_margin(self.stroke_pitch)
        
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
        self.act_roll.reset()
        self.act_pitch.reset()
        
        self.start_time = self.getTime()

        while self.step(self.timestep) != -1:
            current_time = self.getTime()
            elapsed = current_time - self.start_time
            
            if elapsed < self.warmup_time and not self.control_active:
                if elapsed >= self.warmup_time:
                    self.control_active = True
                continue
            
            base_roll, base_pitch = self.get_barge_orientation()
            self.filt_base_roll = (self.alpha_filter * base_roll + 
                                   (1 - self.alpha_filter) * self.filt_base_roll)
            self.filt_base_pitch = (self.alpha_filter * base_pitch + 
                                    (1 - self.alpha_filter) * self.filt_base_pitch)
            
            plat_roll, plat_pitch = self.get_gangway_angles()

            abs_roll = plat_roll + self.filt_base_roll
            abs_pitch = plat_pitch + self.filt_base_pitch

            error_roll = -abs_roll 
            error_pitch = -abs_pitch

            if abs(error_roll) < self.deadzone:
                error_roll = 0.0
            if abs(error_pitch) < self.deadzone:
                error_pitch = 0.0

            roll_margin, pitch_margin = self._update_stroke_estimate(plat_roll, plat_pitch)

            target_force_roll = self.roll_pid.update(error_roll, self.dt, margin=roll_margin)
            target_force_pitch = self.pitch_pid.update(error_pitch, self.dt, margin=pitch_margin)

            actual_force_roll = self.act_roll.update(target_force_roll, self.stroke_roll)
            actual_force_pitch = self.act_roll.update(target_force_pitch, self.stroke_pitch)

            torque_roll = self.geom_roll.force_to_torque(actual_force_roll, plat_roll)
            torque_pitch = self.geom_pitch.force_to_torque(actual_force_pitch, plat_pitch)

            torque_roll = max(-GeometryParams.MAX_TORQUE, 
                              min(GeometryParams.MAX_TORQUE, torque_roll))
            torque_pitch = max(-GeometryParams.MAX_TORQUE, 
                               min(GeometryParams.MAX_TORQUE, torque_pitch))

            self.roll_motor.setTorque(torque_roll)
            self.pitch_motor.setTorque(torque_pitch)

            self.stats_timer += self.dt
            if self.stats_timer >= 1.0:
                self.stats_timer = 0.0
                
                roll_stats = self.roll_pid.get_stats()
                pitch_stats = self.pitch_pid.get_stats()
                
                lin_err_roll = self.geom_roll.get_linearization_error(plat_roll)
                lin_err_pitch = self.geom_pitch.get_linearization_error(plat_pitch)
                
                print(f"t={current_time:5.1f}s | "
                      f"Err: R={math.degrees(abs_roll):6.2f}° P={math.degrees(abs_pitch):6.2f}° | "
                      f"Force: R={actual_force_roll/1000:6.2f}kN P={actual_force_pitch/1000:6.2f}kN | "
                      f"Torque: R={torque_roll/1000:6.2f}kNm P={torque_pitch/1000:6.2f}kNm | "
                      f"Stroke: R={self.stroke_roll*1000:5.1f}mm P={self.stroke_pitch*1000:5.1f}mm | "
                      f"Margin: R={roll_margin:.2f} P={pitch_margin:.2f} | "
                      f"LinErr: R={lin_err_roll:4.1f}% P={lin_err_pitch:4.1f}% | "
                      f"Sat: R={roll_stats['output_saturations']:3d} P={pitch_stats['output_saturations']:3d}")

if __name__ == "__main__":
    controller = StabilizationController()
    if hasattr(controller, 'devices_ok') and controller.devices_ok:
        controller.run()