from controller import Supervisor
import math


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
        self.prev_time = 0.0
        
        self.stats = {
            'p_term': 0.0, 'i_term': 0.0, 'd_term': 0.0,
            'updates': 0, 'integral_activations': 0
        }
        
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        self.prev_time = 0.0
        for key in self.stats:
            self.stats[key] = 0.0 if isinstance(self.stats[key], float) else 0
        
    def update(self, error, dt, error_threshold=0.01):
        if dt <= 0:
            dt = 0.001
            
        self.stats['updates'] += 1
        
        p_term = self.kp * error
        self.stats['p_term'] = p_term
        
        if self.conditional_integration:
            if abs(error) > error_threshold and abs(self.integral) < self.max_integral * 0.8:
                self.integral += error * dt
                self.stats['integral_activations'] += 1
        else:
            self.integral += error * dt
            
        self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
        i_term = self.ki * self.integral
        self.stats['i_term'] = i_term
        
        raw_derivative = (error - self.prev_error) / dt
        self.filtered_derivative = (self.derivative_filter * raw_derivative + 
                                   (1 - self.derivative_filter) * self.filtered_derivative)
        d_term = self.kd * self.filtered_derivative
        self.stats['d_term'] = d_term
        
        self.prev_error = error
        self.prev_time += dt
        
        output = p_term + i_term + d_term
        
        output = max(-self.output_limit, min(self.output_limit, output))
        
        return output
    
    def get_stats(self):
        return self.stats.copy()


class LowPassFilter:
    
    def __init__(self, alpha=0.2, cutoff_freq=None, sample_rate=None):
        self.alpha = alpha
        self.value = 0.0
        self.initialized = False
        
        if cutoff_freq is not None and sample_rate is not None:
            rc = 1.0 / (2.0 * math.pi * cutoff_freq)
            dt = 1.0 / sample_rate
            self.alpha = dt / (rc + dt)
    
    def update(self, new_value):
        if not self.initialized:
            self.value = new_value
            self.initialized = True
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value
    
    def reset(self, value=0.0):
        self.value = value
        self.initialized = True


class StabilizationController(Supervisor):

    def __init__(self):
        super().__init__()

        self.timestep = int(self.getBasicTimeStep())
        self.dt = self.timestep / 1000.0
        self.sample_rate = 1.0 / self.dt

        self.barge = self.getFromDef("BARGE")
        
        if self.barge is None:
            return

        self.roll_motor = self.getDevice("gangway_roll_motor")
        self.pitch_motor = self.getDevice("gangway_pitch_motor")
        self.roll_sensor = self.getDevice("gangway_roll_sensor")
        self.pitch_sensor = self.getDevice("gangway_pitch_sensor")

        self.devices_ok = all([
            self.roll_motor, self.pitch_motor,
            self.roll_sensor, self.pitch_sensor
        ])
        
        if not self.devices_ok:
            return

        self.roll_sensor.enable(self.timestep)
        self.pitch_sensor.enable(self.timestep)

        self.roll_pid = PID(
            kp=2.5, ki=0.2, kd=0.6,
            max_integral=0.2, output_limit=0.5,
            derivative_filter=0.15,
            conditional_integration=True
        )
        self.pitch_pid = PID(
            kp=2.5, ki=0.2, kd=0.6,
            max_integral=0.2, output_limit=0.5,
            derivative_filter=0.15,
            conditional_integration=True
        )

        self.roll_filter = LowPassFilter(alpha=0.15)
        self.pitch_filter = LowPassFilter(alpha=0.15)
        
        self.roll_rate_filter = LowPassFilter(alpha=0.1)
        self.pitch_rate_filter = LowPassFilter(alpha=0.1)

        self.max_angle = math.radians(30)
        self.max_velocity = 0.55
        self.deadzone = math.radians(0.3)
        
        self.warmup_time = 2.0
        self.warmup_elapsed = 0.0
        self.control_active = False
        
        self.prev_roll = 0.0
        self.prev_pitch = 0.0
        self.prev_motor_roll = 0.0
        self.prev_motor_pitch = 0.0
        
        self.debug_timer = 0.0
        self.cycle_count = 0
        self.start_time = 0.0
        
        self.adaptive_gain = 1.0
        self.min_error_window = []
        self.max_error_window_size = 100

    def get_barge_orientation(self):
        if self.barge is None:
            return 0.0, 0.0
            
        m = self.barge.getOrientation()
        roll = math.atan2(m[7], m[8])
        pitch = math.atan2(-m[6], math.sqrt(m[7]**2 + m[8]**2))
        return roll, pitch

    def check_motor_limits(self, target_pos, current_pos):
        max_step = self.max_velocity * self.dt
        delta = abs(target_pos - current_pos)
        return delta <= max_step

    def run(self):
        if not self.devices_ok:
            return
            
        for i in range(50):
            self.step(self.timestep)
            
        init_roll, init_pitch = self.get_barge_orientation()
        self.roll_filter.reset(init_roll)
        self.pitch_filter.reset(init_pitch)
        self.prev_roll = init_roll
        self.prev_pitch = init_pitch
        
        self.roll_motor.setPosition(init_roll)
        self.pitch_motor.setPosition(init_pitch)
        self.roll_motor.setVelocity(self.max_velocity)
        self.pitch_motor.setVelocity(self.max_velocity)
        self.prev_motor_roll = init_roll
        self.prev_motor_pitch = init_pitch
        
        self.start_time = self.getTime()

        while self.step(self.timestep) != -1:
            current_time = self.getTime()
            elapsed = current_time - self.start_time
            self.cycle_count += 1
            
            if elapsed < self.warmup_time and not self.control_active:
                self.warmup_elapsed = elapsed
                roll_barge, pitch_barge = self.get_barge_orientation()
                self.roll_filter.update(roll_barge)
                self.pitch_filter.update(pitch_barge)
                
                if elapsed >= self.warmup_time:
                    self.control_active = True
                    self.roll_pid.reset()
                    self.pitch_pid.reset()
                continue
            
            roll_barge, pitch_barge = self.get_barge_orientation()

            filtered_roll = self.roll_filter.update(roll_barge)
            filtered_pitch = self.pitch_filter.update(pitch_barge)

            roll_rate = (filtered_roll - self.prev_roll) / self.dt
            pitch_rate = (filtered_pitch - self.prev_pitch) / self.dt
            
            roll_rate = self.roll_rate_filter.update(roll_rate)
            pitch_rate = self.pitch_rate_filter.update(pitch_rate)
            
            self.prev_roll = filtered_roll
            self.prev_pitch = filtered_pitch

            target_roll = 0.0
            target_pitch = 0.0

            error_roll = target_roll - filtered_roll
            error_pitch = target_pitch - filtered_pitch

            if abs(error_roll) < self.deadzone:
                error_roll = 0.0
            if abs(error_pitch) < self.deadzone:
                error_pitch = 0.0

            error_magnitude = math.sqrt(error_roll**2 + error_pitch**2)
            if error_magnitude > math.radians(5):
                self.adaptive_gain = min(1.5, 1.0 + error_magnitude * 0.1)
            else:
                self.adaptive_gain = max(0.8, 1.0 - error_magnitude * 0.05)

            roll_output = self.roll_pid.update(error_roll, self.dt) * self.adaptive_gain
            pitch_output = self.pitch_pid.update(error_pitch, self.dt) * self.adaptive_gain

            target_roll_pos = filtered_roll + roll_output
            target_pitch_pos = filtered_pitch + pitch_output

            target_roll_pos = max(-self.max_angle, min(self.max_angle, target_roll_pos))
            target_pitch_pos = max(-self.max_angle, min(self.max_angle, target_pitch_pos))
            
            max_step = self.max_velocity * self.dt
            if target_roll_pos - self.prev_motor_roll > max_step:
                target_roll_pos = self.prev_motor_roll + max_step
            elif target_roll_pos - self.prev_motor_roll < -max_step:
                target_roll_pos = self.prev_motor_roll - max_step
                
            if target_pitch_pos - self.prev_motor_pitch > max_step:
                target_pitch_pos = self.prev_motor_pitch + max_step
            elif target_pitch_pos - self.prev_motor_pitch < -max_step:
                target_pitch_pos = self.prev_motor_pitch - max_step

            self.roll_motor.setPosition(target_roll_pos)
            self.pitch_motor.setPosition(target_pitch_pos)
            
            self.prev_motor_roll = target_roll_pos
            self.prev_motor_pitch = target_pitch_pos

            self.min_error_window.append(abs(error_roll))
            if len(self.min_error_window) > self.max_error_window_size:
                self.min_error_window.pop(0)

            self.debug_timer += self.dt
            if self.debug_timer >= 0.5:
                self.debug_timer = 0.0


if __name__ == "__main__":
    controller = StabilizationController()
    if controller.devices_ok:
        controller.run()