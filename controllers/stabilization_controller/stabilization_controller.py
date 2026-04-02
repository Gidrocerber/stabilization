from controller import Supervisor
import math


class GeometryParams:
    """Геометрические параметры гидроприводов"""
    L1_PITCH = 0.45
    D10_PITCH = 0.62
    H1_PITCH = math.sqrt(max(0.0, D10_PITCH ** 2 - L1_PITCH ** 2)) - D10_PITCH

    L2_ROLL = 0.45
    D20_ROLL = 0.62
    H2_ROLL = math.sqrt(max(0.0, D20_ROLL ** 2 - L2_ROLL ** 2)) - D20_ROLL

    DELTA_D_MAX = 0.15
    MAX_FORCE = 50000.0
    MAX_VELOCITY = 0.5

    MAX_ROLL_ANGLE = 0.6
    MAX_PITCH_ANGLE = 0.4


class ActuatorKinematics:
    """Кинематика линейного гидропривода: угол <-> ход штока"""

    def __init__(self, l=0.45, h=None, d0=0.62, max_stroke=0.15):
        self.l = l
        self.d0 = d0
        self.h = h if h is not None else (math.sqrt(max(0.0, d0 ** 2 - l ** 2)) - d0)
        self.max_stroke = max_stroke

        h_d0 = self.h + self.d0
        self.h_d0 = h_d0
        self.d_neutral = math.sqrt(max(0.0, h_d0 ** 2 + self.l ** 2))

        if abs(self.d_neutral) > 1e-9:
            self.k_p = (self.l * h_d0) / self.d_neutral
        else:
            self.k_p = 0.0

    def angle_to_stroke_nonlinear(self, angle):
        """
        Точный нелинейный расчёт:
        d^2 = (h+d0)^2 + l^2 - 2*l*(h+d0)*sin(angle)
        stroke = d_current - d_neutral
        """
        term = self.h_d0 ** 2 + self.l ** 2 - 2.0 * self.l * self.h_d0 * math.sin(angle)
        term = max(0.0, term)
        d_current = math.sqrt(term)
        stroke = d_current - self.d_neutral
        return max(-self.max_stroke, min(self.max_stroke, stroke))

    def stroke_to_angle_nonlinear(self, stroke):
        """
        Обратное преобразование:
        sin(angle) = ((h+d0)^2 + l^2 - d^2) / (2*l*(h+d0))
        """
        d_current = self.d_neutral + stroke
        denom = 2.0 * self.l * self.h_d0
        if abs(denom) < 1e-9:
            return 0.0

        s = (self.h_d0 ** 2 + self.l ** 2 - d_current ** 2) / denom
        s = max(-1.0, min(1.0, s))
        return math.asin(s)

    def angle_to_stroke_linear(self, angle):
        stroke = -self.k_p * angle
        return max(-self.max_stroke, min(self.max_stroke, stroke))

    def stroke_to_angle_linear(self, stroke):
        if abs(self.k_p) < 1e-9:
            return 0.0
        return -stroke / self.k_p

    def get_linearization_error_percent(self, angle):
        x_nl = self.angle_to_stroke_nonlinear(angle)
        x_lin = self.angle_to_stroke_linear(angle)
        if abs(x_nl) < 1e-9:
            return 0.0
        return abs((x_nl - x_lin) / x_nl) * 100.0

    def get_stroke_margin(self, stroke):
        return max(0.0, min(1.0, 1.0 - abs(stroke) / self.max_stroke))

    def get_max_compensatable_angle(self):
        if abs(self.k_p) < 1e-9:
            return 0.0
        return self.max_stroke / abs(self.k_p)


class HydraulicActuatorDynamics:
    """Упрощённая динамика гидропривода по ходу штока"""

    def __init__(self, dt, max_stroke=0.15, max_force=50000.0, max_velocity=0.5):
        self.dt = dt
        self.max_stroke = max_stroke
        self.max_force = max_force
        self.max_velocity = max_velocity

        self.stroke = 0.0
        self.stroke_rate = 0.0

        self.tau_rate = 0.05
        self.k_spring = 220000.0
        self.k_damp = 2500.0

    def update(self, target_stroke):
        error = target_stroke - self.stroke

        kp_rate = 8.0
        target_rate = kp_rate * error
        target_rate = max(-self.max_velocity, min(self.max_velocity, target_rate))

        alpha = self.dt / max(self.tau_rate, self.dt)
        self.stroke_rate += alpha * (target_rate - self.stroke_rate)
        self.stroke_rate = max(-self.max_velocity, min(self.max_velocity, self.stroke_rate))

        self.stroke += self.stroke_rate * self.dt
        self.stroke = max(-self.max_stroke, min(self.max_stroke, self.stroke))

        spring_force = self.k_spring * (target_stroke - self.stroke)
        damping_force = -self.k_damp * self.stroke_rate
        actual_force = spring_force + damping_force
        actual_force = max(-self.max_force, min(self.max_force, actual_force))

        return actual_force

    def get_stroke(self):
        return self.stroke

    def reset(self):
        self.stroke = 0.0
        self.stroke_rate = 0.0


class DiscretePID:
    """ПИД-регулятор с антинасыщением и мёртвой зоной"""

    def __init__(
        self,
        kp=1.0,
        ki=0.1,
        kd=0.1,
        dt=0.016,
        max_integral=1.0,
        output_limit=1.0,
        derivative_filter=0.15,
        backlash=0.0,
        conditional_integration=True
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.max_integral = max_integral
        self.output_limit = output_limit
        self.derivative_filter = derivative_filter
        self.backlash = backlash
        self.conditional_integration = conditional_integration

        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0

    def _apply_backlash(self, error):
        if abs(error) <= self.backlash:
            return 0.0
        return error - self.backlash if error > 0.0 else error + self.backlash

    def update(self, error, dt, margin=1.0):
        if dt <= 0.0:
            dt = self.dt

        e = self._apply_backlash(error)

        p_term = self.kp * e

        if self.conditional_integration:
            if abs(e) > 1e-5 and margin > 0.08:
                self.integral += e * dt
        else:
            self.integral += e * dt

        self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
        i_term = self.ki * self.integral

        raw_derivative = (e - self.prev_error) / dt
        self.filtered_derivative = (
            self.derivative_filter * raw_derivative
            + (1.0 - self.derivative_filter) * self.filtered_derivative
        )
        d_term = self.kd * self.filtered_derivative
        self.prev_error = e

        out = p_term + i_term + d_term
        out = max(-self.output_limit, min(self.output_limit, out))
        return out


class StabilizationController(Supervisor):
    def __init__(self):
        super().__init__()

        self.timestep = int(self.getBasicTimeStep())
        self.dt = self.timestep / 1000.0

        self.barge = self.getFromDef("BARGE")
        if self.barge is None:
            print("Warning: BARGE not found. Absolute stabilization disabled fallback -> zeros")

        print("Searching for devices...")

        self.roll_motor = self.getDevice("gangway_roll_motor")
        self.pitch_motor = self.getDevice("gangway_pitch_motor")
        self.roll_sensor = self.getDevice("gangway_roll_sensor")
        self.pitch_sensor = self.getDevice("gangway_pitch_sensor")

        self.roll_actuator = self.getDevice("gangway_roll_actuator")
        self.pitch_actuator = self.getDevice("gangway_pitch_actuator")
        self.roll_actuator_sensor = self.getDevice("gangway_roll_actuator_sensor")
        self.pitch_actuator_sensor = self.getDevice("gangway_pitch_actuator_sensor")

        self.devices_ok = all([
            self.roll_motor, self.pitch_motor,
            self.roll_sensor, self.pitch_sensor,
            self.roll_actuator, self.pitch_actuator,
            self.roll_actuator_sensor, self.pitch_actuator_sensor
        ])

        print(f"  roll_motor: {self.roll_motor}")
        print(f"  pitch_motor: {self.pitch_motor}")
        print(f"  roll_sensor: {self.roll_sensor}")
        print(f"  pitch_sensor: {self.pitch_sensor}")
        print(f"  roll_actuator: {self.roll_actuator}")
        print(f"  pitch_actuator: {self.pitch_actuator}")
        print(f"  roll_actuator_sensor: {self.roll_actuator_sensor}")
        print(f"  pitch_actuator_sensor: {self.pitch_actuator_sensor}")

        if not self.devices_ok:
            print("Error: not all required devices were found")
            return

        self.roll_sensor.enable(self.timestep)
        self.pitch_sensor.enable(self.timestep)
        self.roll_actuator_sensor.enable(self.timestep)
        self.pitch_actuator_sensor.enable(self.timestep)

        self.roll_motor.setVelocity(1.0)
        self.pitch_motor.setVelocity(1.0)

        self.roll_actuator.setVelocity(GeometryParams.MAX_VELOCITY)
        self.pitch_actuator.setVelocity(GeometryParams.MAX_VELOCITY)
        self.roll_actuator.setPosition(0.0)
        self.pitch_actuator.setPosition(0.0)

        self.geom_roll = ActuatorKinematics(
            l=GeometryParams.L2_ROLL,
            h=GeometryParams.H2_ROLL,
            d0=GeometryParams.D20_ROLL,
            max_stroke=GeometryParams.DELTA_D_MAX
        )
        self.geom_pitch = ActuatorKinematics(
            l=GeometryParams.L1_PITCH,
            h=GeometryParams.H1_PITCH,
            d0=GeometryParams.D10_PITCH,
            max_stroke=GeometryParams.DELTA_D_MAX
        )

        self.act_roll = HydraulicActuatorDynamics(
            self.dt,
            max_stroke=GeometryParams.DELTA_D_MAX,
            max_force=GeometryParams.MAX_FORCE,
            max_velocity=GeometryParams.MAX_VELOCITY
        )
        self.act_pitch = HydraulicActuatorDynamics(
            self.dt,
            max_stroke=GeometryParams.DELTA_D_MAX,
            max_force=GeometryParams.MAX_FORCE,
            max_velocity=GeometryParams.MAX_VELOCITY
        )

        backlash_rad = math.radians(0.03)

        self.roll_pid = DiscretePID(
            kp=1.8,
            ki=0.35,
            kd=0.25,
            dt=self.dt,
            max_integral=0.4,
            output_limit=GeometryParams.MAX_ROLL_ANGLE,
            derivative_filter=0.15,
            backlash=backlash_rad,
            conditional_integration=True
        )
        self.pitch_pid = DiscretePID(
            kp=2.2,
            ki=0.4,
            kd=0.3,
            dt=self.dt,
            max_integral=0.4,
            output_limit=GeometryParams.MAX_PITCH_ANGLE,
            derivative_filter=0.15,
            backlash=backlash_rad,
            conditional_integration=True
        )

        self.deadzone = math.radians(0.02)

        self.stroke_warning = False

        self.warmup_time = 2.0
        self.start_time = 0.0
        self.control_active = False

        self.alpha_filter = 0.18
        self.filt_base_roll = 0.0
        self.filt_base_pitch = 0.0

        self.stats_timer = 0.0

    def get_barge_orientation(self):
        """Крен/дифферент баржи"""
        if self.barge is None:
            return 0.0, 0.0

        m = self.barge.getOrientation()
        roll = math.atan2(m[7], m[8])
        pitch = math.atan2(-m[6], math.sqrt(m[7] ** 2 + m[8] ** 2))
        return roll, pitch

    def get_gangway_angles(self):
        """Углы трапа по шарнирам"""
        roll = self.roll_sensor.getValue()
        pitch = self.pitch_sensor.getValue()
        return roll, pitch

    def _check_stroke_margin(self, stroke_roll, stroke_pitch):
        roll_margin = self.geom_roll.get_stroke_margin(stroke_roll)
        pitch_margin = self.geom_pitch.get_stroke_margin(stroke_pitch)

        if (roll_margin < 0.2 or pitch_margin < 0.2) and not self.stroke_warning:
            print(
                f"WARNING: low stroke margin | roll={roll_margin:.2f}, pitch={pitch_margin:.2f}"
            )
            self.stroke_warning = True
        elif roll_margin >= 0.2 and pitch_margin >= 0.2:
            self.stroke_warning = False

        return roll_margin, pitch_margin

    def run(self):
        if not self.devices_ok:
            print("Controller cannot run: devices not initialized")
            return

        print("Warming up simulation...")
        for _ in range(30):
            if self.step(self.timestep) == -1:
                return

        rb, pb = self.get_barge_orientation()
        self.filt_base_roll = rb
        self.filt_base_pitch = pb

        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.act_roll.reset()
        self.act_pitch.reset()

        self.roll_motor.setPosition(0.0)
        self.pitch_motor.setPosition(0.0)
        self.roll_actuator.setPosition(0.0)
        self.pitch_actuator.setPosition(0.0)

        self.start_time = self.getTime()
        print(f"Controller initialized at t={self.start_time:.2f}s")

        while self.step(self.timestep) != -1:
            current_time = self.getTime()
            elapsed = current_time - self.start_time

            if not self.control_active:
                if elapsed >= self.warmup_time:
                    self.control_active = True
                    print("Stabilization control ACTIVATED")
                else:
                    continue

            base_roll, base_pitch = self.get_barge_orientation()
            self.filt_base_roll = (
                self.alpha_filter * base_roll + (1.0 - self.alpha_filter) * self.filt_base_roll
            )
            self.filt_base_pitch = (
                self.alpha_filter * base_pitch + (1.0 - self.alpha_filter) * self.filt_base_pitch
            )

            plat_roll, plat_pitch = self.get_gangway_angles()

            abs_roll = plat_roll + self.filt_base_roll
            abs_pitch = plat_pitch + self.filt_base_pitch

            error_roll = -abs_roll
            error_pitch = -abs_pitch

            if abs(error_roll) < self.deadzone:
                error_roll = 0.0
            if abs(error_pitch) < self.deadzone:
                error_pitch = 0.0

            estimated_stroke_roll = self.geom_roll.angle_to_stroke_nonlinear(plat_roll)
            estimated_stroke_pitch = self.geom_pitch.angle_to_stroke_nonlinear(plat_pitch)
            roll_margin, pitch_margin = self._check_stroke_margin(
                estimated_stroke_roll, estimated_stroke_pitch
            )

            target_roll_angle = self.roll_pid.update(error_roll, self.dt, margin=roll_margin)
            target_pitch_angle = self.pitch_pid.update(error_pitch, self.dt, margin=pitch_margin)

            target_roll_angle = max(
                -GeometryParams.MAX_ROLL_ANGLE,
                min(GeometryParams.MAX_ROLL_ANGLE, target_roll_angle)
            )
            target_pitch_angle = max(
                -1.0,
                min(GeometryParams.MAX_PITCH_ANGLE, target_pitch_angle)
            )

            target_stroke_roll = self.geom_roll.angle_to_stroke_nonlinear(target_roll_angle)
            target_stroke_pitch = self.geom_pitch.angle_to_stroke_nonlinear(target_pitch_angle)

            actual_force_roll = self.act_roll.update(target_stroke_roll)
            actual_force_pitch = self.act_pitch.update(target_stroke_pitch)

            actual_stroke_roll = self.act_roll.get_stroke()
            actual_stroke_pitch = self.act_pitch.get_stroke()

            commanded_roll_angle = self.geom_roll.stroke_to_angle_nonlinear(actual_stroke_roll)
            commanded_pitch_angle = self.geom_pitch.stroke_to_angle_nonlinear(actual_stroke_pitch)

            commanded_roll_angle = max(
                -GeometryParams.MAX_ROLL_ANGLE,
                min(GeometryParams.MAX_ROLL_ANGLE, commanded_roll_angle)
            )
            commanded_pitch_angle = max(
                -1.0,
                min(GeometryParams.MAX_PITCH_ANGLE, commanded_pitch_angle)
            )

            self.roll_actuator.setPosition(actual_stroke_roll)
            self.pitch_actuator.setPosition(actual_stroke_pitch)

            self.roll_motor.setPosition(commanded_roll_angle)
            self.pitch_motor.setPosition(commanded_pitch_angle)

            self.stats_timer += self.dt
            if self.stats_timer >= 1.0:
                self.stats_timer = 0.0

                lin_err_roll = self.geom_roll.get_linearization_error_percent(plat_roll)
                lin_err_pitch = self.geom_pitch.get_linearization_error_percent(plat_pitch)

                print(
                    f"t={current_time:5.1f}s | "
                    f"Abs: R={math.degrees(abs_roll):6.2f}° P={math.degrees(abs_pitch):6.2f}° | "
                    f"CmdAng: R={math.degrees(commanded_roll_angle):6.2f}° "
                    f"P={math.degrees(commanded_pitch_angle):6.2f}° | "
                    f"Stroke: R={actual_stroke_roll*1000:6.1f}mm "
                    f"P={actual_stroke_pitch*1000:6.1f}mm | "
                    f"Force: R={actual_force_roll/1000:6.2f}kN "
                    f"P={actual_force_pitch/1000:6.2f}kN | "
                    f"Margin: R={roll_margin:.2f} P={pitch_margin:.2f} | "
                    f"LinErr: R={lin_err_roll:5.1f}% P={lin_err_pitch:5.1f}%"
                )


if __name__ == "__main__":
    controller = StabilizationController()
    if hasattr(controller, "devices_ok") and controller.devices_ok:
        controller.run()
    else:
        print("Controller initialization failed")
