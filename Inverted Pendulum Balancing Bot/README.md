# Inverted Pendulum Balancer

A directly-actuated inverted pendulum mounted on a cart. An MPU6050 IMU rides on
the pendulum and measures its tilt from vertical. A DC motor at the pendulum's
pivot, driven through an L298N, applies torque to rotate the pendulum back toward
upright whenever it deviates. A PID controller decides how much torque to apply.

Unlike a classic cart-pole (where the motor drives the cart and the pendulum
swings freely, so you balance it indirectly), here the motor acts directly on the
pendulum angle you are stabilizing. The control input is collocated with the
variable being controlled, which makes the balancing loop more direct. Note that
if the cart is free to roll, the cart's position is itself an unactuated degree of
freedom, so the full cart-plus-pendulum system is still underactuated (see the
cart-coupling note below).

## How it works

The control loop runs at roughly 100 Hz and does five things on every pass:

1. **Measure loop time** (`dt`) using `millis()` so the integral and derivative
   terms are time-correct regardless of loop jitter.
2. **Read the IMU** over I2C - 14 bytes covering the accelerometer, temperature,
   and gyroscope registers in a single burst.
3. **Estimate the pendulum's tilt** with a complementary filter:
   `angle = 0.98 * (angle + gyroRate * dt) + 0.02 * accAngle`. The gyro term is
   fast but drifts; the accelerometer term is noisy but stable long-term, so the
   filter leans on the gyro for responsiveness and lets the accelerometer correct
   the slow drift.
4. **Run PID** against a setpoint of 0 degrees (vertical):
   `output = Kp*error + Ki*integral + Kd*derivative`, with an anti-windup clamp on
   the integral term.
5. **Apply corrective torque** - the sign of the PID output picks the rotation
   direction, and its magnitude (clamped to 0-255) becomes the PWM duty cycle sent
   to the L298N, which torques the pendulum back toward vertical.

## Hardware

| Component        | Notes                                              |
|------------------|----------------------------------------------------|
| Arduino board    | Uno / Nano (any AVR board with PWM on pin 10)      |
| MPU6050          | 6-axis IMU, I2C address `0x68`, mounted on pendulum |
| L298N            | H-bridge motor driver                              |
| DC motor         | At the pendulum pivot, applies the balancing torque |
| Pendulum + pivot | Rigid rod on a low-friction bearing                 |
| Cart             | Carries the pivot, motor, and electronics           |
| Battery pack     | Sized for the motor (e.g. 2S/3S LiPo or 6xAA)       |

## Wiring

**MPU6050 (on the pendulum) -> Arduino**

| MPU6050 | Arduino       |
|---------|---------------|
| VCC     | 5V (or 3.3V)  |
| GND     | GND           |
| SDA     | A4 (SDA)      |
| SCL     | A5 (SCL)      |

**L298N -> Arduino**

| L298N | Arduino   |
|-------|-----------|
| IN1   | D8        |
| IN2   | D9        |
| ENA   | D10 (PWM) |

Power the L298N from your battery pack (not the Arduino's 5V), and share a common
ground between the Arduino, the L298N, and the battery.

## Build & upload

1. Install the [Arduino IDE](https://www.arduino.cc/en/software).
2. The only library used is `Wire`, which ships with the IDE - no extra installs.
3. Open `InvertedPendulum/InvertedPendulum.ino`.
4. Select your board and port under **Tools**.
5. Click **Upload**.

## Tuning the PID

The three gains at the top of the sketch are where most of the work goes. A common
approach:

- Start with `Ki = 0` and `Kd = 0`. Raise `Kp` until the motor reacts to tilt and
  the pendulum oscillates around vertical.
- Add `Kd` to damp the oscillation - it reacts to how fast the angle is changing.
- Add a small `Ki` last to remove any steady lean (steady-state error).

The current starting values are `Kp = 40.0`, `Ki = 0.5`, `Kd = 2.0`. These are a
starting point, not a final answer - they depend on the pendulum's mass, length,
pivot friction, and motor torque.

The sketch opens the serial port at 9600 baud (`Serial.begin(9600)`) but does not
currently print anything, so the Serial Monitor stays empty as-is. To watch the
angle while tuning, add a line such as `Serial.println(currentAngle);` inside the
loop (for example just after the complementary filter), then open the Serial
Monitor at 9600 baud.

> **Mounting matters:** the filter assumes a specific IMU orientation. The tilt is
> computed from `AccY` and the gyro rate is taken from `GyroX`. If the IMU is
> mounted differently on the pendulum, swap the axes in the sensor-fusion section
> accordingly.
