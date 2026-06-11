#include <Wire.h>

// --- Motor Driver Pins (L298N) ---
#define IN1 8
#define IN2 9
#define ENA 10

// --- IMU Variables (MPU6050) ---
const int MPU_ADDR = 0x68; // Standard I2C address
float AccX, AccY, AccZ;
float GyroX, GyroY, GyroZ;
float accAngle, gyroRate, currentAngle;
unsigned long previousTime, currentTime;
float elapsedTime;

// --- PID Control Variables ---
// These are the values you will spend 90% of your time tuning!
float Kp = 40.0;  // Proportional: Reacts to current error
float Ki = 0.5;   // Integral: Reacts to accumulated past error
float Kd = 2.0;   // Derivative: Reacts to the rate of change of error
float setPoint = 0.0; // 0 degrees = perfectly upright
float error, previousError, integral, derivative, pidOutput;

void setup() {
  Serial.begin(9600);

  // 1. Initialize Motor Pins
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);

  // 2. Initialize IMU (Wake it up)
  Wire.begin();
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); // Power management register
  Wire.write(0x00); // Write 0 to wake up the sensor
  Wire.endTransmission(true);

  previousTime = millis();
}

void loop() {
  // --- 1. Calculate Loop Timing (dt) ---
  currentTime = millis();
  elapsedTime = (currentTime - previousTime) / 1000.0; // Convert milliseconds to seconds
  previousTime = currentTime;

  // --- 2. Read Raw IMU Data ---
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B); // Start at accelerometer X register
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true); // Request 14 bytes (Acc, Temp, Gyro)

  // Read Accelerometer (Divide by 16384.0 for standard 2g range)
  AccX = (Wire.read() << 8 | Wire.read()) / 16384.0;
  AccY = (Wire.read() << 8 | Wire.read()) / 16384.0;
  AccZ = (Wire.read() << 8 | Wire.read()) / 16384.0;

  Wire.read(); Wire.read(); // Skip Temperature registers

  // Read Gyroscope (Divide by 131.0 for standard 250deg/s range)
  GyroX = (Wire.read() << 8 | Wire.read()) / 131.0;
  GyroY = (Wire.read() << 8 | Wire.read()) / 131.0;
  GyroZ = (Wire.read() << 8 | Wire.read()) / 131.0;

  // --- 3. Sensor Fusion (Complementary Filter) ---
  // Calculates tilt based on gravity vector. (Adjust axes based on how you mount the IMU)
  accAngle = (atan(AccY / sqrt(pow(AccX, 2) + pow(AccZ, 2))) * 180 / PI);
  gyroRate = GyroX; // Rotation rate around X axis causes Y-axis tilt

  // Filter: 98% Gyro (fast but drifts) + 2% Accel (noisy but stable long-term)
  currentAngle = 0.98 * (currentAngle + gyroRate * elapsedTime) + 0.02 * accAngle;

  // --- 4. PID Controller Math ---
  error = setPoint - currentAngle;

  integral += error * elapsedTime;
  integral = constrain(integral, -200, 200); // Anti-windup limit

  derivative = (error - previousError) / elapsedTime;

  // The core PID equation
  pidOutput = (Kp * error) + (Ki * integral) + (Kd * derivative);
  previousError = error;

  // --- 5. Output to Motor ---
  driveMotor(pidOutput);

  // Loop delay for stability (approx 100Hz loop rate)
  delay(10);
}

void driveMotor(float pidValue) {
  // Convert PID output to a safe PWM range (0 to 255)
  int motorPWM = abs(pidValue);
  motorPWM = constrain(motorPWM, 0, 255);

  // Set motor direction based on whether PID output is positive or negative
  if (pidValue < 0) {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
  }
  else if (pidValue > 0) {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
  }
  else {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
  }

  // Send power to the L298N
  analogWrite(ENA, motorPWM);
}
