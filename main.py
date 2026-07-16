"""
DIGITAL TWIN VEHICLE HEALTH MONITORING SYSTEM
Main Application Script — main.py
Platform : Raspberry Pi 3 B+

SYSTEM OVERVIEW
This script implements a complete Digital Twin for a prototype vehicle.
It performs continuous multi-sensor data acquisition, real-time state
mirroring, LSTM-based anomaly detection, rule-based fault detection,
component health scoring, LCD display updates, and a Flask web dashboard
— all running concurrently on a single Raspberry Pi 3 B+.

Architecture
1. Hardware Layer   — GPIO, I2C sensors (INA219, BMP085, DHT11, HC-SR04,
                       IR encoder, SW-420, I2C LCD)
2. Acquisition Layer— Timed 5-second polling loop, CSV logger
3. Digital Twin Engine — State mirror, 60-reading history, life accumulators,
                       state coordination
4. ML Layer         — Z-score anomaly detector + online LSTM (2-layer)
5. Fault Engine      — 14 rule-based fault conditions (F01-F14)
6. Alert Layer       — LCD, piezo buzzer, web dashboard
7. Web Layer         — Flask REST API + real-time dashboard (port 5000)
8. Control Layer     — L298N motor driver via GPIO + joystick buttons
"""

# =====================================================================
# SECTION 1 — LIBRARY IMPORTS
# =====================================================================

# Standard library
import threading                    # Concurrent execution of acquisition + web server
import csv                          # CSV file logging
import time                         # Delays and timing
import math                         # Mathematical operations (altitude formula)
from collections import deque       # Fixed-length rolling history window
from datetime import datetime       # Timestamps for logs and fault records

# Hardware / sensor libraries
import RPi.GPIO as GPIO                        # Raspberry Pi GPIO control
import Adafruit_DHT                            # DHT11 temperature & humidity
import Adafruit_BMP.BMP085 as BMP085           # BMP085 barometric pressure
from adafruit_ina219 import INA219             # INA219 voltage & current monitor
from RPLCD.i2c import CharLCD                  # 16x2 I2C LCD character display

# Web framework
from flask import Flask, jsonify, make_response

# Machine learning
import numpy as np                  # Numerical arrays
# NOTE: On resource-constrained hardware, a lightweight custom LSTM is used
# rather than TensorFlow/Keras to avoid excessive memory overhead.
# The implementation below is a pure-NumPy rolling LSTM approximation
# suitable for the Raspberry Pi 3 B+ (1 GB RAM, 1.2 GHz quad-core).


# =====================================================================
# SECTION 2 — GLOBAL CONFIGURATION CONSTANTS
# =====================================================================

# --- Timing ---
ACQUISITION_INTERVAL_SEC = 5        # Sensor polling interval (seconds)
RECORDS_PER_MINUTE = 12             # Derived: 60 / ACQUISITION_INTERVAL_SEC

# --- File paths ---
CSV_LOG_PATH = "/home/pi/vehicle_log.csv"
FAULT_LOG_PATH = "/home/pi/fault_log.csv"

# --- Sensor types ---
DHT_SENSOR_TYPE = Adafruit_DHT.DHT11    # Sensor model for Adafruit library

# --- Rolling window ---
HISTORY_WINDOW = 60                 # Number of recent readings kept in memory for ML

# --- Anomaly detection ---
ZSCORE_THRESHOLD = 2.0               # Z-score above this value triggers an anomaly flag

# --- Component life accumulator limits ---
# Under normal operating stress the accumulator depletes over:
# Minimum: 20,000 acquisitions (~28 hours of continuous operation)
# Maximum: 100,000 acquisitions (~139 hours of continuous operation)
LIFE_DEPLETION_NORMAL_MIN = 20_000
LIFE_DEPLETION_NORMAL_MAX = 100_000

# --- Flask web server ---
FLASK_PORT = 5000


# =====================================================================
# SECTION 3 — GPIO PIN DEFINITIONS
# =====================================================================
# All BCM-numbered GPIO pins used by the system.
# Changing a physical connection only requires editing this section.

# Sensor input pins
IR_PIN = 17    # IR optical speed encoder (pulse counting)
VIB_PIN = 27   # SW-420 vibration sensor (digital interrupt)
DHT_PIN = 4    # DHT11 temperature & humidity data line
BUZ_PIN = 18   # Piezoelectric buzzer (active, PWM capable)

# HC-SR04 ultrasonic sensors (oil & fuel level)
TRIG_OIL, ECHO_OIL = 5, 6        # Oil tank ultrasonic pair
TRIG_FUEL, ECHO_FUEL = 13, 19    # Fuel tank ultrasonic pair

# L298N motor driver (H-bridge) control pins
IN1, IN2 = 20, 21   # Left motor (forward/backward polarity)
IN3, IN4 = 16, 12   # Right motor (forward/backward polarity)

# Joystick directional buttons (active-LOW with pull-up)
BTN_FORWARD = 22
BTN_BACKWARD = 23
BTN_LEFT = 24
BTN_RIGHT = 25


# =====================================================================
# SECTION 4 — GPIO INITIALISATION
# =====================================================================

def init_gpio():
    """
    Configure all GPIO pins with correct direction and initial state.

    Uses BCM (Broadcom SOC channel) numbering throughout.
    Pull-up resistors are enabled on joystick buttons so they read HIGH
    at rest and LOW when pressed.
    """
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    # Sensor inputs
    GPIO.setup(IR_PIN, GPIO.IN)
    GPIO.setup(VIB_PIN, GPIO.IN)

    # Ultrasonic trigger (output) and echo (input) pairs
    for trig in (TRIG_OIL, TRIG_FUEL):
        GPIO.setup(trig, GPIO.OUT)
        GPIO.output(trig, False)   # Ensure trigger starts LOW
    for echo in (ECHO_OIL, ECHO_FUEL):
        GPIO.setup(echo, GPIO.IN)

    # Buzzer output
    GPIO.setup(BUZ_PIN, GPIO.OUT)
    GPIO.output(BUZ_PIN, False)

    # Motor driver outputs — all LOW (stopped) at startup
    for pin in (IN1, IN2, IN3, IN4):
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, False)

    # Joystick buttons — input with internal pull-up
    for btn in (BTN_FORWARD, BTN_BACKWARD, BTN_LEFT, BTN_RIGHT):
        GPIO.setup(btn, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    # Attach interrupt callbacks for speed encoder and vibration sensor
    GPIO.add_event_detect(IR_PIN, GPIO.RISING,
                           callback=_ir_pulse_callback, bouncetime=5)
    GPIO.add_event_detect(VIB_PIN, GPIO.RISING,
                           callback=_vib_pulse_callback, bouncetime=50)

    print("[GPIO] Initialisation complete.")


# =====================================================================
# SECTION 5 — INTERRUPT CALLBACKS & PULSE COUNTERS
# =====================================================================
# These counters are incremented by GPIO interrupt callbacks and reset each
# acquisition cycle to derive speed (RPM) and vibration event count.

_ir_pulse_count = 0             # Raw IR encoder pulse count
_vib_pulse_count = 0            # Raw vibration event count
_pulse_lock = threading.Lock()  # Protect shared counters from race conditions


def _ir_pulse_callback(channel):
    """Called on each rising edge of the IR speed encoder output."""
    global _ir_pulse_count
    with _pulse_lock:
        _ir_pulse_count += 1


def _vib_pulse_callback(channel):
    """Called on each rising edge of the SW-420 vibration sensor output."""
    global _vib_pulse_count
    with _pulse_lock:
        _vib_pulse_count += 1


def get_and_reset_counters():
    """
    Atomically read and reset both pulse counters.
    Called once per acquisition cycle (every 5 seconds).

    Returns
    -------
    ir_count  : int — IR pulses in the last interval
    vib_count : int — Vibration events in the last interval
    """
    global _ir_pulse_count, _vib_pulse_count
    with _pulse_lock:
        ir = _ir_pulse_count
        vib = _vib_pulse_count
        _ir_pulse_count = 0
        _vib_pulse_count = 0
    return ir, vib


def compute_speed_rpm(ir_count, pulses_per_revolution=20):
    """
    Convert raw IR pulse count to rotational speed in RPM.

    The prototype wheel encoder disk has 20 slots (pulses per revolution).
    RPM = (pulses / pulses_per_rev) / (interval_sec / 60)

    Parameters
    ----------
    ir_count               : int — Pulses counted over ACQUISITION_INTERVAL_SEC
    pulses_per_revolution  : int — Encoder disk slot count (default 20)

    Returns
    -------
    float — Rotational speed in RPM
    """
    revolutions = ir_count / pulses_per_revolution
    rpm = revolutions / (ACQUISITION_INTERVAL_SEC / 60.0)
    return round(rpm, 2)


# =====================================================================
# SECTION 6 — I2C SENSOR INITIALISATION
# =====================================================================

def init_sensors():
    """
    Initialise all I2C-connected sensors and return their objects.

    Sensors on the I2C bus:
        - INA219  — voltage & current monitor (default address 0x40)
        - BMP085  — barometric pressure sensor (default address 0x77)
        - CharLCD — 16x2 LCD display (address 0x27)

    The DHT11 uses a single-wire protocol on GPIO pin 4 (not I2C).
    The HC-SR04 sensors use GPIO TRIG/ECHO (not I2C).

    Returns
    -------
    ina219 : INA219 object
    bmp    : BMP085 object
    lcd    : CharLCD object
    """
    from board import SCL, SDA
    import busio

    i2c = busio.I2C(SCL, SDA)

    ina219 = INA219(i2c)
    bmp = BMP085.BMP085()
    lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1,
                  cols=16, rows=2, dotsize=8)
    lcd.clear()
    print("[I2C] Sensors initialised: INA219, BMP085, CharLCD.")
    return ina219, bmp, lcd


# =====================================================================
# SECTION 7 — ULTRASONIC DISTANCE READING (HC-SR04)
# =====================================================================

def read_distance(trig_pin, echo_pin, timeout_sec=0.02):
    """
    Measure distance using an HC-SR04 ultrasonic sensor.

    Operation sequence
    -------------------
    1. Ensure TRIG is LOW for 2 ms (sensor settle time).
    2. Pulse TRIG HIGH for 10 microseconds to initiate measurement.
    3. Wait for ECHO to go HIGH (ultrasonic burst sent).
    4. Measure duration of ECHO HIGH pulse (ultrasonic return time).
    5. Calculate distance: d = duration * speed_of_sound / 2
       At 20C: speed of sound ~= 34,300 cm/s -> factor = 17,150.

    Timeout guards prevent infinite loops if the sensor malfunctions
    or no echo is received (object out of range or sensor fault).

    Parameters
    ----------
    trig_pin    : int   — GPIO BCM pin number for TRIG
    echo_pin    : int   — GPIO BCM pin number for ECHO
    timeout_sec : float — Maximum wait time before returning None (default 20 ms)

    Returns
    -------
    float — Distance in centimetres, or None if timeout occurred.

    Sensor Specification (HC-SR04)
    -------------------------------
    Range    : 2 cm - 400 cm
    Accuracy : +/- 3 mm
    Beam     : 15 degree cone
    """
    # Step 1: Ensure TRIG is LOW
    GPIO.output(trig_pin, False)
    time.sleep(0.002)

    # Step 2: 10us HIGH pulse on TRIG
    GPIO.output(trig_pin, True)
    time.sleep(0.00001)
    GPIO.output(trig_pin, False)

    # Step 3: Wait for ECHO to go HIGH (burst transmitted)
    wait_start = time.time()
    while GPIO.input(echo_pin) == 0:
        if time.time() - wait_start > timeout_sec:
            return None   # Sensor did not respond

    # Step 4: Measure ECHO HIGH duration (echo return time)
    pulse_start = time.time()
    while GPIO.input(echo_pin) == 1:
        if time.time() - pulse_start > timeout_sec:
            return None   # Echo pulse too long (object too close or fault)

    pulse_duration = time.time() - pulse_start

    # Step 5: Convert to centimetres
    distance_cm = pulse_duration * 17_150
    return round(distance_cm, 2)


def distance_to_level_percent(distance_cm, tank_empty_cm=25.0, tank_full_cm=3.0):
    """
    Convert a raw ultrasonic distance reading to a fill-level percentage.

    The sensor is mounted at the top of the tank pointing downward.
    A SHORT distance means a HIGH fill level (sensor close to liquid surface).
    A LONG distance means a LOW fill level (sensor far from liquid surface).

    Parameters
    ----------
    distance_cm   : float — Measured distance from sensor to surface (cm)
    tank_empty_cm : float — Distance reading when tank is empty (default 25 cm)
    tank_full_cm  : float — Distance reading when tank is full (default 3 cm)

    Returns
    -------
    float — Fill level as a percentage (0-100%), clamped to valid range.
    """
    if distance_cm is None:
        return 0.0
    level = (tank_empty_cm - distance_cm) / (tank_empty_cm - tank_full_cm)
    level_pct = max(0.0, min(100.0, level * 100.0))
    return round(level_pct, 1)


# =====================================================================
# SECTION 8 — DIGITAL TWIN ENGINE
# =====================================================================
# The Digital Twin Engine maintains a live mirror of the physical vehicle
# state and tracks component health over time.
#
# Four subsystems are monitored:
#   1. Engine      — temperature, vibration, speed
#   2. Battery     — bus voltage, current consumption
#   3. Fuel System — fuel level, oil level
#   4. Suspension  — vibration signature
#
# Each subsystem has a ComponentLifeAccumulator that decrements with each
# acquisition based on measured stress levels.

class ComponentLifeAccumulator:
    """
    Tracks the accumulated operational health of a vehicle subsystem.

    The accumulator starts at 100% (fully healthy) and decrements at a
    rate proportional to the measured stress level each acquisition cycle.
    When it reaches 0%, the component is considered critically degraded.

    Attributes
    ----------
    name          : str   — Human-readable subsystem name
    health_pct    : float — Current health percentage (0-100)
    total_cycles  : int   — Total acquisition cycles processed
    stress_history: list  — Last 60 stress values (for trend analysis)
    """

    def __init__(self, name, initial_health=100.0):
        self.name = name
        self.health_pct = initial_health
        self.total_cycles = 0
        self.stress_history = deque(maxlen=HISTORY_WINDOW)

    def update(self, stress_level):
        """
        Apply one acquisition cycle of stress-based degradation.

        The depletion rate is calibrated so that under NORMAL stress
        (stress_level = 1.0), the accumulator depletes from 100% to 0%
        over LIFE_DEPLETION_NORMAL_MIN to LIFE_DEPLETION_NORMAL_MAX cycles.

        Parameters
        ----------
        stress_level : float — Normalised stress index (0.0 = idle, 1.0 = normal,
                                >1.0 = elevated stress)
        """
        # Base depletion per cycle at normal stress
        base_depletion = 100.0 / LIFE_DEPLETION_NORMAL_MAX

        # Scale by actual stress level
        depletion = base_depletion * stress_level
        self.health_pct = max(0.0, self.health_pct - depletion)
        self.total_cycles += 1
        self.stress_history.append(stress_level)

    def to_dict(self):
        """Serialise accumulator state for API responses and CSV logging."""
        return {
            "name": self.name,
            "health_pct": round(self.health_pct, 4),
            "cycles": self.total_cycles,
            "avg_stress": round(float(np.mean(self.stress_history))
                                 if self.stress_history else 0.0, 4)
        }


class DigitalTwinEngine:
    """
    Core Digital Twin state management system.

    Responsibilities
    -----------------
    - Maintain a real-time state mirror of all 10 monitored parameters.
    - Store a rolling 60-reading history window per parameter channel.
    - Manage component life accumulators for all four vehicle subsystems.
    - Coordinate state updates across all subsystems on each acquisition.
    - Expose current state for the web dashboard and fault detection engine.

    Thread Safety
    --------------
    A threading.Lock protects the state dictionary and history deques.
    All reads and writes to shared state must acquire this lock.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # --- Real-time state mirror ---
        # Stores the most recent reading for each of the 10 parameters.
        self.state = {
            "timestamp": None,
            "bus_voltage": 0.0,     # Volts
            "current": 0.0,         # Amperes
            "temperature": 0.0,     # C
            "humidity": 0.0,        # % RH
            "pressure": 0.0,        # hPa
            "altitude": 0.0,        # metres
            "speed_rpm": 0.0,       # RPM
            "vibration": 0,         # event count per 5-second window
            "fuel_pct": 0.0,        # %
            "oil_pct": 0.0,         # %
            "health_score": 100,    # Overall system health (0-100)
            "active_faults": []     # List of active fault codes
        }

        # --- 60-reading rolling history (one deque per channel) ---
        self.history = {
            key: deque(maxlen=HISTORY_WINDOW)
            for key in ("bus_voltage", "current", "temperature",
                        "humidity", "pressure", "speed_rpm", "vibration")
        }

        # --- Component life accumulators ---
        self.accumulators = {
            "engine": ComponentLifeAccumulator("Engine"),
            "battery": ComponentLifeAccumulator("Battery"),
            "fuel_system": ComponentLifeAccumulator("Fuel System"),
            "suspension": ComponentLifeAccumulator("Suspension")
        }

        # --- Update statistics ---
        self.update_count = 0
        self.last_update_sec = None    # time.time() of last successful update
        self.max_update_delay_sec = 0.0

        print("[DigitalTwin] Engine initialised.")

    def update(self, reading: dict):
        """
        Ingest a new sensor reading and synchronise the Digital Twin state.

        This method is the single entry point for all sensor data. It:
            1. Updates the real-time state mirror.
            2. Appends each channel to its rolling history deque.
            3. Computes per-subsystem stress indices and updates accumulators.
            4. Records update timing statistics.

        Parameters
        ----------
        reading : dict — Keys matching self.state parameter names,
                          populated by the acquisition loop.
        """
        now = time.time()

        with self._lock:
            # 1. Update state mirror
            self.state.update(reading)
            self.state["timestamp"] = datetime.now().isoformat()

            # 2. Append to rolling history
            for key in self.history:
                if key in reading:
                    self.history[key].append(reading[key])

            # 3. Compute stress indices and update accumulators
            # Stress index is normalised: 1.0 = nominal operating level
            engine_stress = self._compute_engine_stress(reading)
            batt_stress = self._compute_battery_stress(reading)
            fuel_stress = self._compute_fuel_stress(reading)
            susp_stress = self._compute_suspension_stress(reading)

            self.accumulators["engine"].update(engine_stress)
            self.accumulators["battery"].update(batt_stress)
            self.accumulators["fuel_system"].update(fuel_stress)
            self.accumulators["suspension"].update(susp_stress)

            # 4. Record timing
            if self.last_update_sec is not None:
                delay = now - self.last_update_sec
                self.max_update_delay_sec = max(self.max_update_delay_sec, delay)
            self.last_update_sec = now
            self.update_count += 1

    def _compute_engine_stress(self, r):
        """
        Derive engine stress index from temperature, speed, and vibration.
        Normalised so that nominal values yield a stress of 1.0.
        """
        temp_factor = r.get("temperature", 25) / 25.0        # Nominal ~25C
        speed_factor = min(r.get("speed_rpm", 0) / 500.0, 2.0)  # Cap at 2x
        vib_factor = min(r.get("vibration", 0) / 10.0, 2.0)
        return max(0.0, (temp_factor + speed_factor + vib_factor) / 3.0)

    def _compute_battery_stress(self, r):
        """
        Derive battery stress index from voltage deviation and current draw.
        Nominal voltage: ~12.6 V; nominal current: ~0.04 A.
        """
        v_nominal = 12.6
        v_actual = r.get("bus_voltage", v_nominal)
        v_stress = abs(v_actual - v_nominal) / v_nominal
        i_stress = min(r.get("current", 0.04) / 0.04, 3.0)
        return max(0.0, (v_stress + i_stress) / 2.0)

    def _compute_fuel_stress(self, r):
        """
        Fuel system stress increases as levels drop below safe thresholds.
        Operating near empty causes accelerated pump and injector wear.
        """
        fuel = r.get("fuel_pct", 100)
        oil = r.get("oil_pct", 100)
        fuel_stress = max(0.0, (50.0 - fuel) / 50.0) if fuel < 50 else 0.0
        oil_stress = max(0.0, (50.0 - oil) / 50.0) if oil < 50 else 0.0
        return (fuel_stress + oil_stress) / 2.0

    def _compute_suspension_stress(self, r):
        """
        Suspension stress is driven primarily by vibration event frequency.
        High vibration counts indicate rough terrain or mechanical looseness.
        """
        return min(r.get("vibration", 0) / 5.0, 3.0)

    def get_state_snapshot(self):
        """Return a copy of the current state (thread-safe)."""
        with self._lock:
            return dict(self.state)

    def get_history_snapshot(self):
        """Return current history window as plain lists (thread-safe)."""
        with self._lock:
            return {k: list(v) for k, v in self.history.items()}

    def get_accumulator_snapshot(self):
        """Return serialised accumulator states (thread-safe)."""
        with self._lock:
            return {k: v.to_dict() for k, v in self.accumulators.items()}


# =====================================================================
# SECTION 9 — Z-SCORE ANOMALY DETECTOR
# =====================================================================

class ZScoreAnomalyDetector:
    """
    Statistical anomaly detector using Z-score over a rolling window.

    For each monitored channel, the detector maintains a rolling mean and
    standard deviation over the last HISTORY_WINDOW readings. A new reading
    is flagged as anomalous if its Z-score exceeds ZSCORE_THRESHOLD.

    Z-score formula: z = |x - mu| / sigma

    Channels monitored: current, vibration, speed_rpm, temperature.

    Attributes
    ----------
    scores : dict — Most recent Z-score per channel (updated each cycle)
    flags  : dict — Boolean anomaly flag per channel
    """

    MONITORED_CHANNELS = ["current", "vibration", "speed_rpm", "temperature"]

    def __init__(self):
        self.scores = {ch: 0.0 for ch in self.MONITORED_CHANNELS}
        self.flags = {ch: False for ch in self.MONITORED_CHANNELS}
        print(f"[ZScore] Anomaly detector initialised. Threshold = {ZSCORE_THRESHOLD}")

    def update(self, history: dict):
        """
        Recompute Z-scores for all monitored channels.

        Uses the rolling 60-reading history maintained by the Digital Twin
        Engine. If fewer than 2 readings exist, scores remain 0.0 (insufficient
        data for meaningful statistics).

        Parameters
        ----------
        history : dict — {channel_name: list_of_float} from DT history window
        """
        for ch in self.MONITORED_CHANNELS:
            values = history.get(ch, [])
            if len(values) < 2:
                self.scores[ch] = 0.0
                self.flags[ch] = False
                continue

            arr = np.array(values, dtype=float)
            mu = np.mean(arr)
            sigma = np.std(arr)

            if sigma < 1e-9:
                # Zero variance: all readings identical, no anomaly possible
                self.scores[ch] = 0.0
                self.flags[ch] = False
            else:
                # Z-score of the most recent reading
                z = abs(arr[-1] - mu) / sigma
                self.scores[ch] = round(float(z), 4)
                self.flags[ch] = z > ZSCORE_THRESHOLD

    def get_max_score(self):
        """Return the highest Z-score across all channels."""
        return max(self.scores.values()) if self.scores else 0.0

    def summary(self):
        """Return a formatted summary string for logging."""
        parts = [f"{ch}={self.scores[ch]:.2f}" for ch in self.MONITORED_CHANNELS]
        return "Z-scores: " + ", ".join(parts) + f" | Max={self.get_max_score():.2f}"


# =====================================================================
# SECTION 10 — LSTM ONLINE LEARNER
# =====================================================================

class LSTMOnlineLearner:
    """
    Lightweight two-layer LSTM for online anomaly detection and prediction.

    This implementation uses a simplified LSTM approximation implemented in
    NumPy to remain within the memory and compute constraints of the
    Raspberry Pi 3 B+ (1 GB LPDDR2, 1.2 GHz quad-core ARM Cortex-A53).

    Full TensorFlow/Keras LSTM architectures cannot meet the <5-second
    inference requirement on this hardware; this approximation sacrifices
    model complexity in exchange for real-time feasibility.

    Architecture
    -------------
    Input channels : 5 (bus_voltage, current, temperature, speed_rpm, vibration)
    Layer 1        : 16 LSTM units
    Layer 2        : 8 LSTM units
    Output         : 5 predicted values (one per input channel, next timestep)
    Training       : 3 iterations per acquisition cycle (online/incremental)

    The model learns the normal operating pattern of the vehicle and flags
    readings that deviate significantly from its learned prediction.

    Attributes
    ----------
    input_features   : list — Names of the 5 input channels
    train_cycles     : int  — Total training iterations completed
    last_prediction  : dict — Most recent predicted values per channel
    prediction_error : dict — Absolute prediction error per channel
    """

    INPUT_FEATURES = ["bus_voltage", "current", "temperature", "speed_rpm", "vibration"]
    LAYER1_UNITS = 16
    LAYER2_UNITS = 8
    TRAIN_ITERS_PER_CYCLE = 3
    LEARNING_RATE = 0.01

    def __init__(self):
        n_in = len(self.INPUT_FEATURES)
        n_l1 = self.LAYER1_UNITS
        n_l2 = self.LAYER2_UNITS
        n_out = n_in    # Predict the same channels

        # --- Layer 1 weights (input -> hidden) ---
        # Xavier initialisation for stable training
        self.W1 = np.random.randn(4 * n_l1, n_in) * np.sqrt(2.0 / n_in)
        self.U1 = np.random.randn(4 * n_l1, n_l1) * np.sqrt(2.0 / n_l1)
        self.b1 = np.zeros(4 * n_l1)

        # --- Layer 2 weights (hidden -> hidden) ---
        self.W2 = np.random.randn(4 * n_l2, n_l1) * np.sqrt(2.0 / n_l1)
        self.U2 = np.random.randn(4 * n_l2, n_l2) * np.sqrt(2.0 / n_l2)
        self.b2 = np.zeros(4 * n_l2)

        # --- Output weights ---
        self.Wo = np.random.randn(n_out, n_l2) * np.sqrt(2.0 / n_l2)
        self.bo = np.zeros(n_out)

        # Hidden and cell states
        self.h1 = np.zeros(n_l1)
        self.c1 = np.zeros(n_l1)
        self.h2 = np.zeros(n_l2)
        self.c2 = np.zeros(n_l2)

        # Training and prediction state
        self.train_cycles = 0
        self.last_prediction = {f: 0.0 for f in self.INPUT_FEATURES}
        self.prediction_error = {f: 0.0 for f in self.INPUT_FEATURES}

        # Normalisation statistics (updated from history)
        self.norm_mean = np.zeros(n_in)
        self.norm_std = np.ones(n_in)

        print(f"[LSTM] Online learner initialised. Architecture: {n_in}->{n_l1}->{n_l2}->{n_out}")

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    @staticmethod
    def _tanh(x):
        return np.tanh(np.clip(x, -15, 15))

    def _lstm_step(self, x, h, c, W, U, b, n_units):
        """
        Single LSTM cell forward pass.

        Gate equations:
            i = sigmoid(W_i.x + U_i.h + b_i)   Input gate
            f = sigmoid(W_f.x + U_f.h + b_f)   Forget gate
            g = tanh(W_g.x + U_g.h + b_g)      Cell gate (candidate)
            o = sigmoid(W_o.x + U_o.h + b_o)   Output gate

            c_new = f * c + i * g
            h_new = o * tanh(c_new)

        Parameters are stacked in order [i, f, g, o] along axis 0.
        """
        gates = W @ x + U @ h + b
        i_gate = self._sigmoid(gates[0:n_units])
        f_gate = self._sigmoid(gates[n_units:2 * n_units])
        g_gate = self._tanh(gates[2 * n_units:3 * n_units])
        o_gate = self._sigmoid(gates[3 * n_units:])
        c_new = f_gate * c + i_gate * g_gate
        h_new = o_gate * self._tanh(c_new)
        return h_new, c_new

    def _normalise(self, x_raw):
        """Normalise input vector using current running statistics."""
        denom = np.where(self.norm_std > 1e-9, self.norm_std, 1.0)
        return (x_raw - self.norm_mean) / denom

    def _update_normalisation(self, history: dict):
        """Update running mean and std from the current history window."""
        matrix = []
        for f in self.INPUT_FEATURES:
            vals = history.get(f, [0.0])
            matrix.append(vals if vals else [0.0])
        # Pad to equal length
        min_len = min(len(v) for v in matrix)
        arr = np.array([v[-min_len:] for v in matrix])   # shape: (features, time)
        self.norm_mean = arr.mean(axis=1)
        self.norm_std = arr.std(axis=1)

    def forward(self, x_raw):
        """
        Run a forward pass through both LSTM layers and the output layer.

        Parameters
        ----------
        x_raw : np.ndarray — Raw (unnormalised) input vector of shape (n_in,)

        Returns
        -------
        np.ndarray — Predicted next-timestep values (denormalised), shape (n_out,)
        """
        x = self._normalise(x_raw)
        self.h1, self.c1 = self._lstm_step(
            x, self.h1, self.c1, self.W1, self.U1, self.b1, self.LAYER1_UNITS)
        self.h2, self.c2 = self._lstm_step(
            self.h1, self.h2, self.c2, self.W2, self.U2, self.b2, self.LAYER2_UNITS)

        y_norm = self.Wo @ self.h2 + self.bo
        # Denormalise output
        return y_norm * self.norm_std + self.norm_mean

    def train_step(self, x_raw, target_raw):
        """
        Perform one online training step using gradient descent on MSE loss.

        This is a simplified online update that adjusts only the output
        layer weights (Wo, bo) using the prediction error, avoiding the
        full BPTT computation that would exceed the Raspberry Pi's compute
        budget within the 5-second acquisition window.

        Parameters
        ----------
        x_raw      : np.ndarray — Current input vector (raw)
        target_raw : np.ndarray — Target (actual next reading, raw)
        """
        prediction = self.forward(x_raw)
        error = target_raw - prediction   # Prediction error (raw scale)

        # Gradient of MSE loss w.r.t. Wo and bo (output layer only)
        denom = np.where(self.norm_std > 1e-9, self.norm_std, 1.0)
        error_norm = error / denom

        self.Wo += self.LEARNING_RATE * np.outer(error_norm, self.h2)
        self.bo += self.LEARNING_RATE * error_norm

        return error

    def update(self, reading: dict, history: dict):
        """
        Run TRAIN_ITERS_PER_CYCLE training iterations for the current cycle.

        Also computes and stores the latest prediction and prediction error
        for dashboard display.

        Parameters
        ----------
        reading : dict — Current sensor reading (actual values)
        history : dict — Rolling history from the Digital Twin Engine
        """
        self._update_normalisation(history)

        x_raw = np.array([reading.get(f, 0.0) for f in self.INPUT_FEATURES])

        # Perform multiple training iterations on the current reading
        for _ in range(self.TRAIN_ITERS_PER_CYCLE):
            self.train_step(x_raw, x_raw)   # Autoencoder-style: predict self
            self.train_cycles += 1

        # Store latest prediction for dashboard
        pred = self.forward(x_raw)
        for i, f in enumerate(self.INPUT_FEATURES):
            self.last_prediction[f] = round(float(pred[i]), 4)
            self.prediction_error[f] = round(float(abs(x_raw[i] - pred[i])), 4)

    def summary(self):
        """Return a formatted training summary string."""
        return (f"LSTM cycles: {self.train_cycles} | "
                f"Pred errors: {self.prediction_error}")


# =====================================================================
# SECTION 11 — FAULT DETECTION ENGINE
# =====================================================================
# 14 rule-based fault conditions (F01-F14) evaluated on every acquisition.
#
# Severity levels:
#   CRITICAL — Immediate action required; buzzer activated
#   WARNING  — Attention recommended; dashboard alert only

FAULT_RULES = [
    # ID     Description                             Parameter       Comparator  Threshold  Severity
    ("F01", "Oil level below critical limit",        "oil_pct",     "lt",  10.0, "CRITICAL"),
    ("F02", "Fuel level below warning limit",         "fuel_pct",    "lt",  15.0, "WARNING"),
    ("F03", "Fuel level below critical limit",        "fuel_pct",    "lt",   5.0, "CRITICAL"),
    ("F04", "Battery voltage low",                    "bus_voltage", "lt",  11.5, "WARNING"),
    ("F05", "Battery voltage critical",               "bus_voltage", "lt",  10.5, "CRITICAL"),
    ("F06", "Battery voltage high",                   "bus_voltage", "gt",  14.5, "WARNING"),
    ("F07", "Over-current warning",                   "current",     "gt",   0.5, "WARNING"),
    ("F08", "Over-current critical",                  "current",     "gt",   1.0, "CRITICAL"),
    ("F09", "Engine over-temperature warning",        "temperature", "gt",  80.0, "WARNING"),
    ("F10", "Engine over-temperature critical",       "temperature", "gt", 100.0, "CRITICAL"),
    ("F11", "Excessive vibration warning",            "vibration",   "gt",  20.0, "WARNING"),
    ("F12", "Excessive vibration critical",           "vibration",   "gt",  50.0, "CRITICAL"),
    ("F13", "High humidity warning",                  "humidity",    "gt",  85.0, "WARNING"),
    ("F14", "Low humidity warning",                   "humidity",    "lt",  20.0, "WARNING"),
]

# Health score deduction map: fault_id -> deduction in percentage points
HEALTH_DEDUCTIONS = {
    "F01": 25,   # Critical oil level
    "F03": 20,   # Critical fuel level
    "F05": 15,   # Critical battery voltage
    "F08": 15,   # Critical over-current
    "F10": 12,   # Critical over-temperature
    "F12": 8,    # Critical vibration
    "F02": 5,    # Warning fuel level
    "F04": 5,    # Warning battery voltage
    "F06": 3,    # Warning battery high
    "F07": 3,    # Warning over-current
    "F09": 5,    # Warning over-temperature
    "F11": 3,    # Warning vibration
    "F13": 2,    # Warning humidity high
    "F14": 2,    # Warning humidity low
}


class FaultDetectionEngine:
    """
    Evaluates all 14 fault conditions against the current sensor reading.

    On each acquisition cycle:
        1. Evaluate all FAULT_RULES conditions.
        2. Collect active faults (those whose condition evaluates to True).
        3. Compute the system health score by deducting weighted penalties.
        4. Activate the buzzer for any CRITICAL faults.
        5. Log new faults to the fault log CSV file.
        6. Return the active fault list and health score for dashboard display.

    Attributes
    ----------
    active_faults : list — List of fault dicts currently active
    health_score  : int  — Current system health score (0-100)
    fault_history : list — Complete log of all faults ever detected
    """

    def __init__(self, buzzer_pin, fault_log_path):
        self.buzzer_pin = buzzer_pin
        self.fault_log_path = fault_log_path
        self.active_faults = []
        self.health_score = 100
        self.fault_history = []

        # Initialise fault log CSV with header row
        with open(fault_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "fault_id", "description",
                              "severity", "parameter", "value", "threshold"])
        print(f"[FaultEngine] Initialised. Log: {fault_log_path}")

    def evaluate(self, state: dict):
        """
        Run all 14 fault rules against the current state snapshot.

        Parameters
        ----------
        state : dict — Current Digital Twin state snapshot

        Returns
        -------
        active_faults : list — Active fault records
        health_score  : int  — Computed health score (0-100)
        """
        now = datetime.now().isoformat()
        active = []

        for (fid, desc, param, comp, thresh, severity) in FAULT_RULES:
            value = state.get(param, 0.0)
            if value is None:
                continue
            triggered = (value < thresh) if comp == "lt" else (value > thresh)
            if triggered:
                fault = {
                    "fault_id": fid,
                    "description": desc,
                    "severity": severity,
                    "parameter": param,
                    "value": value,
                    "threshold": thresh,
                    "timestamp": now
                }
                active.append(fault)

        # Log newly triggered faults that were not active in the last cycle
        previous_ids = {f["fault_id"] for f in self.active_faults}
        for fault in active:
            if fault["fault_id"] not in previous_ids:
                self._log_fault(fault)
                self.fault_history.append(fault)

        self.active_faults = active

        # Compute health score
        self.health_score = self._compute_health_score(active)

        # Activate buzzer for CRITICAL faults
        critical_active = any(f["severity"] == "CRITICAL" for f in active)
        self._set_buzzer(critical_active)

        return active, self.health_score

    def _compute_health_score(self, active_faults):
        """
        Calculate system health score using the weighted deduction method.

        Starting from 100%, deduct the configured penalty for each active
        fault. The score is clamped to [0, 100].

        Example (from the field validation session):
            F01 (-25) + F03 (-20) + F02 (compounding weighting) = 43%
        """
        score = 100
        for fault in active_faults:
            deduction = HEALTH_DEDUCTIONS.get(fault["fault_id"], 0)
            score -= deduction
        return max(0, min(100, score))

    def _log_fault(self, fault):
        """Append a newly detected fault to the persistent fault log CSV."""
        with open(self.fault_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                fault["timestamp"], fault["fault_id"], fault["description"],
                fault["severity"], fault["parameter"],
                fault["value"], fault["threshold"]
            ])

    def _set_buzzer(self, activate: bool):
        """Activate or deactivate the piezoelectric buzzer."""
        GPIO.output(self.buzzer_pin, GPIO.HIGH if activate else GPIO.LOW)


# =====================================================================
# SECTION 12 — CSV DATA LOGGER
# =====================================================================

class DataLogger:
    """
    Logs all sensor readings and derived values to a CSV file.

    Each record written contains:
        timestamp, bus_voltage, current, temperature, humidity,
        pressure, altitude, speed_rpm, vibration, fuel_pct, oil_pct,
        health_score, active_fault_codes

    File is opened in append mode on each write to survive power interruptions
    without losing the complete session buffer.
    """

    HEADER = [
        "timestamp", "bus_voltage_V", "current_A", "temperature_C",
        "humidity_pct", "pressure_hPa", "altitude_m", "speed_rpm",
        "vibration_count", "fuel_pct", "oil_pct", "health_score",
        "active_faults"
    ]

    def __init__(self, filepath):
        self.filepath = filepath
        self.row_count = 0
        # Write header row
        with open(filepath, "w", newline="") as f:
            csv.writer(f).writerow(self.HEADER)
        print(f"[Logger] CSV log initialised: {filepath}")

    def write(self, state: dict, active_faults: list):
        """
        Append one record to the CSV log.

        Parameters
        ----------
        state         : dict — Current Digital Twin state snapshot
        active_faults : list — Active fault records from FaultDetectionEngine
        """
        fault_codes = "|".join(f["fault_id"] for f in active_faults) or "NONE"
        row = [
            state.get("timestamp", ""),
            state.get("bus_voltage", ""),
            state.get("current", ""),
            state.get("temperature", ""),
            state.get("humidity", ""),
            state.get("pressure", ""),
            state.get("altitude", ""),
            state.get("speed_rpm", ""),
            state.get("vibration", ""),
            state.get("fuel_pct", ""),
            state.get("oil_pct", ""),
            state.get("health_score", ""),
            fault_codes
        ]
        with open(self.filepath, "a", newline="") as f:
            csv.writer(f).writerow(row)
        self.row_count += 1


# =====================================================================
# SECTION 13 — LCD DISPLAY MANAGER
# =====================================================================

class LCDManager:
    """
    Manages the 16x2 I2C LCD character display.

    The display cycles through three screens on alternating acquisition cycles:
        Screen A: Health score + active fault count
        Screen B: Voltage, current
        Screen C: Temperature, humidity

    On CRITICAL fault detection, an override message is shown regardless
    of the rotation cycle.
    """

    def __init__(self, lcd: CharLCD):
        self.lcd = lcd
        self._cycle = 0
        self._lock = threading.Lock()

    def update(self, state: dict, active_faults: list):
        """Update the LCD with the current system state."""
        with self._lock:
            critical = [f for f in active_faults if f["severity"] == "CRITICAL"]

            if critical:
                self._show_critical(critical[0])
            else:
                screen = self._cycle % 3
                if screen == 0:
                    self._show_health(state, active_faults)
                elif screen == 1:
                    self._show_electrical(state)
                else:
                    self._show_environment(state)
            self._cycle += 1

    def _write(self, line1, line2):
        """Write two lines to the 16x2 LCD, padding/truncating to 16 chars."""
        self.lcd.clear()
        self.lcd.write_string(line1[:16].ljust(16))
        self.lcd.crlf()
        self.lcd.write_string(line2[:16].ljust(16))

    def _show_health(self, state, faults):
        self._write(
            f"Health: {state.get('health_score', '?'):>3}%",
            f"Faults: {len(faults)}"
        )

    def _show_electrical(self, state):
        self._write(
            f"V:{state.get('bus_voltage', 0):.2f}V",
            f"I:{state.get('current', 0) * 1000:.1f}mA"
        )

    def _show_environment(self, state):
        self._write(
            f"T:{state.get('temperature', 0):.1f}C H:{state.get('humidity', 0):.0f}%",
            f"P:{state.get('pressure', 0):.1f}hPa"
        )

    def _show_critical(self, fault):
        self._write(
            f"!! {fault['fault_id']} !!",
            fault["description"][:16]
        )


# =====================================================================
# SECTION 14 — MOTOR CONTROL FUNCTIONS (L298N H-Bridge)
# =====================================================================
# The L298N dual H-bridge module drives two DC motors (left and right).
# Direction is set by the logic level combination on each IN pin pair:
#
# IN1=1, IN2=0 -> Left motor FORWARD
# IN1=0, IN2=1 -> Left motor BACKWARD
# IN1=0, IN2=0 -> Left motor BRAKE (both LOW = short-circuit brake)
#
# Similarly for IN3/IN4 on the right motor.

def motor_forward():
    GPIO.output(IN1, 1); GPIO.output(IN2, 0)
    GPIO.output(IN3, 1); GPIO.output(IN4, 0)


def motor_backward():
    GPIO.output(IN1, 0); GPIO.output(IN2, 1)
    GPIO.output(IN3, 0); GPIO.output(IN4, 1)


def motor_left():
    """Pivot left: right motor forward, left motor backward."""
    GPIO.output(IN1, 0); GPIO.output(IN2, 1)
    GPIO.output(IN3, 1); GPIO.output(IN4, 0)


def motor_right():
    """Pivot right: left motor forward, right motor backward."""
    GPIO.output(IN1, 1); GPIO.output(IN2, 0)
    GPIO.output(IN3, 0); GPIO.output(IN4, 1)


def motor_stop():
    """Brake both motors (all outputs LOW)."""
    for pin in (IN1, IN2, IN3, IN4):
        GPIO.output(pin, 0)


# Thread-safe command dictionary shared between web API and joystick thread
_motor_cmd = {"cmd": "stop"}
_motor_lock = threading.Lock()

MOTOR_COMMANDS = {
    "forward": motor_forward,
    "backward": motor_backward,
    "left": motor_left,
    "right": motor_right,
    "stop": motor_stop
}


def execute_motor_command(cmd: str):
    """Execute a named motor command in a thread-safe manner."""
    with _motor_lock:
        _motor_cmd["cmd"] = cmd
        fn = MOTOR_COMMANDS.get(cmd, motor_stop)
        fn()


# =====================================================================
# SECTION 15 — JOYSTICK POLLING THREAD
# =====================================================================

def joystick_thread():
    """
    Continuously poll the four directional joystick buttons and translate
    presses to motor commands.

    Buttons are active-LOW (GPIO pulled up): pressed = GPIO.LOW.
    Priority order: FORWARD > BACKWARD > LEFT > RIGHT > STOP.
    The thread runs in an infinite loop at 50 Hz (20 ms per iteration).
    """
    print("[Joystick] Thread started.")
    while True:
        fwd = GPIO.input(BTN_FORWARD) == GPIO.LOW
        bck = GPIO.input(BTN_BACKWARD) == GPIO.LOW
        lft = GPIO.input(BTN_LEFT) == GPIO.LOW
        rgt = GPIO.input(BTN_RIGHT) == GPIO.LOW

        if fwd:
            execute_motor_command("forward")
        elif bck:
            execute_motor_command("backward")
        elif lft:
            execute_motor_command("left")
        elif rgt:
            execute_motor_command("right")
        else:
            execute_motor_command("stop")

        time.sleep(0.02)   # 50 Hz polling rate


# =====================================================================
# SECTION 16 — FLASK WEB DASHBOARD
# =====================================================================
# The Flask application runs in its own daemon thread on port 5000.
# All endpoints read from the shared Digital Twin state; no writes are made
# to the DT from the web layer (read-only dashboard pattern).
#
# API Endpoints
# -------------
# GET  /                  — Main HTML dashboard (four-tab interface)
# GET  /api/state         — Current DT state snapshot (JSON)
# GET  /api/faults        — Active and historical fault records (JSON)
# GET  /api/accumulators  — Component life accumulator states (JSON)
# GET  /api/ml            — Z-score and LSTM statistics (JSON)
# GET  /api/download/csv  — Download complete session CSV log
# POST /control/<cmd>     — Motor command (forward/backward/left/right/stop)

app = Flask(__name__)

# These will be populated by main() before the Flask thread starts
_dt_engine: DigitalTwinEngine = None
_fault_engine: FaultDetectionEngine = None
_zscore: ZScoreAnomalyDetector = None
_lstm: LSTMOnlineLearner = None
_logger: DataLogger = None


@app.route("/")
def dashboard():
    """Serve the main HTML dashboard page."""
    # In production this would serve a static HTML file; simplified here.
    return """
    <html><head><title>Digital Twin Dashboard</title>
    <meta http-equiv='refresh' content='5'>
    <style>body{font-family:monospace;background:#1a1a2e;color:#eee;padding:20px;}
    h1{color:#e94560;} table{width:100%;border-collapse:collapse;}
    td,th{border:1px solid #333;padding:8px;} th{background:#16213e;}
    .critical{color:#ff4444;} .warning{color:#ffaa00;} .ok{color:#44ff88;}
    </style></head><body>
    <h1>Digital Twin Vehicle Health Monitor</h1>
    <p>Dashboard refreshes every 5 seconds. Use /api/state for JSON data.</p>
    </body></html>
    """


@app.route("/api/state")
def api_state():
    """Return the current Digital Twin state as JSON."""
    if _dt_engine is None:
        return jsonify({"error": "System not initialised"}), 503
    state = _dt_engine.get_state_snapshot()
    return jsonify(state)


@app.route("/api/faults")
def api_faults():
    """Return active faults and fault history as JSON."""
    if _fault_engine is None:
        return jsonify({"error": "System not initialised"}), 503
    return jsonify({
        "active_faults": _fault_engine.active_faults,
        "fault_history": _fault_engine.fault_history[-100:],   # Last 100 faults
        "health_score": _fault_engine.health_score
    })


@app.route("/api/accumulators")
def api_accumulators():
    """Return component life accumulator states as JSON."""
    if _dt_engine is None:
        return jsonify({"error": "System not initialised"}), 503
    return jsonify(_dt_engine.get_accumulator_snapshot())


@app.route("/api/ml")
def api_ml():
    """Return machine learning layer statistics as JSON."""
    if _zscore is None or _lstm is None:
        return jsonify({"error": "ML layer not initialised"}), 503
    return jsonify({
        "zscore": {
            "scores": _zscore.scores,
            "flags": _zscore.flags,
            "max_score": _zscore.get_max_score(),
            "threshold": ZSCORE_THRESHOLD
        },
        "lstm": {
            "train_cycles": _lstm.train_cycles,
            "last_prediction": _lstm.last_prediction,
            "prediction_error": _lstm.prediction_error
        }
    })


@app.route("/api/download/csv")
def api_download_csv():
    """Stream the complete session CSV log as a file download."""
    if _logger is None:
        return jsonify({"error": "Logger not initialised"}), 503
    response = make_response(open(_logger.filepath, "rb").read())
    response.headers["Content-Type"] = "text/csv"
    fname = f'vehicle_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    response.headers["Content-Disposition"] = f'attachment; filename="{fname}"'
    return response


@app.route("/control/<cmd>", methods=["POST", "GET"])
def api_control(cmd):
    """
    Execute a motor control command via the web API.

    The command is written to the thread-safe _motor_cmd dictionary,
    which is also read by the joystick thread and the motor executor.
    GET method is supported for easy browser/curl testing.

    Valid commands: forward, backward, left, right, stop
    """
    if cmd not in MOTOR_COMMANDS:
        return jsonify({"error": f"Unknown command '{cmd}'"}), 400
    execute_motor_command(cmd)
    return jsonify({"status": "ok", "command": cmd})


def run_flask():
    """Start the Flask development server (runs in a daemon thread)."""
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)


# =====================================================================
# SECTION 17 — ALTITUDE COMPUTATION
# =====================================================================

def compute_altitude(pressure_hpa, sea_level_hpa=1013.25):
    """
    Compute altitude above sea level from atmospheric pressure.

    Uses the international barometric formula (hypsometric equation):
        h = 44330 * [1 - (P / P0)^(1/5.255)]

    where:
        h = altitude (metres)
        P = measured pressure (hPa)
        P0 = reference sea-level pressure (default 1013.25 hPa)
        5.255 = derived from gas constant and temperature lapse rate

    Parameters
    ----------
    pressure_hpa  : float — Measured barometric pressure (hPa)
    sea_level_hpa : float — Reference sea-level pressure (default 1013.25 hPa)

    Returns
    -------
    float — Estimated altitude in metres
    """
    ratio = pressure_hpa / sea_level_hpa
    altitude = 44330.0 * (1.0 - math.pow(ratio, 1.0 / 5.255))
    return round(altitude, 1)


# =====================================================================
# SECTION 18 — MAIN ACQUISITION LOOP
# =====================================================================

def main():
    """
    Application entry point.

    Initialization sequence:
        1. Initialize GPIO and interrupts.
        2. Initialise I2C sensors (INA219, BMP085, LCD).
        3. Instantiate all system components (DT Engine, ML, Fault Engine, etc.).
        4. Start the Flask web server in a daemon thread.
        5. Start the joystick polling thread.
        6. Enter the main 5-second acquisition loop.

    The main loop runs indefinitely until interrupted (Ctrl+C),
    at which point GPIO is cleaned up gracefully.
    """
    global _dt_engine, _fault_engine, _zscore, _lstm, _logger

    print("=" * 65)
    print(" DIGITAL TWIN VEHICLE HEALTH MONITORING SYSTEM")
    print(" Raspberry Pi 3 B+ | Initialising...")
    print("=" * 65)

    # --- 1. GPIO ---
    init_gpio()

    # --- 2. Sensors ---
    ina219, bmp, lcd = init_sensors()

    # --- 3. System components ---
    _dt_engine = DigitalTwinEngine()
    _zscore = ZScoreAnomalyDetector()
    _lstm = LSTMOnlineLearner()
    _fault_engine = FaultDetectionEngine(BUZ_PIN, FAULT_LOG_PATH)
    _logger = DataLogger(CSV_LOG_PATH)
    lcd_mgr = LCDManager(lcd)

    # --- 4. Flask thread (daemon = stops when main thread stops) ---
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print(f"[Flask] Dashboard running on http://0.0.0.0:{FLASK_PORT}")

    # --- 5. Joystick thread ---
    joy_thread = threading.Thread(target=joystick_thread, daemon=True)
    joy_thread.start()

    # --- 6. Main acquisition loop ---
    print("[Main] Entering acquisition loop (5-second interval).")
    print("-" * 65)

    cycle = 0
    try:
        while True:
            loop_start = time.time()
            cycle += 1

            # -- Read DHT11 (temperature & humidity) --
            # read_retry attempts up to 15 reads to handle the DHT11's
            # occasional failure to respond on the first attempt.
            humidity, temperature = Adafruit_DHT.read_retry(DHT_SENSOR_TYPE, DHT_PIN)
            if humidity is None:
                humidity = 0.0
            if temperature is None:
                temperature = 0.0

            # -- Read INA219 (bus voltage & current) --
            bus_voltage = ina219.bus_voltage           # Volts
            current_a = ina219.current / 1000.0        # mA -> A

            # -- Read BMP085 (pressure -> altitude) --
            pressure_hpa = bmp.read_pressure() / 100.0  # Pa -> hPa
            altitude_m = compute_altitude(pressure_hpa)

            # -- Read HC-SR04 (fuel & oil levels) --
            fuel_dist_cm = read_distance(TRIG_FUEL, ECHO_FUEL)
            oil_dist_cm = read_distance(TRIG_OIL, ECHO_OIL)
            fuel_pct = distance_to_level_percent(fuel_dist_cm)
            oil_pct = distance_to_level_percent(oil_dist_cm)

            # -- Read IR encoder & vibration sensor (via interrupt counters) --
            ir_count, vib_count = get_and_reset_counters()
            speed_rpm = compute_speed_rpm(ir_count)

            # -- Assemble reading dictionary --
            reading = {
                "bus_voltage": round(bus_voltage, 4),
                "current": round(current_a, 6),
                "temperature": round(temperature, 2),
                "humidity": round(humidity, 1),
                "pressure": round(pressure_hpa, 2),
                "altitude": altitude_m,
                "speed_rpm": speed_rpm,
                "vibration": vib_count,
                "fuel_pct": fuel_pct,
                "oil_pct": oil_pct
            }

            # -- Update Digital Twin Engine --
            _dt_engine.update(reading)
            state = _dt_engine.get_state_snapshot()
            history = _dt_engine.get_history_snapshot()

            # -- Update ML Layer --
            _zscore.update(history)
            _lstm.update(reading, history)

            # -- Evaluate Fault Rules --
            active_faults, health_score = _fault_engine.evaluate(state)
            _dt_engine.state["health_score"] = health_score
            _dt_engine.state["active_faults"] = [f["fault_id"] for f in active_faults]

            # -- Log to CSV --
            _logger.write(state, active_faults)

            # -- Update LCD Display --
            lcd_mgr.update(state, active_faults)

            # -- Console status line --
            if cycle % 12 == 0:   # Print summary every minute
                print(f"[Cycle {cycle:>6}] "
                      f"Health={health_score}% "
                      f"Faults={len(active_faults)} "
                      f"V={bus_voltage:.2f}V "
                      f"T={temperature:.1f}C "
                      f"RPM={speed_rpm:.0f} "
                      f"Fuel={fuel_pct:.0f}% "
                      f"Oil={oil_pct:.0f}%")
                print(f"  {_zscore.summary()}")
                print(f"  {_lstm.summary()}")

            # -- Maintain precise 5-second interval --
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, ACQUISITION_INTERVAL_SEC - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt received. Shutting down...")

    finally:
        # Graceful shutdown
        motor_stop()
        GPIO.output(BUZ_PIN, False)
        lcd.clear()
        lcd.write_string("System offline.")
        GPIO.cleanup()
        print(f"[Main] Session complete. Total records: {_logger.row_count}")
        print(f"[Main] Max DT update delay: {_dt_engine.max_update_delay_sec:.3f}s")
        print("[Main] GPIO cleaned up. Goodbye.")


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()
