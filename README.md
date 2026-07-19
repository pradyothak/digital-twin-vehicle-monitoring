# Digital Twin-Based Vehicle Health Monitoring & Predictive Maintenance System

A fully embedded Digital Twin framework for real-time vehicle health monitoring and predictive maintenance — running entirely on a Raspberry Pi 3 Model B+, with **no cloud dependency and no GPU**.

The system fuses 9 live sensor channels into a real-time virtual state model, layers statistical anomaly detection and an online-trained LSTM on top of it. It evaluates 14 rule-based fault conditions every cycle — surfacing all findings via a live Flask dashboard.

**[View all dashboard photos](images/)**
## Why this project

Most Digital Twin predictive-maintenance systems in the literature depend on cloud infrastructure or GPU-accelerated hardware and typically monitor only one or two vehicle subsystems in isolation. This project shows that a complete, multi-parameter, ML-enabled Digital Twin can run entirely on a $35 single-board computer — while simultaneously covering the electrical, thermal, mechanical, and fluid domains of a vehicle in one unified system.

## Key Results (31 hr 49 min field validation)

| Metric | Result |
|---|---|
| Session length | 31 h 49 min 53 s continuous operation |
| Acquisition cycles | 22,918 (5-second intervals) |
| State-update latency | < 5 seconds, sustained throughout |
| Data integrity | Zero bus collisions, GPIO conflicts, or log write errors |
| Anomaly detection | Max Z-score of 1.41 (correctly below the Z=2 threshold under stationary test conditions) |
| Fault detection | Correctly identified 3 active faults (critical oil depletion, warning + critical low fuel) |
| Computed health score | 43%, numerically consistent with the weighted fault-deduction formula |

All 7 design objectives (real-time twin sync, multi-sensor monitoring, live dashboard, ML-based anomaly detection, dual-channel alerting, autonomous + manual control, low-cost scalability) were empirically validated against this session.

## System Architecture
- **Real-time state mirror** — thread-safe dictionary holding the latest reading per channel, refreshed every 5s
- **Rolling history buffers** — 60-entry deque per channel (5-min window) for anomaly scoring, plus a 3,600-entry buffer (~5 hrs) feeding the LSTM
- **Component-life accumulators** — track cumulative health for engine, battery, fuel system, and suspension
- **Three concurrent analysis layers**:
  1. Rule-based engine — 14 fault codes (F01–F14), evaluated every cycle
  2. Z-score anomaly detector — flags deviations against a 60-reading rolling window (Z > 2 unusual, Z > 3 anomalous)
  3. Online-trained 2-layer LSTM — predicts next-step vehicle health from a rolling 20-step, 5-channel input sequence, retrained live every cycle (3 epochs/cycle, no offline training needed)

## Sensor Suite

| Sensor | Measures | Interface |
|---|---|---|
| INA219 | Bus voltage & current | I²C |
| DHT11 | Ambient temperature & humidity | GPIO |
| BMP180 | Barometric pressure & altitude | I²C |
| SW-420 | Vibration events | GPIO interrupt |
| IR encoder | Wheel speed | GPIO interrupt |
| HC-SR04 (×2) | Fuel & oil level (ultrasonic) | GPIO, median-filtered |

## Tech Stack

`Python 3` · `TensorFlow / Keras` (LSTM) · `Flask` (REST API + dashboard) · `NumPy` · `RPi.GPIO` / Adafruit drivers · `Raspberry Pi OS`
Four concurrent threads (sensor acquisition, LSTM inference, LCD update, Flask server) share state through a lock-protected dictionary to avoid race conditions.

## Source Code
**[View source code](src/)**

## Hardware Prototype

*Final hardware prototype — sensor placement across fuel/oil tanks, Raspberry Pi, battery pack, and joystick.*

## Live Dashboard

The system exposes a browser-based dashboard (polling every 2s) across four tabs — **Sensors, Digital Twin, Faults, Summary** — with colour-coded status and CSV export.

*Sensors tab — live vehicle readings on the LCD (left) and web dashboard (right).*

*Digital Twin tab — component life accumulators and Z-score anomaly detection.*

*Faults tab (active fault list) and Summary tab (session overview + sensor averages).*

*Session log view — timestamped sensor data, downloadable as CSV.*


## Limitations & Future Work

- Validated on a single stationary session — dynamic driving conditions (variable speed, mechanical load) remain to be tested
- Prototype-grade sensor tolerances (DHT11, HC-SR04) are wider than automotive-grade instrumentation
- Planned: migration to a more powerful board (Jetson Nano / Pi 5), OBD-II + automotive-grade sensors, fleet-wide cloud sync, and adaptive fault thresholds


Built as part of a final-year major project at VNR Vignana Jyothi Institute of Engineering & Technology, Department of Electronics & Instrumentation Engineering.

Team - Pradyotha, Harshini, Pranathi
