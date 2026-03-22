🔋 Battery ESC + SPKF Engine (Python + Rust)

## Overview
This project was inspired by my coursework in Algorithms for Battery Management Systems at the University of Colorado at Boulder which I completed during my MS in Computer Science. Professor Gregory Plett has published several online resources related to this topic, including http://mocha-java.uccs.edu/BMS1/index.html. 

I implemented a battery equivalent circuit (ESC) model and a Sigma-Point Kalman Filter (SPKF) for state-of-charge (SOC) estimation, with a dual-language architecture:
Python: orchestration, data handling, testing, visualization
Rust: high-performance numerical engine for ESC dynamics

The system is designed to support both offline analysis and future streaming, real-time inference pipelines.

## Why This Matters
This project recasts matlab/Octave code into new languages that extend the reach of these valuable BMS related algorithms:
hybrid Python + Rust system design
numerical modeling + estimation algorithms
production-ready validation (parity testing)
extensibility toward real-time energy systems

## Key Features
1. ESC Battery Model
Temperature-dependent parameter interpolation
OCV(SOC, T) and dOCV/dSOC lookup
RC branch dynamics and hysteresis modeling
Fully validated against reference Python implementation
2. SPKF Pipeline (Python baseline)
Sigma-point generation
Nonlinear state propagation
Measurement update
SOC estimation from voltage/current signals
3. Rust Acceleration Layer
Rust implementation of:
interpolation
parameter lookup
ESC state equation
ESC output equation
Exposed via PyO3 + maturin
Designed as a reusable compute engine
4. Python ↔ Rust Bridge
Clean interface via RustBatterySpkfEngine
Model exported as flat dict → Rust struct
Runtime toggle:
USE_RUST = True / False
5. Numerical Parity Validation
A dedicated test suite verifies that Rust matches Python:
ESC state evolution (ir, hk, soc)
Predicted voltage
Innovation

All comparisons match to machine precision (~1e-16).

## Example Results
Python and Rust pipelines produce identical metrics:
soc_rmse: 0.0014775
voltage_rmse: 7.98e-05

State trajectories and outputs are numerically equivalent.

## Architecture
Python Layer
├── Data loading (MAT files)
├── SPKF orchestration
├── Testing + plotting
└── Rust bridge adapter

Rust Layer
├── interp.rs      (linear interpolation)
├── params.rs      (temperature parameter lookup)
├── ocv.rs         (OCV + derivative)
├── esc.rs         (state + output equations)
└── engine.rs      (stateful compute engine)

## Design Goals
Separate model orchestration (Python) from numerical compute (Rust)
Enable future:
real-time streaming pipelines
multi-device simulation
high-throughput inference
Maintain strict numerical parity across implementations

## Tech Stack
Python (NumPy, Pandas)
Rust (PyO3, maturin)
MATLAB data ingestion
Battery modeling (ESC, SPKF)

## Phase 1 Status
ESC model fully implemented in Rust
Python ↔ Rust integration complete
Numerical parity validated
End-to-end SOC pipeline working

## Next Steps
Full SPKF implementation in Rust (sigma-point update)
Streaming integration (Kafka / Redpanda)
Multi-cell / pack-level modeling
GPU / SIMD optimization exploration

## Rust Extension Setup
This project includes a Rust-based compute engine exposed to Python using PyO3 and maturin.

### Prerequisites
- Python 3.8+
- Rust toolchain
- maturin