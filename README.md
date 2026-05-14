# doppler-positioning-sim

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A desktop Tkinter simulator of the **Transit / NAVSAT** Doppler-positioning method — the operational pre-GPS satellite navigation system used by the U.S. Navy from 1964 until 1996. Generates a realistic single-pass geometry, computes integrated Doppler counts on a 400 MHz carrier, adds ionospheric delay and observation noise, then inverts via batch Gauss–Newton least squares for the receiver's ECEF position.

## Features

- Closed-form circular polar-orbit propagator with full ECI → ECEF rotation.
- Geodetic ↔ ECEF conversion using WGS-84 parameters with iterative inverse.
- Single-layer Klobuchar-style slant ionospheric delay (TECU input).
- Batch least-squares solver with covariance estimate and residual statistics.
- Live plots of Doppler count series and elevation profile.

## Why this matters

Doppler positioning is the conceptual bridge between celestial navigation and modern GNSS pseudoranging. Reconstructing it from first principles makes the underlying integrated-range observable transparent — invaluable for teaching, for evaluating future LEO-PNT systems (Iridium, Starlink-based timing), and for understanding the constraints that drove GPS architecture.

## Quick start

```bash
pip install -r requirements.txt
python doppler_positioning_sim.py
```

## References

- Stansell, T. A. (1978). *The Navy Navigation Satellite System: Description and Status.* Navigation, 25(2).
- Hofmann-Wellenhof, B., Lichtenegger, H., & Wasle, E. (2008). *GNSS — Global Navigation Satellite Systems.* Springer.
- Misra, P. & Enge, P. (2011). *Global Positioning System: Signals, Measurements, and Performance.* Ganga-Jamuna Press.

## Author

**Dr. Mosab Hawarey**
>
PhD, Geodetic & Photogrammetric Engineering (ITU) | MSc, Geomatics (Purdue) | MBA (Wales) | BSc, MSc (METU)

- GitHub: https://github.com/mhawarey
- Personal: https://hawarey.org/mosab
- ORCID: https://orcid.org/0000-0001-7846-951X

## License

MIT License