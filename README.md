# Circular Restricted Three-Body Problem (CR3BP) Solver

[![CI Test Suite](https://github.com/rtfisher/circular_restricted_three_body_problem/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rtfisher/circular_restricted_three_body_problem/actions/workflows/ci.yml)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A Python implementation of the circular restricted three-body problem integrator with automatic plot bounds, equipotential contours, and animation support.

## Sample Output

![L1 Lagrange Point Instability](cr3bp_orbit_L1.png)

*Example trajectory showing the unstable nature of the L₁ Lagrange point in the Earth-Moon system (μ = 0.1). The particle begins near L₁ and departs along the unstable manifold.*

## Overview

This repository contains a single Python script that numerically integrates the motion of a massless particle in the **circular restricted three-body problem (CR3BP)** within the uniformly rotating reference frame. The solver automatically sizes plot windows to contain the entire trajectory plus both primary bodies, ensuring that orbits never run out of bounds.

The CR3BP is a simplified model of celestial mechanics where two massive bodies orbit their common center of mass in circular orbits, and a third massless body (such as a spacecraft or asteroid) moves under their gravitational influence. This model is particularly useful for studying:

- Lagrange points (L1-L5) and their stability
- Halo orbits and transfer trajectories
- Earth-Moon system dynamics
- Trojan asteroids in planetary systems

## Features

- **Adaptive integration**: Uses SciPy's `solve_ivp` with the high-order DOP853 method
- **Automatic plot bounds**: Computes tight, square plot windows from integrated trajectories
- **Equipotential contours**: Visualizes the effective potential in the rotating frame
- **Static and animated output**: Generates both PNG images and MP4 animations
- **Dimensionless scaling**: Uses standard CR3BP units where total mass = 1, separation = 1, and rotation rate Ω = 1
- **Jacobi constant tracking**: Monitors the conserved quantity throughout integration

## Installation

### Requirements

Python 3.9 or higher with the following packages:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy scipy matplotlib pytest pytest-cov
```

### Optional: Animation Support

To generate MP4 animations, you'll need `ffmpeg` installed and available on your PATH:

- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Usage

### Basic Command

```bash
python three_body_integrator.py --mu 0.1 --x0 0.3 --y0 0.0 --vx0 0.0 --vy0 0.5 --tmax 40
```

### Command-Line Arguments

- `--mu`: Reduced mass parameter μ ∈ [0, 1] (mass ratio of the system)
- `--x0`, `--y0`: Initial position in the rotating frame
- `--vx0`, `--vy0`: Initial velocity in the rotating frame
- `--tmax`: Final integration time (dimensionless)
- `--max-step`: Maximum integrator step size (default: 5e-3)
- `--no-static`: Skip saving the static PNG plot
- `--no-anim`: Skip saving the MP4 animation
- `--trail`: Number of integration steps to show in animation trail (default: 300)
- `--inertial`: Plot in the inertial frame (primaries rotate; no effective potential contours)

### Output Files

**Rotating Frame (default):**
- `cr3bp_orbit.png` — Static plot showing the complete trajectory with effective potential contours
- `cr3bp_animation.mp4` — Animation of the particle's motion (requires ffmpeg)

**Inertial Frame (with `--inertial` flag):**
- `cr3bp_orbit_inertial.png` — Static plot with rotating primaries (no potential contours)
- `cr3bp_animation_inertial.mp4` — Animation showing rotating primaries and particle

## Example Scenarios

### 1. L₁ Instability

Demonstrate the unstable nature of the L₁ Lagrange point with μ = 0.1:

**Position perturbation only:**
```bash
python three_body_integrator.py \
  --mu 0.1 --x0 0.5791 --y0 0.0 --vx0 0.0 --vy0 0.0 --tmax 80 --max-step 5e-3
```

**Tiny velocity perturbation:**
```bash
python three_body_integrator.py \
  --mu 0.1 --x0 0.5790 --y0 0.0 --vx0 0.0 --vy0 1e-4 --tmax 60 --max-step 5e-3
```

In both cases, the particle initially hovers near L₁ before departing along the unstable manifold — a hallmark of saddle equilibrium.

### 2. L₄ and L₅ Stability

The triangular Lagrange points L₄ and L₅ are linearly stable in the planar CR3BP when μ ≲ 0.03852 (Routh threshold).

**Near L₄ with μ = 0.01:**
```bash
python three_body_integrator.py \
  --mu 0.01 --x0 0.54 --y0 0.8660254 --vx0 0.0 --vy0 -0.045 --tmax 350 --max-step 3e-3
```

**Near L₅ with μ = 0.01:**
```bash
python three_body_integrator.py \
  --mu 0.01 --x0 0.49 --y0 -0.8660254 --vx0 0.0 --vy0 1e-4 --tmax 120 --max-step 5e-3
```

These examples show bounded tadpole-like librations, illustrating the stability of L₄ and L₅ for small mass ratios.

### 3. Earth-Moon System

For the Earth-Moon CR3BP, the mass parameter is approximately μ ≈ 0.01215:

```bash
python three_body_integrator.py \
  --mu 0.01215 --x0 0.48785 --y0 0.8660254 --vx0 0.0 --vy0 8e-5 --tmax 200
```

### 4. Inertial Frame Visualization

View the same orbit in the **inertial frame** where the primaries rotate:

```bash
python three_body_integrator.py \
  --mu 0.1 --x0 0.3 --y0 0.0 --vx0 0.0 --vy0 0.5 --tmax 40 --inertial
```

In the inertial frame:
- The two primaries rotate around their common barycenter (at the origin)
- The effective potential contours are **not** shown (they only exist in the rotating frame)
- Circular orbits show the paths of M₁ and M₂
- Primary positions are shown at t=0 (circles) and t=end (squares)

## Physics and Coordinates

### Reference Frames

The code integrates in the **uniformly rotating frame** where:
- The two primary bodies remain fixed on the x-axis
- Primary M₁ (mass 1-μ) is located at (-μ, 0)
- Primary M₂ (mass μ) is located at (1-μ, 0)
- The frame rotates with angular velocity Ω = 1

Visualization can be in either frame:
- **Rotating frame** (default): Primaries are stationary, effective potential is shown
- **Inertial frame** (`--inertial`): Primaries rotate around the barycenter at the origin

### Equations of Motion

The CR3BP equations in the rotating frame are:

```
ẍ - 2ẏ = ∂Ω/∂x
ÿ + 2ẋ = ∂Ω/∂y
```

where Ω(x,y) is the effective potential:

```
Ω = (x² + y²)/2 + (1-μ)/r₁ + μ/r₂
```

and r₁, r₂ are the distances from the particle to M₁ and M₂, respectively.

### Jacobi Constant

The Jacobi constant C = 2Ω - v² is an integral of motion in the CR3BP, meaning it remains constant along trajectories. The code computes and reports C at t = 0.

## Testing

The repository includes a comprehensive test suite covering unit tests, integration tests, and physics validation.

### Running Tests

Run all tests:
```bash
pytest test_three_body.py -v
```

Run with coverage report:
```bash
pytest test_three_body.py --cov=three_body_integrator --cov-report=html
```

Run specific test categories:
```bash
# Test Jacobi constant conservation
pytest test_three_body.py::TestJacobiConstantConservation -v

# Test numerical accuracy
pytest test_three_body.py::TestNumericalAccuracy -v

# Test Lagrange point equilibria
pytest test_three_body.py::TestLagrangePoints -v
```

### Test Coverage

The test suite includes:
- **Unit tests**: Individual function validation (radii, potential, equations of motion)
- **Integration tests**: Full orbital dynamics with various initial conditions
- **Conservation laws**: Jacobi constant conservation to machine precision
- **Angular momentum**: Rotating frame angular momentum behavior
- **Numerical accuracy**: Convergence tests with different tolerances
- **Lagrange points**: Equilibrium verification at L1, L2, L4, L5
- **Edge cases**: Extreme mass ratios, zero velocities, equal masses

### Continuous Integration

Tests run automatically on:
- Every push to `main` or `develop` branches
- All pull requests
- Daily at 2:00 AM UTC (scheduled)

The CI workflow tests across multiple Python versions (3.9-3.12) and operating systems (Ubuntu, macOS, Windows).

## Tips and Best Practices

- **Sharpening curves**: Reduce `--max-step` (e.g., to 2e-3) for smoother trajectories at the cost of longer computation
- **Faster runs**: Increase `--max-step` for quicker but less precise integration
- **Missing animations**: Use `--no-anim` if ffmpeg is not installed
- **Plot bounds**: The script automatically computes axis limits after integration, ensuring long trajectories fit cleanly
- **Contour levels**: Equipotential levels are chosen from quantiles to remain readable and avoid deep singular wells near the primaries

## Project Structure

```
.
├── three_body_integrator.py    # Main solver and visualization code
├── test_three_body.py          # Comprehensive test suite
├── requirements.txt            # Python package dependencies
├── pytest.ini                  # Pytest configuration
├── README.md                   # This file
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore rules
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI workflow
├── cr3bp_orbit.png             # Sample static output
├── cr3bp_orbit_L1.png          # Sample L1 instability plot
├── cr3bp_animation.mp4         # Sample animation
└── cr3bp_animation_L1.mp4      # Sample L1 instability animation
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and send pull requests.

## References

For more information on the circular restricted three-body problem:

- Koon, W. S., Lo, M. W., Marsden, J. E., & Ross, S. D. (2011). *Dynamical Systems, the Three-Body Problem and Space Mission Design*
- Szebehely, V. (1967). *Theory of Orbits: The Restricted Problem of Three Bodies*
- Murray, C. D., & Dermott, S. F. (1999). *Solar System Dynamics*

## Acknowledgments

This implementation was developed as an educational tool for studying orbital mechanics and dynamical systems in celestial mechanics.
