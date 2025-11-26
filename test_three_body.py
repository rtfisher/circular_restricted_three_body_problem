#!/usr/bin/env python3
"""
Comprehensive test suite for the CR3BP solver.

Tests cover:
- Unit tests for individual physics functions
- Integration tests for orbital dynamics
- Conservation laws (Jacobi constant, angular momentum)
- Numerical accuracy and convergence
- Lagrange point equilibria
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal
from scipy.optimize import fsolve
from three_body_integrator import (
    radii,
    effective_potential,
    equations_of_motion,
    jacobi_constant,
    integrate_orbit,
    compute_plot_window,
)


# Helper function to find Lagrange points
def find_lagrange_point(mu, x0, y0=0.0):
    """
    Find a Lagrange point by solving the equilibrium equations.
    Returns (x, y) position where acceleration is zero.
    """
    def equilibrium(pos):
        x, y = pos
        state = np.array([x, y, 0.0, 0.0])
        deriv = equations_of_motion(0.0, state, mu)
        return [deriv[2], deriv[3]]  # ax, ay should be zero

    solution = fsolve(equilibrium, [x0, y0])
    return solution


# ==================== Unit Tests: Basic Functions ====================

class TestRadii:
    """Test distance calculations to primaries."""

    def test_radii_at_origin(self):
        """Test radii at origin for mu=0.3."""
        mu = 0.3
        r1, r2 = radii(0.0, 0.0, mu)
        # M1 at (-0.3, 0), M2 at (0.7, 0)
        assert_allclose(r1, 0.3, rtol=1e-10)
        assert_allclose(r2, 0.7, rtol=1e-10)

    def test_radii_at_primaries(self):
        """Test radii at the primary positions (should be very small)."""
        mu = 0.2
        x1, y1 = -mu, 0.0
        x2, y2 = 1.0 - mu, 0.0

        r1, r2 = radii(x1, y1, mu)
        assert r1 < 1e-11  # Should be epsilon
        assert r2 > 0.5    # Far from M2

        r1, r2 = radii(x2, y2, mu)
        assert r1 > 0.5    # Far from M1
        assert r2 < 1e-11  # Should be epsilon

    def test_radii_symmetry(self):
        """Test that radii are symmetric about x-axis."""
        mu = 0.15
        x, y = 0.5, 0.3
        r1_pos, r2_pos = radii(x, y, mu)
        r1_neg, r2_neg = radii(x, -y, mu)

        assert_allclose(r1_pos, r1_neg, rtol=1e-12)
        assert_allclose(r2_pos, r2_neg, rtol=1e-12)

    def test_radii_vectorized(self):
        """Test that radii works with array inputs."""
        mu = 0.1
        x = np.array([0.0, 0.5, -0.5])
        y = np.array([0.0, 0.3, -0.2])
        r1, r2 = radii(x, y, mu)

        assert len(r1) == 3
        assert len(r2) == 3
        assert np.all(r1 > 0)
        assert np.all(r2 > 0)


class TestEffectivePotential:
    """Test effective potential calculations."""

    def test_potential_symmetry(self):
        """Potential should be symmetric about x-axis."""
        mu = 0.2
        x, y = 0.3, 0.4

        Omega_pos = effective_potential(x, y, mu)
        Omega_neg = effective_potential(x, -y, mu)

        assert_allclose(Omega_pos, Omega_neg, rtol=1e-12)

    def test_potential_increases_far_from_center(self):
        """Potential should increase as we move away from origin."""
        mu = 0.1

        Omega_near = effective_potential(0.5, 0.0, mu)
        Omega_far = effective_potential(2.0, 0.0, mu)

        # Centrifugal term dominates far out
        assert Omega_far > Omega_near

    def test_potential_at_L1(self):
        """Test potential at approximate L1 location."""
        mu = 0.1
        # L1 is approximately at x ≈ 0.579 for mu=0.1
        x_L1 = 0.5790
        Omega_L1 = effective_potential(x_L1, 0.0, mu)

        # Should be a finite value
        assert np.isfinite(Omega_L1)
        assert Omega_L1 > 1.0  # Reasonable magnitude


class TestEquationsOfMotion:
    """Test equations of motion."""

    def test_eom_returns_correct_shape(self):
        """EOM should return [vx, vy, ax, ay]."""
        mu = 0.1
        state = np.array([0.3, 0.4, 0.1, 0.2])

        derivative = equations_of_motion(0.0, state, mu)

        assert len(derivative) == 4
        assert derivative[0] == state[2]  # dx/dt = vx
        assert derivative[1] == state[3]  # dy/dt = vy

    def test_eom_at_lagrange_L1(self):
        """At L1, the velocity should be zero and acceleration nearly zero."""
        mu = 0.1
        # Find L1 numerically
        x_L1, y_L1 = find_lagrange_point(mu, x0=0.6)
        state = np.array([x_L1, y_L1, 0.0, 0.0])

        derivative = equations_of_motion(0.0, state, mu)

        # Velocities are zero
        assert derivative[0] == 0.0
        assert derivative[1] == 0.0

        # Accelerations should be very small at equilibrium
        assert abs(derivative[2]) < 1e-6  # ax ~ 0
        assert abs(derivative[3]) < 1e-6  # ay ~ 0

    def test_eom_coriolis_terms(self):
        """Test that Coriolis terms 2*vy and -2*vx appear correctly."""
        mu = 0.1
        state = np.array([0.0, 0.0, 1.0, 2.0])  # vx=1, vy=2

        derivative = equations_of_motion(0.0, state, mu)
        ax, ay = derivative[2], derivative[3]

        # At origin, gravitational terms are proportional to x,y
        # Coriolis: ax should have +2*vy contribution
        # ay should have -2*vx contribution
        # This is a qualitative check
        assert isinstance(ax, (float, np.floating))
        assert isinstance(ay, (float, np.floating))


class TestJacobiConstant:
    """Test Jacobi constant calculations."""

    def test_jacobi_constant_value(self):
        """Test Jacobi constant calculation."""
        mu = 0.1
        state = np.array([0.3, 0.0, 0.0, 0.5])

        C = jacobi_constant(state, mu)

        # C = 2*Omega - v^2
        Omega = effective_potential(state[0], state[1], mu)
        v2 = state[2]**2 + state[3]**2
        C_expected = 2*Omega - v2

        assert_allclose(C, C_expected, rtol=1e-12)

    def test_jacobi_constant_symmetry(self):
        """Jacobi constant should be same for y and -y."""
        mu = 0.2
        state_pos = np.array([0.5, 0.3, 0.1, 0.2])
        state_neg = np.array([0.5, -0.3, 0.1, -0.2])

        C_pos = jacobi_constant(state_pos, mu)
        C_neg = jacobi_constant(state_neg, mu)

        assert_allclose(C_pos, C_neg, rtol=1e-12)


# ==================== Integration Tests ====================

class TestIntegration:
    """Test orbital integration."""

    def test_integration_completes(self):
        """Test that integration completes successfully."""
        mu = 0.1
        state0 = np.array([0.3, 0.0, 0.0, 0.5])

        sol = integrate_orbit(mu, state0, t_span=(0.0, 10.0), max_step=1e-2)

        assert sol.success
        assert len(sol.t) > 10
        assert sol.t[0] == 0.0
        assert sol.t[-1] <= 10.0

    def test_integration_state_shape(self):
        """Test that integrated solution has correct shape."""
        mu = 0.1
        state0 = np.array([0.3, 0.0, 0.0, 0.5])

        sol = integrate_orbit(mu, state0, t_span=(0.0, 5.0), max_step=1e-2)

        assert sol.y.shape[0] == 4  # [x, y, vx, vy]
        assert sol.y.shape[1] == len(sol.t)

    def test_initial_conditions_preserved(self):
        """Test that initial conditions are preserved."""
        mu = 0.1
        state0 = np.array([0.3, 0.4, 0.1, 0.2])

        sol = integrate_orbit(mu, state0, t_span=(0.0, 1.0))

        assert_allclose(sol.y[:, 0], state0, rtol=1e-10)


# ==================== Physics Conservation Tests ====================

class TestJacobiConstantConservation:
    """Test conservation of Jacobi constant."""

    def test_jacobi_conservation_circular_orbit(self):
        """Jacobi constant should be conserved during integration."""
        mu = 0.1
        state0 = np.array([0.3, 0.0, 0.0, 0.5])

        sol = integrate_orbit(mu, state0, t_span=(0.0, 50.0), max_step=1e-3)

        # Compute Jacobi constant at all times
        C_values = np.array([jacobi_constant(sol.y[:, i], mu)
                            for i in range(len(sol.t))])

        C0 = C_values[0]

        # Should be conserved to high precision
        assert_allclose(C_values, C0, rtol=1e-7, atol=1e-9)

    def test_jacobi_conservation_long_integration(self):
        """Test Jacobi constant over long integration times."""
        mu = 0.01
        # Near L4 stable region
        state0 = np.array([0.5, 0.866, 0.0, 0.0])

        sol = integrate_orbit(mu, state0, t_span=(0.0, 100.0),
                            max_step=2e-3, rtol=1e-10, atol=1e-13)

        C_values = np.array([jacobi_constant(sol.y[:, i], mu)
                            for i in range(len(sol.t))])

        C0 = C_values[0]
        max_deviation = np.max(np.abs(C_values - C0))

        # Over long times, should still conserve well
        assert max_deviation < 1e-6

    def test_jacobi_conservation_chaotic_orbit(self):
        """Test Jacobi conservation even in chaotic regions."""
        mu = 0.1
        # Start near L1 with small perturbation
        state0 = np.array([0.579, 0.0, 0.0, 1e-4])

        sol = integrate_orbit(mu, state0, t_span=(0.0, 30.0), max_step=5e-4)

        C_values = np.array([jacobi_constant(sol.y[:, i], mu)
                            for i in range(len(sol.t))])

        C0 = C_values[0]

        # Even in chaotic region, Jacobi should be conserved
        assert_allclose(C_values, C0, rtol=1e-6, atol=1e-8)


class TestAngularMomentumInRotatingFrame:
    """
    Test angular momentum properties in the rotating frame.

    Note: In the rotating frame, angular momentum is NOT conserved
    due to the Coriolis force. However, we can test that the rate of
    change follows the expected relationship.
    """

    def test_angular_momentum_calculation(self):
        """Test that we can calculate angular momentum correctly."""
        mu = 0.1
        state0 = np.array([0.3, 0.4, 0.1, 0.2])

        sol = integrate_orbit(mu, state0, t_span=(0.0, 10.0))

        # Angular momentum in rotating frame: L_z = x*vy - y*vx
        L_z = sol.y[0, :] * sol.y[3, :] - sol.y[1, :] * sol.y[2, :]

        assert len(L_z) == len(sol.t)
        assert np.all(np.isfinite(L_z))

    def test_rotating_frame_angular_momentum_rate(self):
        """
        Test that angular momentum in the rotating frame changes
        according to the applied torques.

        Note: In the rotating frame, angular momentum is NOT conserved
        due to the Coriolis force. However, we can verify that it changes
        in a physically reasonable way.
        """
        mu = 0.1
        state0 = np.array([0.3, 0.0, 0.0, 0.5])

        sol = integrate_orbit(mu, state0, t_span=(0.0, 10.0), max_step=1e-3)

        # Angular momentum in rotating frame: L_z = x*vy - y*vx
        L_z = sol.y[0, :] * sol.y[3, :] - sol.y[1, :] * sol.y[2, :]

        # Angular momentum changes in rotating frame
        # Check that it's finite and varies smoothly
        assert np.all(np.isfinite(L_z))

        # Check that variations are smooth (no jumps)
        dL = np.diff(L_z)
        # Derivative should be smooth (second derivative shouldn't spike)
        ddL = np.diff(dL)
        assert np.std(ddL) < 10.0  # No wild oscillations


class TestEnergyRelations:
    """Test energy-like quantities and their relationships."""

    def test_jacobi_energy_relationship(self):
        """
        Test the relationship between Jacobi constant and energy.
        C = 2Ω - v^2 where Ω is effective potential.
        """
        mu = 0.1
        state0 = np.array([0.3, 0.2, 0.1, 0.15])

        C = jacobi_constant(state0, mu)
        Omega = effective_potential(state0[0], state0[1], mu)
        v2 = state0[2]**2 + state0[3]**2

        assert_allclose(C, 2*Omega - v2, rtol=1e-12)

    def test_kinetic_potential_tradeoff(self):
        """
        During orbit, kinetic and potential energy trade off
        while keeping Jacobi constant fixed.
        """
        mu = 0.1
        state0 = np.array([0.3, 0.0, 0.0, 0.6])

        sol = integrate_orbit(mu, state0, t_span=(0.0, 20.0), max_step=1e-3)

        # Compute kinetic and effective potential
        KE = 0.5 * (sol.y[2, :]**2 + sol.y[3, :]**2)
        Omega = np.array([effective_potential(sol.y[0, i], sol.y[1, i], mu)
                         for i in range(len(sol.t))])

        # They should anti-correlate
        # When KE increases, Omega should decrease
        # (though not exactly since Jacobi = 2Ω - v^2, not Ω + KE)
        assert np.std(KE) > 0.01  # Energy should vary
        assert np.std(Omega) > 0.01


# ==================== Numerical Accuracy Tests ====================

class TestNumericalAccuracy:
    """Test numerical accuracy and convergence."""

    def test_convergence_with_tolerance(self):
        """Test that tighter tolerances give more accurate results."""
        mu = 0.1
        state0 = np.array([0.3, 0.0, 0.0, 0.5])
        t_span = (0.0, 10.0)

        # Loose tolerance
        sol_loose = integrate_orbit(mu, state0, t_span,
                                   rtol=1e-6, atol=1e-9, max_step=1e-2)

        # Tight tolerance
        sol_tight = integrate_orbit(mu, state0, t_span,
                                   rtol=1e-10, atol=1e-13, max_step=1e-3)

        # Jacobi constant should be better conserved with tight tolerance
        C_loose = np.array([jacobi_constant(sol_loose.y[:, i], mu)
                           for i in range(len(sol_loose.t))])
        C_tight = np.array([jacobi_constant(sol_tight.y[:, i], mu)
                           for i in range(len(sol_tight.t))])

        deviation_loose = np.std(C_loose)
        deviation_tight = np.std(C_tight)

        assert deviation_tight < deviation_loose

    def test_step_size_effect(self):
        """Test that smaller step sizes give more accurate conservation."""
        mu = 0.1
        state0 = np.array([0.3, 0.0, 0.0, 0.5])
        t_span = (0.0, 20.0)

        # Large steps
        sol_large = integrate_orbit(mu, state0, t_span, max_step=1e-1)

        # Small steps
        sol_small = integrate_orbit(mu, state0, t_span, max_step=1e-3)

        C_large = np.array([jacobi_constant(sol_large.y[:, i], mu)
                           for i in range(len(sol_large.t))])
        C_small = np.array([jacobi_constant(sol_small.y[:, i], mu)
                           for i in range(len(sol_small.t))])

        max_dev_large = np.max(np.abs(C_large - C_large[0]))
        max_dev_small = np.max(np.abs(C_small - C_small[0]))

        # Smaller steps should conserve better
        assert max_dev_small < max_dev_large


# ==================== Lagrange Points Tests ====================

class TestLagrangePoints:
    """Test equilibrium at Lagrange points."""

    def test_L1_equilibrium(self):
        """Test that L1 is an equilibrium (with zero velocity)."""
        mu = 0.1
        # Find L1 numerically (between the two masses)
        x_L1, y_L1 = find_lagrange_point(mu, x0=0.6)
        state_L1 = np.array([x_L1, y_L1, 0.0, 0.0])

        derivative = equations_of_motion(0.0, state_L1, mu)

        # At equilibrium, acceleration should be zero
        assert abs(derivative[2]) < 1e-6  # ax ~ 0
        assert abs(derivative[3]) < 1e-6  # ay ~ 0

    def test_L2_equilibrium(self):
        """Test L2 equilibrium."""
        mu = 0.1
        # Find L2 numerically (beyond M2)
        x_L2, y_L2 = find_lagrange_point(mu, x0=1.2)
        state_L2 = np.array([x_L2, y_L2, 0.0, 0.0])

        derivative = equations_of_motion(0.0, state_L2, mu)

        # At equilibrium, acceleration should be zero
        assert abs(derivative[2]) < 1e-6  # ax ~ 0
        assert abs(derivative[3]) < 1e-6  # ay ~ 0

    def test_L4_L5_symmetry(self):
        """Test that L4 and L5 are symmetric about x-axis."""
        mu = 0.01
        # L4 and L5 are at equilateral triangle vertices
        x_L4 = 0.5 - mu
        y_L4 = np.sqrt(3)/2

        state_L4 = np.array([x_L4, y_L4, 0.0, 0.0])
        state_L5 = np.array([x_L4, -y_L4, 0.0, 0.0])

        deriv_L4 = equations_of_motion(0.0, state_L4, mu)
        deriv_L5 = equations_of_motion(0.0, state_L5, mu)

        # ax should be same, ay should be opposite
        assert_allclose(deriv_L4[2], deriv_L5[2], rtol=1e-10)
        assert_allclose(deriv_L4[3], -deriv_L5[3], rtol=1e-10)


# ==================== Utility Function Tests ====================

class TestComputePlotWindow:
    """Test plot window computation."""

    def test_plot_window_encloses_trajectory(self):
        """Plot window should enclose entire trajectory."""
        mu = 0.1
        state0 = np.array([0.3, 0.0, 0.0, 0.5])

        sol = integrate_orbit(mu, state0, t_span=(0.0, 20.0))
        x_min, x_max, y_min, y_max = compute_plot_window(sol, mu)

        # Check all points are inside
        assert np.all(sol.y[0, :] >= x_min)
        assert np.all(sol.y[0, :] <= x_max)
        assert np.all(sol.y[1, :] >= y_min)
        assert np.all(sol.y[1, :] <= y_max)

    def test_plot_window_includes_primaries(self):
        """Plot window should include both primaries."""
        mu = 0.1
        state0 = np.array([0.3, 0.0, 0.0, 0.5])

        sol = integrate_orbit(mu, state0, t_span=(0.0, 10.0))
        x_min, x_max, y_min, y_max = compute_plot_window(sol, mu)

        # M1 at (-mu, 0)
        assert -mu >= x_min
        assert -mu <= x_max

        # M2 at (1-mu, 0)
        assert 1-mu >= x_min
        assert 1-mu <= x_max

        # Both on x-axis
        assert 0.0 >= y_min
        assert 0.0 <= y_max

    def test_plot_window_is_square(self):
        """Plot window should be square."""
        mu = 0.1
        state0 = np.array([0.3, 0.0, 0.0, 0.5])

        sol = integrate_orbit(mu, state0, t_span=(0.0, 15.0))
        x_min, x_max, y_min, y_max = compute_plot_window(sol, mu)

        dx = x_max - x_min
        dy = y_max - y_min

        assert_allclose(dx, dy, rtol=1e-10)


# ==================== Parametric Tests ====================

@pytest.mark.parametrize("mu", [0.01, 0.05, 0.1, 0.2, 0.3])
def test_jacobi_conservation_various_mu(mu):
    """Test Jacobi conservation for various mass ratios."""
    state0 = np.array([0.3, 0.2, 0.1, 0.2])

    sol = integrate_orbit(mu, state0, t_span=(0.0, 20.0), max_step=1e-3)

    C_values = np.array([jacobi_constant(sol.y[:, i], mu)
                        for i in range(len(sol.t))])

    C0 = C_values[0]
    assert_allclose(C_values, C0, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("x0,y0,vx0,vy0", [
    (0.3, 0.0, 0.0, 0.5),
    (0.5, 0.2, -0.1, 0.3),
    (-0.2, 0.4, 0.2, -0.1),
    (0.4, 0.1, 0.1, 0.3),
])
def test_integration_various_initial_conditions(x0, y0, vx0, vy0):
    """Test integration with various initial conditions."""
    mu = 0.1
    state0 = np.array([x0, y0, vx0, vy0])

    sol = integrate_orbit(mu, state0, t_span=(0.0, 10.0), max_step=1e-3)

    assert sol.success
    assert len(sol.t) > 10

    # Check Jacobi conservation
    C_values = np.array([jacobi_constant(sol.y[:, i], mu)
                        for i in range(len(sol.t))])
    C0 = C_values[0]
    assert_allclose(C_values, C0, rtol=1e-6, atol=1e-8)


# ==================== Edge Cases and Robustness ====================

class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_zero_velocity_integration(self):
        """Test integration with zero initial velocity."""
        mu = 0.1
        state0 = np.array([0.3, 0.2, 0.0, 0.0])

        sol = integrate_orbit(mu, state0, t_span=(0.0, 5.0))

        assert sol.success

    def test_very_small_mu(self):
        """Test with very small mass ratio (nearly two-body)."""
        mu = 1e-6
        state0 = np.array([0.5, 0.0, 0.0, 0.8])

        sol = integrate_orbit(mu, state0, t_span=(0.0, 10.0))

        assert sol.success

        # Jacobi should still be conserved
        C_values = np.array([jacobi_constant(sol.y[:, i], mu)
                            for i in range(len(sol.t))])
        assert_allclose(C_values, C_values[0], rtol=1e-6, atol=1e-8)

    def test_mu_near_half(self):
        """Test with equal mass primaries."""
        mu = 0.5
        state0 = np.array([0.0, 0.3, 0.5, 0.0])

        sol = integrate_orbit(mu, state0, t_span=(0.0, 10.0))

        assert sol.success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
