#!/usr/bin/env python3
# Circular Restricted Three-Body Problem Solver and Animator (Refactored)
#
# This script solves the CR3BP in the uniformly rotating frame and then:
#   1) Computes a tight, square plot window that encloses the *entire*
#      integrated trajectory (plus the two primaries).
#   2) Uses that window for both the static figure and the animation, so
#      nothing runs out of bounds.
#   3) Picks contour levels from quantiles to avoid the deep singular wells.
#
# Scaling: dimensionless (G(M1+M2)=1, separation a=1, rotation rate Ω=1).
#
# Author: ChatGPT
# License: MIT
#
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp


# ------------------------------ Physics core ---------------------------------
def radii(x, y, mu, eps=1e-12):
    """Distances from (x,y) to the two primaries in the rotating frame.
    M1 at (-mu, 0), M2 at (1-mu, 0).
    Returns r1, r2.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x1 = x + mu          # vector from M1 (-mu,0) to (x,y)
    y1 = y
    x2 = x - (1.0 - mu)  # vector from M2 (1-mu,0) to (x,y)
    y2 = y
    r1 = np.sqrt(x1*x1 + y1*y1) + eps
    r2 = np.sqrt(x2*x2 + y2*y2) + eps
    return r1, r2


def effective_potential(x, y, mu):
    """Effective potential Ω(x,y) for the CR3BP in the rotating frame.
    Ω = 1/2 (x^2 + y^2) + (1-μ)/r1 + μ/r2
    """
    r1, r2 = radii(x, y, mu)
    return 0.5 * (x*x + y*y) + (1.0 - mu)/r1 + mu/r2


def equations_of_motion(t, state, mu):
    """CR3BP equations in the rotating frame. state = [x, y, vx, vy]."""
    x, y, vx, vy = state
    r1, r2 = radii(x, y, mu)

    # Accelerations (standard rotating-frame CR3BP)
    ax = 2.0*vy + x - (1.0 - mu)*(x + mu)/r1**3 - mu*(x - (1.0 - mu))/r2**3
    ay = -2.0*vx + y - (1.0 - mu)*y/r1**3 - mu*y/r2**3

    return np.array([vx, vy, ax, ay], dtype=float)


def jacobi_constant(state, mu):
    """Jacobi constant C = 2Ω - v^2 evaluated from a single state vector."""
    x, y, vx, vy = state
    Omega = effective_potential(x, y, mu)
    v2 = vx*vx + vy*vy
    return 2.0*Omega - v2


# ------------------------------ Integration ----------------------------------
def integrate_orbit(mu, state0, t_span=(0.0, 20.0), max_step=1e-2, rtol=1e-9, atol=1e-12):
    """Integrate the CR3BP equations with solve_ivp."""
    fun = lambda t, y: equations_of_motion(t, y, mu)
    sol = solve_ivp(fun, t_span, np.array(state0, dtype=float),
                    method="DOP853", max_step=max_step, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return sol


# ------------------------------ Plot window ----------------------------------
def compute_plot_window(solution, mu, pad_frac=0.08):
    """Compute a square plot window enclosing the entire trajectory and primaries.
    Returns (x_min, x_max, y_min, y_max).
    """
    x = np.asarray(solution.y[0])
    y = np.asarray(solution.y[1])

    # Include primaries
    x1, y1 = -mu, 0.0
    x2, y2 = 1.0 - mu, 0.0

    xs = np.concatenate([x, [x1, x2]])
    ys = np.concatenate([y, [y1, y2]])

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    # Square box
    dx = xmax - xmin
    dy = ymax - ymin
    span = max(dx, dy, 1e-9)
    cx = 0.5*(xmin + xmax)
    cy = 0.5*(ymin + ymax)

    pad = pad_frac * span
    half = 0.5*span + pad
    return (cx - half, cx + half, cy - half, cy + half)


# ------------------------------ Visualization --------------------------------
def _potential_contours(ax, mu, x_min, x_max, y_min, y_max, nxy=400):
    xr = np.linspace(x_min, x_max, nxy)
    yr = np.linspace(y_min, y_max, nxy)
    X, Y = np.meshgrid(xr, yr)
    Z = effective_potential(X, Y, mu)

    # Choose safe contour levels
    z_lo = np.nanquantile(Z, 0.02)
    z_hi = np.nanquantile(Z, 0.85)
    levels = np.linspace(z_lo, z_hi, 28)
    cs = ax.contour(X, Y, Z, levels=levels, alpha=0.4)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%1.2f')


def plot_static_orbit(solution, mu, filename="cr3bp_orbit.png"):
    """Static plot of the full orbit with bounds sized to the trajectory + primaries."""
    x_min, x_max, y_min, y_max = compute_plot_window(solution, mu, pad_frac=0.08)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Potential
    _potential_contours(ax, mu, x_min, x_max, y_min, y_max, nxy=400)

    # Primaries
    x1, y1 = -mu, 0.0
    x2, y2 = 1.0 - mu, 0.0
    ax.scatter([x1], [y1], s=200*(1-mu), marker='o', edgecolors='black', linewidths=1.2, zorder=5)
    ax.scatter([x2], [y2], s=200*mu,     marker='o', edgecolors='black', linewidths=1.2, zorder=5)

    # Trajectory
    ax.plot(solution.y[0], solution.y[1], linewidth=1.6, alpha=0.8, label="Trajectory", zorder=4)
    ax.plot(solution.y[0, 0], solution.y[1, 0], 'o', ms=8, label="Start", zorder=6)

    C = jacobi_constant(solution.y[:, 0], mu)
    ax.set_title(f"CR3BP (rotating frame) — q = {mu/(1-mu):.3f},  C = {C:.5f}", fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=160, bbox_inches="tight")
    print(f"[static] saved {filename}")
    plt.close(fig)


def create_animation(solution, mu, filename="cr3bp_animation.mp4", trail_length=200):
    """Animate the orbit using bounds computed from the full trajectory."""
    x_min, x_max, y_min, y_max = compute_plot_window(solution, mu, pad_frac=0.08)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Potential
    _potential_contours(ax, mu, x_min, x_max, y_min, y_max, nxy=350)

    # Primaries
    x1, y1 = -mu, 0.0
    x2, y2 = 1.0 - mu, 0.0
    ax.scatter([x1], [y1], s=200*(1-mu), marker='o', edgecolors='black', linewidths=1.2, zorder=5)
    ax.scatter([x2], [y2], s=200*mu,     marker='o', edgecolors='black', linewidths=1.2, zorder=5)

    # Trajectory handles
    trail, = ax.plot([], [], linewidth=1.6, alpha=0.8, label="Trajectory")
    dot,   = ax.plot([], [], 'o', ms=8, zorder=6, label="Current position")

    C = jacobi_constant(solution.y[:, 0], mu)
    ax.set_title(f"CR3BP (rotating frame) — q = {mu/(1-mu):.3f},  C = {C:.5f}", fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    n = solution.y.shape[1]
    frames = np.arange(n)

    def update(k):
        i0 = max(0, k - trail_length)
        x = solution.y[0, i0:k+1]
        y = solution.y[1, i0:k+1]
        trail.set_data(x, y)
        dot.set_data([solution.y[0, k]], [solution.y[1, k]])
        return trail, dot

    print(f"[anim] saving {filename} ... (requires ffmpeg)")
    anim = FuncAnimation(fig, update, frames=frames, interval=20, blit=True)
    try:
        anim.save(filename, writer="ffmpeg", fps=30, dpi=100)
        print(f"[anim] saved {filename}")
    except Exception as e:
        print(f"[anim] could not save animation: {e}")
    plt.close(fig)


# ------------------------------ Inertial Frame Transformation ----------------
def rotating_to_inertial(x_rot, y_rot, t):
    """
    Transform coordinates from rotating frame to inertial frame.

    At time t, the rotating frame has rotated by angle θ = Ω*t = t (since Ω=1).
    Rotation matrix: R(θ) = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]

    Args:
        x_rot: x position(s) in rotating frame
        y_rot: y position(s) in rotating frame
        t: time or array of times

    Returns:
        x_inertial, y_inertial: positions in inertial frame
    """
    cos_t = np.cos(t)
    sin_t = np.sin(t)

    x_inertial = x_rot * cos_t - y_rot * sin_t
    y_inertial = x_rot * sin_t + y_rot * cos_t

    return x_inertial, y_inertial


def plot_static_orbit_inertial(solution, mu, filename="cr3bp_orbit_inertial.png"):
    """
    Static plot of the full orbit in the inertial frame.
    In the inertial frame, the two primaries rotate around the barycenter.
    No effective potential is shown (it only exists in the rotating frame).
    """
    # Transform particle trajectory to inertial frame
    x_rot = solution.y[0]
    y_rot = solution.y[1]
    t = solution.t

    x_inertial, y_inertial = rotating_to_inertial(x_rot, y_rot, t)

    # Compute plot window in inertial frame
    # Include primaries at their maximum extent (radius = separation from barycenter)
    r1 = mu  # M1 orbits at radius mu
    r2 = 1.0 - mu  # M2 orbits at radius 1-mu
    max_r = max(r1, r2)

    x_min = min(x_inertial.min(), -max_r) - 0.1
    x_max = max(x_inertial.max(), max_r) + 0.1
    y_min = min(y_inertial.min(), -max_r) - 0.1
    y_max = max(y_inertial.max(), max_r) + 0.1

    # Make square
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    span = max(x_max - x_min, y_max - y_min)
    half = 0.5 * span * 1.08

    x_min, x_max = cx - half, cx + half
    y_min, y_max = cy - half, cy + half

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the circular orbits of the primaries
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(r1 * np.cos(theta), r1 * np.sin(theta), '--',
            color='gray', alpha=0.3, linewidth=1, label='M1 orbit')
    ax.plot(r2 * np.cos(theta), r2 * np.sin(theta), '--',
            color='gray', alpha=0.3, linewidth=1, label='M2 orbit')

    # Plot initial positions of primaries (at t=0)
    x1_0, y1_0 = rotating_to_inertial(-mu, 0.0, 0.0)
    x2_0, y2_0 = rotating_to_inertial(1.0 - mu, 0.0, 0.0)

    ax.scatter([x1_0], [y1_0], s=200*(1-mu), marker='o',
               edgecolors='black', linewidths=1.2, zorder=5, label='M1 (t=0)')
    ax.scatter([x2_0], [y2_0], s=200*mu, marker='o',
               edgecolors='black', linewidths=1.2, zorder=5, label='M2 (t=0)')

    # Plot final positions of primaries (at t=tmax)
    x1_f, y1_f = rotating_to_inertial(-mu, 0.0, t[-1])
    x2_f, y2_f = rotating_to_inertial(1.0 - mu, 0.0, t[-1])

    ax.scatter([x1_f], [y1_f], s=200*(1-mu), marker='s',
               edgecolors='black', linewidths=1.2, zorder=5, alpha=0.5, label='M1 (t=end)')
    ax.scatter([x2_f], [y2_f], s=200*mu, marker='s',
               edgecolors='black', linewidths=1.2, zorder=5, alpha=0.5, label='M2 (t=end)')

    # Particle trajectory in inertial frame
    ax.plot(x_inertial, y_inertial, linewidth=1.6, alpha=0.8,
            label="Trajectory (inertial)", zorder=4)
    ax.plot(x_inertial[0], y_inertial[0], 'o', ms=8, label="Start", zorder=6)

    # Barycenter at origin
    ax.scatter([0], [0], s=50, marker='x', color='black', zorder=5, label='Barycenter')

    C = jacobi_constant(solution.y[:, 0], mu)
    ax.set_title(f"CR3BP (inertial frame) — q = {mu/(1-mu):.3f},  C = {C:.5f}", fontsize=14)
    ax.set_xlabel("x (inertial)")
    ax.set_ylabel("y (inertial)")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=160, bbox_inches="tight")
    print(f"[static inertial] saved {filename}")
    plt.close(fig)


def create_animation_inertial(solution, mu, filename="cr3bp_animation_inertial.mp4", trail_length=200):
    """
    Animate the orbit in the inertial frame with rotating primaries.
    """
    # Transform trajectory to inertial frame
    x_rot = solution.y[0]
    y_rot = solution.y[1]
    t = solution.t

    x_inertial, y_inertial = rotating_to_inertial(x_rot, y_rot, t)

    # Compute plot window
    r1 = mu
    r2 = 1.0 - mu
    max_r = max(r1, r2)

    x_min = min(x_inertial.min(), -max_r) - 0.1
    x_max = max(x_inertial.max(), max_r) + 0.1
    y_min = min(y_inertial.min(), -max_r) - 0.1
    y_max = max(y_inertial.max(), max_r) + 0.1

    # Make square
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    span = max(x_max - x_min, y_max - y_min)
    half = 0.5 * span * 1.08

    x_min, x_max = cx - half, cx + half
    y_min, y_max = cy - half, cy + half

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot circular orbits
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(r1 * np.cos(theta), r1 * np.sin(theta), '--',
            color='gray', alpha=0.3, linewidth=1)
    ax.plot(r2 * np.cos(theta), r2 * np.sin(theta), '--',
            color='gray', alpha=0.3, linewidth=1)

    # Barycenter
    ax.scatter([0], [0], s=50, marker='x', color='black', zorder=5)

    # Animation elements
    trail, = ax.plot([], [], linewidth=1.6, alpha=0.8, label="Trajectory")
    dot, = ax.plot([], [], 'o', ms=8, zorder=6, label="Particle")
    m1_marker = ax.scatter([], [], s=200*(1-mu), marker='o',
                          edgecolors='black', linewidths=1.2, zorder=5, label='M1')
    m2_marker = ax.scatter([], [], s=200*mu, marker='o',
                          edgecolors='black', linewidths=1.2, zorder=5, label='M2')

    C = jacobi_constant(solution.y[:, 0], mu)
    ax.set_title(f"CR3BP (inertial frame) — q = {mu/(1-mu):.3f},  C = {C:.5f}", fontsize=14)
    ax.set_xlabel("x (inertial)")
    ax.set_ylabel("y (inertial)")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    n = len(t)
    frames = np.arange(n)

    def update(k):
        # Update particle trail and position
        i0 = max(0, k - trail_length)
        x = x_inertial[i0:k+1]
        y = y_inertial[i0:k+1]
        trail.set_data(x, y)
        dot.set_data([x_inertial[k]], [y_inertial[k]])

        # Update primary positions
        x1_k, y1_k = rotating_to_inertial(-mu, 0.0, t[k])
        x2_k, y2_k = rotating_to_inertial(1.0 - mu, 0.0, t[k])
        m1_marker.set_offsets([[x1_k, y1_k]])
        m2_marker.set_offsets([[x2_k, y2_k]])

        return trail, dot, m1_marker, m2_marker

    print(f"[anim inertial] saving {filename} ... (requires ffmpeg)")
    anim = FuncAnimation(fig, update, frames=frames, interval=20, blit=True)
    try:
        anim.save(filename, writer="ffmpeg", fps=30, dpi=100)
        print(f"[anim inertial] saved {filename}")
    except Exception as e:
        print(f"[anim inertial] could not save animation: {e}")
    plt.close(fig)


# ------------------------------ CLI / main -----------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="CR3BP solver + plots with auto-bounded axes.")
    p.add_argument("--mu", type=float, default=0.1, help="Reduced mass parameter μ in [0,1].")
    p.add_argument("--x0", type=float, default=0.3, help="Initial x")
    p.add_argument("--y0", type=float, default=0.0, help="Initial y")
    p.add_argument("--vx0", type=float, default=0.0, help="Initial vx")
    p.add_argument("--vy0", type=float, default=0.5, help="Initial vy")
    p.add_argument("--tmax", type=float, default=40.0, help="Final time")
    p.add_argument("--max-step", type=float, default=5e-3, help="Max step for integrator")
    p.add_argument("--no-static", action="store_true", help="Skip static PNG output")
    p.add_argument("--no-anim", action="store_true", help="Skip MP4 animation output")
    p.add_argument("--trail", type=int, default=300, help="Trail length for animation (in steps)")
    p.add_argument("--inertial", action="store_true", help="Plot in inertial frame (rotating primaries)")
    return p.parse_args()


def main():
    args = parse_args()

    mu = float(args.mu)
    if not (0.0 <= mu <= 1.0):
        raise ValueError("mu must be in [0,1]")

    state0 = np.array([args.x0, args.y0, args.vx0, args.vy0], dtype=float)

    frame_type = "inertial" if args.inertial else "rotating"
    print(f"Integrating with mu={mu:.6f}, state0={state0}, tmax={args.tmax} ...")
    print(f"Output frame: {frame_type}")

    sol = integrate_orbit(mu, state0, t_span=(0.0, args.tmax), max_step=args.max_step)

    # Compute and echo Jacobi constant at t0
    C0 = jacobi_constant(state0, mu)
    print(f"Jacobi constant at t0: C0 = {C0:.8f}")

    if args.inertial:
        # Inertial frame outputs
        if not args.no_static:
            plot_static_orbit_inertial(sol, mu, filename="cr3bp_orbit_inertial.png")

        if not args.no_anim:
            create_animation_inertial(sol, mu, filename="cr3bp_animation_inertial.mp4",
                                     trail_length=args.trail)
    else:
        # Rotating frame outputs (default)
        if not args.no_static:
            plot_static_orbit(sol, mu, filename="cr3bp_orbit.png")

        if not args.no_anim:
            create_animation(sol, mu, filename="cr3bp_animation.mp4", trail_length=args.trail)

    print("Done.")


if __name__ == "__main__":
    main()

