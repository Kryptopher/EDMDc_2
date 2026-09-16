import numpy as np
import random
from quadcopter import quadcopter
from Cascaded_Controllers import QuadPIDController6Fixed, QuadPX4LikeController
from Closed_loop import ClosedLoopQuad


ACC_BALANCED_PROFILE_CONFIG = {
    "version": "acc_balanced_waypoint_v2",
    "duration_seconds": 60.0,
    "ramp_duration_seconds": 5.0,
    "max_reference_speed_mps": 5.0,
    "max_reference_acceleration_mps2": 6.0,
    "helix_target_cruise_speed_mps": [1.0, 2.5],
    "figure8_size_m": [20.0, 30.0],
    "lissajous_xy_size_m": [12.0, 20.0],
    "waypoint_xy_range_m": [12.0, 20.0],
    "waypoint_max_reference_speed_mps": 3.5,
    "waypoint_max_reference_acceleration_mps2": 4.0,
    "start_ramp_families": ["waypoint", "hover_excitation"],
    "family_parameters": {
        "helix": {
            "radius_m": [3.0, 10.0],
            "z_end_m": [3.0, 10.0],
            "target_cruise_speed_mps": [1.0, 2.5],
            "turns": "solved_from_target_speed",
        },
        "figure8": {
            "a_m": [20.0, 30.0],
            "b_m": [20.0, 30.0],
            "loops": 1.0,
            "tilt_deg": [10.0, 80.0],
        },
        "lissajous": {
            "xy_amplitude_m": [12.0, 20.0],
            "z_amplitude_m": [3.0, 7.0],
            "center_z_m": [1.0, 4.0],
            "axis_frequencies": [1.0, 2.0, 3.0],
            "phase_y_z_rad": [0.0, "pi"],
        },
        "waypoint": {
            "count": [5, 15],
            "xy_range_m": [12.0, 20.0],
            "z_lower_m": 0.5,
            "z_upper_m": [3.0, 8.0],
            "interpolation": "natural_cubic_spline",
            "max_reference_speed_mps": 3.5,
            "max_reference_acceleration_mps2": 4.0,
        },
        "hover_excitation": {
            "xy_amplitude_m": [2.0, 4.0],
            "z_amplitude_m": [1.0, 2.0],
            "xy_base_frequency_hz": [0.05, 0.12],
            "z_base_frequency_hz": [0.06, 0.15],
            "yaw_amplitude_deg": [2.0, 8.0],
            "sine_count": [2, 4],
        },
        "yaw_prbs": {
            "yaw_rate_radps": [-0.8, 0.8],
            "hold_seconds_at_100hz": [0.4, 1.2],
            "seed_start": 7000,
        },
    },
}


class quad_sim:
    # Simulation Parameters
    q_mass = 3.33819 # kg
    g = 9.80665 # m/s^2
    q_l = 0.28881 # m
    kD = 4.8e-7 # aerodynamic drag/yaw torque factor
    kT = 3.44e-5 # N/(rad/s)^2

    # Drag estimates for a 5.5 inch cube center body.
    # Assumptions: rho=1.225 kg/m^3, Cd~=1.05 for a cube, linearized at
    # v_ref=15 m/s and omega_ref=220 deg/s to match this simulator's
    # F=-k*v and tau=-k*omega damping model.
    cube_side = 5.5 * 0.0254 # m
    k_drag_linear = 0.18826928071875 # kg/s
    k_drag_angular = 6.569729303413257e-5 # N*m*s/rad

    Ixx, Iyy, Izz = 0.04164, 0.03963, 0.04758
    I = np.diag([Ixx, Iyy, Izz])

    kp_pos = [2.0, 2.0, 15.] #[x,y,z]
    ki_pos = [0.2, 0.2, 5.] #[x,y,z]
    kd_pos = [3.2, 3.2, 15.] #[x,y,z]
    kp_ang = [6.5, 6.5, 2.8] #[phi,theta,psi]
    ki_ang = [0.1, 0.1, 0.1] #[phi,theta,psi]
    kd_ang = [3.7, 3.7, 5.] #[phi,theta,psi]

    max_speed = 16380.0 * 2.0 * np.pi / 60.0 # rad/s

    dt = 0.01

    def __init__(self):
        """Create an independent plant and controller for each simulation."""
        self.quad = quadcopter(
            self.q_mass, self.g, self.q_l, self.I.copy(), self.kD, self.kT,
            self.k_drag_linear, self.k_drag_angular,
            prop_efficiency=[1.0, 1.0, 1.0, 1.0],
        )
        self.controller_PID = QuadPIDController6Fixed(
            self.quad,
            self.kp_pos, self.ki_pos, self.kd_pos,
            self.kp_ang, self.ki_ang, self.kd_ang,
            max_speed=self.max_speed,
            a_xy_max=10.0,
            a_z_max=10.0,
            tilt_max_deg=45.0,
            torque_roll_pitch_max=1.10,
            yaw_tau_max=0.35,
        )
        self.controller_PX4 = QuadPX4LikeController(
            self.quad,
            max_speed=self.max_speed,
            pos_p=(1.0, 1.0, 2.0),
            vel_p=(3.0, 3.0, 5.0),
            vel_i=(0.08, 0.08, 1.0),
            att_p=(5.0, 5.0, 3.2),
            # These gains output physical torque units in this simulator.
            rate_p=(0.45, 0.45, 0.32),
            rate_sp_max=(np.deg2rad(220.0), np.deg2rad(220.0), np.deg2rad(200.0)),
            vel_sp_max_xy=15.0,
            vel_sp_max_z=15.0,
            acc_max_xy=10.0,
            acc_max_z=10.0,
            tilt_max_deg=45.0,
            thrust_max=self.q_mass * (self.g + 10.0),
            torque_max=(1.14, 1.09, 0.35),
        )
        self.sim_PID = ClosedLoopQuad(self.quad, self.controller_PX4)
        self.time = np.arange(0.0, 45.0, self.dt)
        self.last_requested_inputs = None
        self.last_applied_inputs = None
        self.last_outer_inputs = None

    def fct_unwrap_trajectory_yaw(self, traj):
        yaws = np.unwrap([r["yaw"] for r in traj])
        for r, yaw in zip(traj, yaws):
            r["yaw"] = float(yaw)
        return traj

    def fct_limit_trajectory_yaw_rate(self, traj, time, max_abs_yaw_rate=0.8):
        """Create a continuous, actuator-compatible yaw reference profile.

        A path tangent is undefined at a smooth start/stop and may require more
        yaw authority than the motor allocation can provide. The controller
        tracks yaw rate directly, so integrate a bounded rate reference while
        retaining the path's initial heading.
        """
        if max_abs_yaw_rate <= 0.0:
            raise ValueError("max_abs_yaw_rate must be positive")
        if len(traj) != len(time):
            raise ValueError("Trajectory and time vectors must have equal length")
        if not traj:
            return traj

        time = np.asarray(time, dtype=float)
        raw_rate = np.array([float(point.get("yaw_rate", 0.0)) for point in traj])
        yaw_rate = np.clip(raw_rate, -max_abs_yaw_rate, max_abs_yaw_rate)
        yaw = np.empty(len(traj), dtype=float)
        yaw[0] = float(traj[0]["yaw"])
        for k in range(1, len(traj)):
            yaw[k] = yaw[k - 1] + yaw_rate[k - 1] * (time[k] - time[k - 1])

        for point, yaw_value, yaw_rate_value in zip(traj, yaw, yaw_rate):
            point["yaw"] = float(yaw_value)
            point["yaw_rate"] = float(yaw_rate_value)
        return traj

    def fct_scale_trajectory_to_limits(
        self,
        traj,
        max_speed=5.0,
        max_acceleration=6.0,
    ):
        """Uniformly reduce path scale until reference kinematics are feasible.

        Position displacement, velocity, and acceleration are scaled by the
        same factor, preserving their derivative relationship and path shape.
        Heading and heading rate are unchanged because a positive uniform
        spatial scale does not change the path tangent direction.
        """
        if not traj:
            return traj
        if max_speed <= 0.0 or max_acceleration <= 0.0:
            raise ValueError("trajectory limits must be positive")

        peak_speed = max(float(np.linalg.norm(point["vel"])) for point in traj)
        peak_acceleration = max(
            float(np.linalg.norm(point["acc"])) for point in traj
        )
        scale = min(
            1.0,
            max_speed / max(peak_speed, 1e-12),
            max_acceleration / max(peak_acceleration, 1e-12),
        )
        if scale >= 1.0:
            return traj

        origin = np.asarray(traj[0]["pos"], dtype=float).copy()
        for point in traj:
            point["pos"] = origin + scale * (
                np.asarray(point["pos"], dtype=float) - origin
            )
            point["vel"] = scale * np.asarray(point["vel"], dtype=float)
            point["acc"] = scale * np.asarray(point["acc"], dtype=float)
        return traj

    def fct_ramp_trajectory_start(self, traj, time, ramp_duration_seconds=5.0):
        """Apply a derivative-consistent quintic startup ramp to a reference."""
        if not traj:
            return traj
        time = np.asarray(time, dtype=float)
        if len(traj) != len(time):
            raise ValueError("Trajectory and time vectors must have equal length")
        if ramp_duration_seconds <= 0.0:
            raise ValueError("ramp_duration_seconds must be positive")

        phase = np.clip(
            (time - time[0]) / float(ramp_duration_seconds), 0.0, 1.0
        )
        window = 6.0 * phase**5 - 15.0 * phase**4 + 10.0 * phase**3
        positions = np.asarray([point["pos"] for point in traj], dtype=float)
        origin = positions[0].copy()
        positions = origin + window[:, None] * (positions - origin)

        yaw = np.unwrap(np.asarray([point["yaw"] for point in traj], dtype=float))
        yaw = yaw[0] + window * (yaw - yaw[0])
        edge_order = 2 if len(time) >= 3 else 1
        velocities = np.gradient(positions, time, axis=0, edge_order=edge_order)
        accelerations = np.gradient(
            velocities, time, axis=0, edge_order=edge_order
        )
        yaw_rate = np.gradient(yaw, time, edge_order=edge_order)

        for k, point in enumerate(traj):
            point["pos"] = positions[k]
            point["vel"] = velocities[k]
            point["acc"] = accelerations[k]
            point["yaw"] = float(yaw[k])
            point["yaw_rate"] = float(yaw_rate[k])
        return self.fct_limit_trajectory_yaw_rate(traj, time)

    def fct_smooth_time_scaling(self, tau, T, ramp_duration_seconds=None):
        """Return normalized path distance, speed, and acceleration.

        ``None`` retains the paper profile's original 30%-of-run ramp exactly.
        ACC-balanced trajectories instead pass a fixed duration so shortening a
        run does not consume most of it in near-stationary startup/shutdown.
        """
        if ramp_duration_seconds is None:
            r = 0.3
        else:
            if ramp_duration_seconds <= 0.0:
                raise ValueError("ramp_duration_seconds must be positive")
            r = float(np.clip(ramp_duration_seconds / T, 1e-9, 0.49))
        cruise_scale = 1.0 / (1.0 - r)

        def ramp_distance(xi):
            return xi**3 - 0.5 * xi**4

        def ramp_speed(xi):
            return 3.0 * xi**2 - 2.0 * xi**3

        def ramp_accel(xi):
            return 6.0 * xi - 6.0 * xi**2

        if tau < r:
            xi = tau / r
            sigma = cruise_scale * r * ramp_distance(xi)
            sigma_dot = cruise_scale * ramp_speed(xi) / T
            sigma_ddot = cruise_scale * ramp_accel(xi) / (r * T**2)
        elif tau > 1.0 - r:
            tau_remaining = 1.0 - tau
            xi = tau_remaining / r
            sigma = 1.0 - cruise_scale * r * ramp_distance(xi)
            sigma_dot = cruise_scale * ramp_speed(xi) / T
            sigma_ddot = -cruise_scale * ramp_accel(xi) / (r * T**2)
        else:
            sigma = cruise_scale * (0.5 * r + tau - r)
            sigma_dot = cruise_scale / T
            sigma_ddot = 0.0
        return sigma, sigma_dot, sigma_ddot

    def fct_make_helical_trajectory(self, time,
                                    center=(0.0, 0.0),
                                    radius=1.0,
                                    z_start=0.5,
                                    z_end=3.0,
                                    n_turns=3.0,
                                    yaw_follows_path=True,
                                    ramp_duration_seconds=None):
        """
        Make a helical trajectory:
        - circle of given radius around (cx, cy)
        - altitude increases linearly from z_start to z_end
        - n_turns full revolutions over the full duration of 'time'
        - total trajectory duration is time[-1] - time[0]
        - constant speed along the path

        Returns a list of dicts with keys:
            "pos": np.array([x,y,z])
            "vel": np.array([vx,vy,vz])
            "yaw": float
        """

        time = np.asarray(time, dtype=float)
        t0 = float(time[0])
        T  = float(time[-1] - time[0])  # total duration

        if T <= 0.0:
            raise ValueError("time array must span a positive duration")

        cx, cy = float(center[0]), float(center[1])

        # Angle and altitude as functions of time
        # tau in [0,1]
        traj = []
        for t in time:
            tau = (t - t0) / T  # normalized time in [0,1]
            sigma, sigma_dot, sigma_ddot = self.fct_smooth_time_scaling(
                tau, T, ramp_duration_seconds=ramp_duration_seconds
            )

            # Angle (n_turns full revolutions)
            theta = 2.0 * np.pi * n_turns * sigma
            theta_dot = 2.0 * np.pi * n_turns * sigma_dot
            theta_ddot = 2.0 * np.pi * n_turns * sigma_ddot

            # Position
            x = cx + radius * np.cos(theta)
            y = cy + radius * np.sin(theta)
            z = z_start + (z_end - z_start) * sigma

            # Velocity (derivatives)
            vx = -radius * np.sin(theta) * theta_dot
            vy =  radius * np.cos(theta) * theta_dot
            vz = (z_end - z_start) * sigma_dot

            # Acceleration (derivatives)
            ax = -radius * (np.cos(theta) * theta_dot**2 + np.sin(theta) * theta_ddot)
            ay = radius * (-np.sin(theta) * theta_dot**2 + np.cos(theta) * theta_ddot)
            az = (z_end - z_start) * sigma_ddot

            # Yaw: either follow the tangent direction or stay fixed
            if yaw_follows_path:
                tangent_x = -radius * np.sin(theta)
                tangent_y = radius * np.cos(theta)
                yaw = np.arctan2(tangent_y, tangent_x)  # heading along the path
                yaw_rate = theta_dot
            else:
                yaw = 0.0  # or any constant you like
                yaw_rate = 0.0

            traj.append({
                "pos": np.array([x, y, z], dtype=float),
                "vel": np.array([vx, vy, vz], dtype=float),
                "acc": np.array([ax, ay, az], dtype=float),
                "yaw": float(yaw),
                "yaw_rate": float(yaw_rate)
            })

        traj = self.fct_unwrap_trajectory_yaw(traj)
        return self.fct_limit_trajectory_yaw_rate(traj, time)

    def fct_make_figure8_trajectory(self, time,
                                    center=(0.0, 0.0, 1.0),
                                    a=1.0,
                                    b=0.5,
                                    n_loops=1.0,
                                    tilt_deg=30.0,
                                    yaw_follows_path=True,
                                    yaw_constant=0.0,
                                    ramp_duration_seconds=None):
        """
        Make a 3D figure-8 trajectory.

        Base curve (before tilt) is a lemniscate of Gerono in the XY-plane:
            x' = a * sin(ω t)
            y' = b * sin(ω t) * cos(ω t) = 0.5*b*sin(2ω t)
            z' = 0

        Then we rotate that plane around the X-axis by `tilt_deg`, so z varies.

        Parameters
        ----------
        time : array-like
            Time stamps for the trajectory.
        center : (cx, cy, cz)
            Center of the figure-8 in world coordinates.
        a, b : float
            Horizontal/vertical scales of the figure-8.
        n_loops : float
            Number of figure-8 loops over the full time interval.
        tilt_deg : float
            Tilt angle (degrees) around the X-axis. 0° -> flat in XY, z = const.
        yaw_follows_path : bool
            If True, yaw is aligned with the XY projection of the velocity.
            If False, yaw is constant (yaw_constant).
        yaw_constant : float
            Constant yaw (rad) if yaw_follows_path is False.

        Returns
        -------
        traj : list of dict
            Each element has keys "pos", "vel", "yaw".
        """
        time = np.asarray(time, dtype=float)
        t0 = float(time[0])
        T  = float(time[-1] - time[0])
        if T <= 0.0:
            raise ValueError("time array must span a positive duration")

        cx, cy, cz = map(float, center)

        # Angular frequency to get n_loops over duration T
        omega = 2.0 * np.pi * n_loops / T

        # Rotation about x-axis
        tilt = np.deg2rad(tilt_deg)
        cth = np.cos(tilt)
        sth = np.sin(tilt)

        traj = []
        for t in time:
            tau = (t - t0) / T
            sigma, sigma_dot, sigma_ddot = self.fct_smooth_time_scaling(
                tau, T, ramp_duration_seconds=ramp_duration_seconds
            )

            # ---- base planar figure-8 (XY plane) ----
            s    = 2.0 * np.pi * n_loops * sigma - 0.25 * np.pi
            s_dot = 2.0 * np.pi * n_loops * sigma_dot
            s_ddot = 2.0 * np.pi * n_loops * sigma_ddot
            sin_s = np.sin(s)
            cos_s = np.cos(s)

            # Position in local (unrotated) frame
            x_local = a * sin_s
            y_local = b * sin_s * cos_s   # 0.5*b*sin(2s)
            z_local = 0.0

            # Velocity in local frame (time derivatives)
            dx_local = a * cos_s * s_dot
            # derivative of b*sin(s)*cos(s) = b*omega*(cos^2 - sin^2) = b*omega*cos(2s)
            dy_local = b * (cos_s**2 - sin_s**2) * s_dot
            dz_local = 0.0

            ddx_local = a * (cos_s * s_ddot - sin_s * s_dot**2)
            ddy_local = b * ((cos_s**2 - sin_s**2) * s_ddot - 4.0 * sin_s * cos_s * s_dot**2)
            ddz_local = 0.0

            # ---- rotate around X-axis to introduce z-variation ----
            # x' = x
            # y' =  y*cos(tilt) - z*sin(tilt) = y*cos(tilt)
            # z' =  y*sin(tilt) + z*cos(tilt) = y*sin(tilt)
            x_world = x_local
            y_world = y_local * cth
            z_world = y_local * sth

            dx_world = dx_local
            dy_world = dy_local * cth
            dz_world = dy_local * sth

            ddx_world = ddx_local
            ddy_world = ddy_local * cth
            ddz_world = ddy_local * sth

            # ---- shift to center ----
            x = cx + x_world
            y = cy + y_world
            z = cz + z_world

            vx = dx_world
            vy = dy_world
            vz = dz_world

            ax = ddx_world
            ay = ddy_world
            az = ddz_world

            # ---- yaw ----
            if yaw_follows_path:
                # Heading in the XY plane
                tangent_x = a * cos_s
                tangent_y = b * (cos_s**2 - sin_s**2) * cth
                yaw = np.arctan2(tangent_y, tangent_x)
                yaw_rate = (vx * ay - vy * ax) / (vx**2 + vy**2 + 1e-12)
            else:
                yaw = float(yaw_constant)
                yaw_rate = 0.0

            traj.append({
                "pos": np.array([x, y, z], dtype=float),
                "vel": np.array([vx, vy, vz], dtype=float),
                "acc": np.array([ax, ay, az], dtype=float),
                "yaw": float(yaw),
                "yaw_rate": float(yaw_rate)
            })

        traj = self.fct_unwrap_trajectory_yaw(traj)
        return self.fct_limit_trajectory_yaw_rate(traj, time)

    def fct_make_lissajous_trajectory(self, time,
                                      center=(0.0, 0.0, 0.0),
                                      ax=2.0,
                                      ay=2.0,
                                      az=1.0,
                                      fx=1.0,
                                      fy=2.0,
                                      fz=3.0,
                                      phase_y=np.pi / 2.0,
                                      phase_z=np.pi / 4.0,
                                      harmonic_scale=0.0,
                                      harmonic_phase_x=0.0,
                                      harmonic_phase_y=0.0,
                                      yaw_follows_path=True,
                                      yaw_constant=0.0,
                                      ramp_duration_seconds=None):
        """
        Make a 3D Lissajous trajectory with smooth start/stop timing.

        The curve is parameterized as:
            x = ax * sin(fx*s)
            y = ay * sin(fy*s + phase_y)
            z = az * sin(fz*s + phase_z)

        where s is smoothly swept from 0 to 2*pi over the simulation.
        """
        time = np.asarray(time, dtype=float)
        t0 = float(time[0])
        T = float(time[-1] - time[0])
        if T <= 0.0:
            raise ValueError("time array must span a positive duration")

        cx, cy, cz = map(float, center)
        traj = []

        for t in time:
            tau = (t - t0) / T
            sigma, sigma_dot, sigma_ddot = self.fct_smooth_time_scaling(
                tau, T, ramp_duration_seconds=ramp_duration_seconds
            )

            s = 2.0 * np.pi * sigma
            s_dot = 2.0 * np.pi * sigma_dot
            s_ddot = 2.0 * np.pi * sigma_ddot

            sx = fx * s
            sy = fy * s + phase_y
            sz = fz * s + phase_z
            sx2 = (fx + 1.0) * s + harmonic_phase_x
            sy2 = (fy + 1.0) * s + harmonic_phase_y

            x = cx + ax * (np.sin(sx) + harmonic_scale * np.sin(sx2))
            y = cy + ay * (np.sin(sy) + harmonic_scale * np.sin(sy2))
            z = cz + az * np.sin(sz)

            vx = ax * (
                fx * np.cos(sx)
                + harmonic_scale * (fx + 1.0) * np.cos(sx2)
            ) * s_dot
            vy = ay * (
                fy * np.cos(sy)
                + harmonic_scale * (fy + 1.0) * np.cos(sy2)
            ) * s_dot
            vz = az * fz * np.cos(sz) * s_dot

            ax_w = ax * (
                fx * np.cos(sx) * s_ddot
                - fx**2 * np.sin(sx) * s_dot**2
                + harmonic_scale * (fx + 1.0) * np.cos(sx2) * s_ddot
                - harmonic_scale * (fx + 1.0)**2 * np.sin(sx2) * s_dot**2
            )
            ay_w = ay * (
                fy * np.cos(sy) * s_ddot
                - fy**2 * np.sin(sy) * s_dot**2
                + harmonic_scale * (fy + 1.0) * np.cos(sy2) * s_ddot
                - harmonic_scale * (fy + 1.0)**2 * np.sin(sy2) * s_dot**2
            )
            az_w = az * fz * (np.cos(sz) * s_ddot - fz * np.sin(sz) * s_dot**2)

            if yaw_follows_path:
                yaw = np.arctan2(vy, vx)
                yaw_rate = (vx * ay_w - vy * ax_w) / (vx**2 + vy**2 + 1e-12)
            else:
                yaw = float(yaw_constant)
                yaw_rate = 0.0

            traj.append({
                "pos": np.array([x, y, z], dtype=float),
                "vel": np.array([vx, vy, vz], dtype=float),
                "acc": np.array([ax_w, ay_w, az_w], dtype=float),
                "yaw": float(yaw),
                "yaw_rate": float(yaw_rate)
            })

        traj = self.fct_unwrap_trajectory_yaw(traj)
        return self.fct_limit_trajectory_yaw_rate(traj, time)

    def fct_make_hover_excitation_trajectory(
        self,
        time,
        rng,
        xyz_amp=(0.15, 0.15, 0.12),
        xyz_freq=(0.08, 0.10, 0.12),
        yaw_amp_deg=5.0,
        n_sines_range=(2, 4),
    ):
        """Generate the old-paper hover family with an explicit yaw reference.

        The translational excitation distribution matches the original data
        generator.  Unlike the old yaw-free version, the heading is retained
        as a smooth, bounded-rate reference so it excites the yaw dynamics
        without requesting an infeasible yaw step.
        """
        time = np.asarray(time, dtype=float)
        n_samples = len(time)
        if n_samples < 2:
            raise ValueError("time must contain at least two samples")

        ax_amp, ay_amp, az_amp = map(float, xyz_amp)
        fx_base, fy_base, fz_base = map(float, xyz_freq)
        n_sines = rng.randint(*n_sines_range)

        x = np.zeros(n_samples)
        y = np.zeros(n_samples)
        z = np.zeros(n_samples)
        for _ in range(n_sines):
            wx = 2.0 * np.pi * rng.uniform(0.5 * fx_base, 1.8 * fx_base)
            wy = 2.0 * np.pi * rng.uniform(0.5 * fy_base, 1.8 * fy_base)
            wz = 2.0 * np.pi * rng.uniform(0.5 * fz_base, 1.8 * fz_base)
            phx = rng.uniform(0.0, 2.0 * np.pi)
            phy = rng.uniform(0.0, 2.0 * np.pi)
            phz = rng.uniform(0.0, 2.0 * np.pi)
            x += (ax_amp / n_sines) * np.sin(wx * time + phx)
            y += (ay_amp / n_sines) * np.sin(wy * time + phy)
            z += (az_amp / n_sines) * np.sin(wz * time + phz)

        z = z - z[0] + 0.1
        yaw_amplitude = np.deg2rad(yaw_amp_deg)
        yaw_frequency = 2.0 * np.pi * rng.uniform(0.03, 0.10)
        yaw_phase = rng.uniform(0.0, 2.0 * np.pi)
        yaw = yaw_amplitude * np.sin(yaw_frequency * time + yaw_phase)
        yaw_rate = yaw_amplitude * yaw_frequency * np.cos(
            yaw_frequency * time + yaw_phase
        )

        dt = float(np.median(np.diff(time)))
        vx, vy, vz = (np.gradient(values, dt) for values in (x, y, z))
        ax, ay, az = (np.gradient(values, dt) for values in (vx, vy, vz))
        traj = [
            {
                "pos": np.array([x[k], y[k], z[k]], dtype=float),
                "vel": np.array([vx[k], vy[k], vz[k]], dtype=float),
                "acc": np.array([ax[k], ay[k], az[k]], dtype=float),
                "yaw": float(yaw[k]),
                "yaw_rate": float(yaw_rate[k]),
            }
            for k in range(n_samples)
        ]
        return self.fct_limit_trajectory_yaw_rate(traj, time)

    def fct_make_random_waypoint_trajectory(
        self,
        time,
        rng,
        n_waypoints=8,
        xy_range=3.0,
        z_range=(0.5, 4.0),
        smooth_sigma=50,
        interpolation="legacy_gaussian",
    ):
        """Generate a deterministic, yaw-aware random-waypoint family.

        ``legacy_gaussian`` preserves the original paper construction.
        ``natural_cubic_spline`` evaluates position, velocity, and acceleration
        analytically in physical time so the ACC waypoint has continuous
        fly-through velocity and acceleration at every interior waypoint.
        """
        from scipy.ndimage import gaussian_filter1d
        from scipy.interpolate import CubicSpline

        time = np.asarray(time, dtype=float)
        n_samples = len(time)
        if n_samples < 2:
            raise ValueError("time must contain at least two samples")

        waypoint_indices = np.linspace(0, n_samples - 1, n_waypoints).astype(int)
        sample_indices = np.arange(n_samples)
        waypoint_values = np.column_stack([
            [rng.uniform(-xy_range, xy_range) for _ in range(n_waypoints)],
            [rng.uniform(-xy_range, xy_range) for _ in range(n_waypoints)],
            [rng.uniform(*z_range) for _ in range(n_waypoints)],
        ])

        if interpolation == "legacy_gaussian":
            positions = np.column_stack([
                gaussian_filter1d(
                    np.interp(sample_indices, waypoint_indices, waypoint_values[:, axis]),
                    smooth_sigma,
                )
                for axis in range(3)
            ])
            dt = float(np.median(np.diff(time)))
            velocities = np.gradient(positions, dt, axis=0)
            accelerations = np.gradient(velocities, dt, axis=0)
        elif interpolation == "natural_cubic_spline":
            waypoint_times = time[waypoint_indices]
            spline = CubicSpline(
                waypoint_times, waypoint_values, axis=0, bc_type="natural"
            )
            positions = np.asarray(spline(time), dtype=float)
            velocities = np.asarray(spline(time, 1), dtype=float)
            accelerations = np.asarray(spline(time, 2), dtype=float)
        else:
            raise ValueError(
                "interpolation must be legacy_gaussian or natural_cubic_spline"
            )

        # As in the old generator, starts are translated to the origin.  Do
        # this before differentiating so the position and derivative fields
        # remain exactly consistent.
        positions -= positions[0]
        x, y, z = positions.T
        vx, vy, vz = velocities.T
        ax, ay, az = accelerations.T
        yaw = np.unwrap(np.arctan2(vy, vx))
        yaw_rate = (vx * ay - vy * ax) / (vx**2 + vy**2 + 1e-12)

        traj = [
            {
                "pos": np.array([x[k], y[k], z[k]], dtype=float),
                "vel": np.array([vx[k], vy[k], vz[k]], dtype=float),
                "acc": np.array([ax[k], ay[k], az[k]], dtype=float),
                "yaw": float(yaw[k]),
                "yaw_rate": float(yaw_rate[k]),
            }
            for k in range(n_samples)
        ]
        return self.fct_limit_trajectory_yaw_rate(traj, time)

    def fct_sample_trajectory(self, traj, rng, profile="paper"):
        """Sample a deterministic paper or ACC-balanced trajectory.

        The seed convention is managed by :meth:`fct_run_simulation`: run
        ``i`` from family ``traj`` always uses ``1000 * traj + i``.  The
        ``paper`` branch reproduces the original reference-data setup.  The
        ``acc_balanced`` branch uses 60-second runs with a fixed five-second
        parametric ramp, deliberately raises helix speed, and limits the
        already-aggressive families to 5 m/s and 6 m/s^2.
        """
        if profile == "acc_balanced":
            ramp = ACC_BALANCED_PROFILE_CONFIG["ramp_duration_seconds"]
            max_speed = ACC_BALANCED_PROFILE_CONFIG["max_reference_speed_mps"]
            max_acceleration = ACC_BALANCED_PROFILE_CONFIG[
                "max_reference_acceleration_mps2"
            ]

            if traj == 1:
                radius = rng.uniform(3.0, 10.0)
                z_end = rng.uniform(3.0, 10.0)
                target_speed = rng.uniform(1.0, 2.5)
                duration = float(self.time[-1] - self.time[0])
                cruise_duration = max(duration - ramp, self.dt)
                target_path_length = target_speed * cruise_duration
                horizontal_length = np.sqrt(
                    max(target_path_length**2 - z_end**2, 0.0)
                )
                n_turns = max(horizontal_length / (2.0 * np.pi * radius), 0.25)
                result = self.fct_make_helical_trajectory(
                    self.time,
                    center=(0.0, 0.0),
                    radius=radius,
                    z_start=0.0,
                    z_end=z_end,
                    n_turns=n_turns,
                    yaw_follows_path=True,
                    ramp_duration_seconds=ramp,
                )
            elif traj == 2:
                result = self.fct_make_figure8_trajectory(
                    self.time,
                    center=(0.0, 0.0, 0.0),
                    a=rng.uniform(20.0, 30.0),
                    b=rng.uniform(20.0, 30.0),
                    n_loops=1.0,
                    tilt_deg=rng.uniform(10.0, 80.0),
                    yaw_follows_path=True,
                    ramp_duration_seconds=ramp,
                )
            elif traj == 3:
                result = self.fct_make_lissajous_trajectory(
                    self.time,
                    center=(0.0, 0.0, rng.uniform(1.0, 4.0)),
                    ax=rng.uniform(12.0, 20.0),
                    ay=rng.uniform(12.0, 20.0),
                    az=rng.uniform(3.0, 7.0),
                    fx=rng.choice([1.0, 2.0, 3.0]),
                    fy=rng.choice([1.0, 2.0, 3.0]),
                    fz=rng.choice([1.0, 2.0, 3.0]),
                    phase_y=rng.uniform(0.0, np.pi),
                    phase_z=rng.uniform(0.0, np.pi),
                    yaw_follows_path=True,
                    ramp_duration_seconds=ramp,
                )
            elif traj == 4:
                result = self.fct_make_random_waypoint_trajectory(
                    self.time,
                    rng=rng,
                    n_waypoints=rng.randint(5, 15),
                    xy_range=rng.uniform(12.0, 20.0),
                    z_range=(0.5, rng.uniform(3.0, 8.0)),
                    interpolation="natural_cubic_spline",
                )
            elif traj == 5:
                result = self.fct_make_hover_excitation_trajectory(
                    self.time,
                    rng=rng,
                    xyz_amp=(
                        rng.uniform(2.0, 4.0),
                        rng.uniform(2.0, 4.0),
                        rng.uniform(1.0, 2.0),
                    ),
                    xyz_freq=(
                        rng.uniform(0.05, 0.12),
                        rng.uniform(0.05, 0.12),
                        rng.uniform(0.06, 0.15),
                    ),
                    yaw_amp_deg=rng.uniform(2.0, 8.0),
                    n_sines_range=(2, 4),
                )
            else:
                raise ValueError("traj must be an integer from 1 through 5")

            if traj in (4, 5):
                result = self.fct_ramp_trajectory_start(
                    result,
                    self.time,
                    ramp_duration_seconds=ramp,
                )
            family_max_speed = (
                ACC_BALANCED_PROFILE_CONFIG["waypoint_max_reference_speed_mps"]
                if traj == 4 else max_speed
            )
            family_max_acceleration = (
                ACC_BALANCED_PROFILE_CONFIG[
                    "waypoint_max_reference_acceleration_mps2"
                ]
                if traj == 4 else max_acceleration
            )
            return self.fct_scale_trajectory_to_limits(
                result,
                max_speed=family_max_speed,
                max_acceleration=family_max_acceleration,
            )

        if profile not in ("paper", "compact", "custom"):
            raise ValueError(
                "profile must be paper, acc_balanced, compact, or custom"
            )
        if traj == 1:
            return self.fct_make_helical_trajectory(
                self.time,
                center=(0.0, 0.0),
                radius=rng.uniform(3.0, 10.0),
                z_start=0.0,
                z_end=rng.uniform(3.0, 10.0),
                n_turns=1.0,
                yaw_follows_path=True,
            )
        if traj == 2:
            return self.fct_make_figure8_trajectory(
                self.time,
                center=(0.0, 0.0, 0.0),
                a=rng.uniform(25.0, 35.0),
                b=rng.uniform(25.0, 35.0),
                n_loops=1.0,
                tilt_deg=rng.uniform(10.0, 80.0),
                yaw_follows_path=True,
            )
        if traj == 3:
            return self.fct_make_lissajous_trajectory(
                self.time,
                center=(0.0, 0.0, rng.uniform(1.0, 4.0)),
                ax=rng.uniform(15.0, 25.0),
                ay=rng.uniform(15.0, 25.0),
                az=rng.uniform(3.0, 7.0),
                fx=rng.choice([1.0, 2.0, 3.0]),
                fy=rng.choice([1.0, 2.0, 3.0]),
                fz=rng.choice([1.0, 2.0, 3.0]),
                phase_y=rng.uniform(0.0, np.pi),
                phase_z=rng.uniform(0.0, np.pi),
                yaw_follows_path=True,
            )
        if traj == 4:
            return self.fct_make_random_waypoint_trajectory(
                self.time,
                rng=rng,
                n_waypoints=rng.randint(5, 15),
                xy_range=rng.uniform(15.0, 25.0),
                z_range=(0.5, rng.uniform(3.0, 8.0)),
                smooth_sigma=rng.randint(25, 50),
            )
        if traj == 5:
            return self.fct_make_hover_excitation_trajectory(
                self.time,
                rng=rng,
                xyz_amp=(
                    rng.uniform(2.0, 4.0),
                    rng.uniform(2.0, 4.0),
                    rng.uniform(1.0, 2.0),
                ),
                xyz_freq=(
                    rng.uniform(0.05, 0.12),
                    rng.uniform(0.05, 0.12),
                    rng.uniform(0.06, 0.15),
                ),
                yaw_amp_deg=rng.uniform(2.0, 8.0),
                n_sines_range=(2, 4),
            )
        raise ValueError("traj must be an integer from 1 through 5")

    def fct_run_single_simulation(self, traj, run_index, profile="paper"):
        """Run one deterministic realization from a trajectory family."""
        import random

        rng = random.Random(1000 * traj + run_index)
        ref_traj = self.fct_sample_trajectory(traj, rng, profile=profile)

        # Keep the original paper's origin convention for every family.
        p0 = ref_traj[0]["pos"].copy()
        for point in ref_traj:
            point["pos"] = point["pos"] - p0

        # Starting at the reference heading avoids turning the first logged
        # transition into an artificial yaw-setpoint transient.
        init_state = np.zeros(12)
        init_state[8] = float(ref_traj[0].get("yaw", 0.0))
        t, states, _, U, U_requested, U_outer = self.sim_PID.fct_simulate(
            self.time, self.dt, ref_traj, init_state,
            return_requested=True, return_outer=True,
        )
        return t, states, U, U_requested, U_outer, ref_traj

    def fct_run_simulation(self, traj, n, profile="paper"):
        """
        Run n simulations using a single trajectory type.

        All runs start at zero position/velocity with yaw initialized to the
        first reference heading. Roll, pitch, and angular rates start at zero.

        traj = 1  -> helical trajectory
        traj = 2  -> figure-8 trajectory
        traj = 3  -> lissajous trajectory
        traj = 4  -> random waypoint trajectory
        traj = 5  -> hover excitation trajectory

        Each run gets different randomized trajectory parameters,
        but they are deterministic by run index. That means:

        - calling this function multiple times with the same traj and n
        will generate the exact same trajectories
        - run 0 always uses the same trajectory parameters
        - run 1 always uses the same trajectory parameters
        - etc.

        Returns
        -------
        t : (n, T)
        states : (n, T, 12)
        U : (n, T, n_inputs)
        ref_traj_list : list of reference trajectories used for each run
        """

        t_runs = []
        states_runs = []
        U_runs = []
        U_requested_runs = []
        U_outer_runs = []
        ref_traj_list = []

        for i in range(n):
            t_i, states_i, U_i, U_requested_i, U_outer_i, ref_traj = (
                self.fct_run_single_simulation(traj, i, profile=profile)
            )

            t_runs.append(t_i)
            states_runs.append(states_i)
            U_runs.append(U_i)
            U_requested_runs.append(U_requested_i)
            U_outer_runs.append(U_outer_i)
            ref_traj_list.append(ref_traj)

        # =====================================================
        # Stack results
        # =====================================================
        t = np.stack(t_runs, axis=0)
        states = np.stack(states_runs, axis=0)
        U = np.stack(U_runs, axis=0)
        self.last_requested_inputs = np.stack(U_requested_runs, axis=0)
        self.last_applied_inputs = U
        self.last_outer_inputs = np.stack(U_outer_runs, axis=0)

        return t, states, U, ref_traj_list

    # def fct_run_simulation(self, traj, n):
    #     """
    #     Run n simulations cycling through available trajectory types.

    #     Deterministic randomization per run:
    #     - Same (traj, n) → same trajectories every time
    #     - Different runs → different parameters

    #     All runs start from:
    #         state = 0
    #         trajectory start = (0,0,0)
    #     """

    #     import random

    #     # ---------------------------------------------------------
    #     # Available trajectory types
    #     # ---------------------------------------------------------
    #     traj_ids = [1, 2]
    #     num_traj_types = len(traj_ids)

    #     start_index = (traj - 1) % num_traj_types

    #     t_runs = []
    #     states_runs = []
    #     U_runs = []
    #     ref_traj_list = []

    #     for i in range(n):

    #         # -----------------------------------------------------
    #         # Select trajectory type (cycling)
    #         # -----------------------------------------------------
    #         traj_id = traj_ids[(start_index + i) % num_traj_types]

    #         # -----------------------------------------------------
    #         # Deterministic RNG (KEY PART)
    #         # -----------------------------------------------------
    #         seed = 1000 * traj_id + i
    #         rng = random.Random(seed)

    #         # =====================================================
    #         # 1) Build trajectory (deterministic per run)
    #         # =====================================================
    #         if traj_id == 1:

    #             ref_traj = self.fct_make_helical_trajectory(
    #                 self.time,
    #                 center=(0.0, 0.0),
    #                 radius=rng.uniform(1, 5),
    #                 z_start=0.0,
    #                 z_end=rng.uniform(5, 10),
    #                 n_turns=1,
    #                 yaw_follows_path=True
    #             )

    #         elif traj_id == 2:

    #             ref_traj = self.fct_make_figure8_trajectory(
    #                 self.time,
    #                 center=(0.0, 0.0, 0.0),
    #                 a=rng.uniform(1, 5),
    #                 b=rng.uniform(1, 5),
    #                 n_loops=1,
    #                 tilt_deg=rng.uniform(10, 80),
    #                 yaw_follows_path=True
    #             )

    #         else:
    #             raise ValueError(f"Unknown trajectory id: {traj_id}")

    #         # =====================================================
    #         # 2) Shift trajectory to start at origin
    #         # =====================================================
    #         p0 = ref_traj[0]["pos"].copy()
    #         for k in range(len(ref_traj)):
    #             ref_traj[k]["pos"] -= p0

    #         ref_traj_list.append(ref_traj)

    #         # =====================================================
    #         # 3) Initial state = ZERO
    #         # =====================================================
    #         init_state = np.zeros(12)

    #         # =====================================================
    #         # 4) Run simulation
    #         # =====================================================
    #         t_i, states_i, omegas_i, U_i = self.sim_PID.fct_simulate(
    #             self.time, self.dt, ref_traj, init_state
    #         )

    #         t_runs.append(t_i)
    #         states_runs.append(states_i)
    #         U_runs.append(U_i)

    #     # =====================================================
    #     # 5) Stack results
    #     # =====================================================
    #     t = np.stack(t_runs, axis=0)
    #     states = np.stack(states_runs, axis=0)
    #     U = np.stack(U_runs, axis=0)

    #     return t, states, U, ref_traj_list

    def fct_save_simulation_runs(self, traj, n, filename="saved_runs.pkl"):
        """
        Run simulation using the current deterministic randomized
        fct_run_simulation(...) and save the results to a pickle file.

        This preserves the exact same trajectory generation behavior
        as the current fct_run_simulation method.
        """
        import pickle

        # Use the existing simulation function exactly as-is
        t, states, U, ref_traj_list = self.fct_run_simulation(traj, n)

        data = {
            "traj": traj,
            "n": n,
            "sim_dt": self.dt,
            "time": self.time,
            "t": t,
            "states": states,
            "U": U,
            "ref_traj_list": ref_traj_list,
        }

        with open(filename, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved simulation runs to {filename}")
