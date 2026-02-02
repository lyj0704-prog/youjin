import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


class ReactiveFollowGap(Node):
    """
    Reactive Follow Gap (Follow-the-Gap) for F1TENTH
    Final tuned version for Easy Map:
      - Stable gap selection (min_gap_dist, bubble, best_window, smoothing)
      - Edge trimming to avoid picking risky gap borders
      - Closest-distance speed cap to prevent long-run crashes
      - Steering low-pass with straight/turn split (less wobble on straights)
      - Wall-follow-like speed tiers with slightly higher straight speed
    """

    def __init__(self):
        super().__init__('reactive_node')

        # Topics
        self.lidarscan_topic = '/scan'
        self.drive_topic = '/drive'

        # ROS I/O
        self.scan_sub = self.create_subscription(
            LaserScan, self.lidarscan_topic, self.lidar_callback, 10
        )
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 10
        )

        # =============================
        # Tunable parameters (Easy Map)
        # =============================

        # (1) Lidar preprocessing
        self.max_range = 8.0           # clamp large ranges for stability
        self.smooth_window = 11 #oving average window (odd)

        # (2) Use only front field-of-view
        self.fov_deg = 140      # front FOV

        # (3) Gap logic
        self.min_gap_dist = 0.55    # free-space threshold
        self.bubble_radius = 0.88   # safety bubble around closest obstacle (m)
        self.best_window = 61  # stabilize best point choice (odd)

        # (4) Steering
        self.steer_limit = 0.55        # steering clamp (rad)

        # (5) Speed mapping (wall-follow-like but faster on straights)
        self.steer_slow_th = 0.34
        self.steer_mid_th = 0.32
        self.v_slow = 2.3
        self.v_mid = 3.4
        self.v_fast = 5.3

        # (6) Steering smoothing (straight vs turn)
        self.prev_steer = 0.0
        self.steer_alpha_straight = 0.12
        self.steer_alpha_turn = 0.15

        # Failsafe
        self.stop_on_no_scan = True

        # (7) Safety add-ons (keep the car alive long-run)
        self.edge_margin_m = 0.34 # trim risky gap edges (meters)
        self.cap_close_1 = 0.55        # if closest < this -> strong cap
        self.cap_close_2 = 0.80
     # if closest < this -> mid cap
        self.cap_speed_1 = 2.6        # strong cap speed
        self.cap_speed_2 = 4.8 # mid cap speed

    # ---------- Helper functions ----------

    def preprocess_lidar(self, ranges: np.ndarray) -> np.ndarray:
        """Clamp + sanitize + smooth lidar ranges."""
        proc = np.array(ranges, dtype=np.float32)
        proc = np.nan_to_num(proc, nan=0.0, posinf=self.max_range, neginf=0.0)
        proc[proc < 0.0] = 0.0
        proc[proc > self.max_range] = self.max_range

        # simple moving average smoothing
        w = int(self.smooth_window)
        if w >= 3 and w % 2 == 1:
            kernel = np.ones(w, dtype=np.float32) / w
            proc = np.convolve(proc, kernel, mode='same')

        return proc

    def slice_front_fov(self, scan: LaserScan, ranges: np.ndarray):
        """Return (s, e, front_ranges) for the front FOV slice."""
        total = len(ranges)
        center = total // 2
        half = int((np.deg2rad(self.fov_deg) / 2.0) / scan.angle_increment)
        s = max(0, center - half)
        e = min(total, center + half)
        return s, e, ranges[s:e].copy()

    def find_closest_index(self, proc: np.ndarray):
        """Closest obstacle index in proc, ignoring invalid(0-ish) points."""
        valid = proc > 0.05
        if not np.any(valid):
            return None
        masked = np.where(valid, proc, np.inf)
        return int(np.argmin(masked))

    def apply_bubble(self, proc: np.ndarray, closest_i: int, angle_increment: float):
        """Zero out points within bubble radius around closest_i."""
        bubble_n = int(self.bubble_radius / angle_increment)
        s = max(0, closest_i - bubble_n)
        e = min(len(proc) - 1, closest_i + bubble_n)
        proc[s:e + 1] = 0.0

    def find_max_gap(self, proc: np.ndarray):
        """
        Find the longest contiguous segment where distance > min_gap_dist.
        Returns (start_i, end_i) in proc indices.
        """
        free = proc > self.min_gap_dist
        free_idx = np.where(free)[0]
        if free_idx.size == 0:
            return None

        gaps = []
        start = int(free_idx[0])
        prev = int(free_idx[0])

        for idx in free_idx[1:]:
            idx = int(idx)
            if idx == prev + 1:
                prev = idx
            else:
                gaps.append((start, prev))
                start = idx
                prev = idx
        gaps.append((start, prev))

        return max(gaps, key=lambda g: g[1] - g[0])

    def find_best_point(self, start_i: int, end_i: int, proc: np.ndarray):
        """
        Choose a stable target inside the max gap.
        - Find farthest point in gap
        - Then stabilize with a local window around it
        """
        if end_i < start_i:
            return None

        gap_slice = proc[start_i:end_i + 1]
        if gap_slice.size == 0:
            return None

        local_max = int(np.argmax(gap_slice))
        best = start_i + local_max

        # Stabilize: search again within +/- best_window//2
        w = int(self.best_window) // 2
        lo = max(start_i, best - w)
        hi = min(end_i, best + w)
        window = proc[lo:hi + 1]
        if window.size == 0:
            return best
        best2 = lo + int(np.argmax(window))
        return best2

    def steering_to_speed(self, steer: float) -> float:
        """Speed tiers based on steering magnitude."""
        a = abs(steer)
        if a > self.steer_slow_th:
            return self.v_slow
        elif a > self.steer_mid_th:
            return self.v_mid
        else:
            return self.v_fast

    def publish_drive(self, speed: float, steer: float):
        msg = AckermannDriveStamped()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steer)
        self.drive_pub.publish(msg)

    # ---------- Main callback ----------

    def lidar_callback(self, scan: LaserScan):
        ranges = self.preprocess_lidar(scan.ranges)

        # Use only front FOV
        s, e, proc = self.slice_front_fov(scan, ranges)

        # Closest obstacle (ignore zeros)
        closest_i = self.find_closest_index(proc)
        if closest_i is None:
            if self.stop_on_no_scan:
                self.publish_drive(0.0, 0.0)
            else:
                self.publish_drive(self.v_slow, 0.0)
            return

        # Bubble around closest obstacle
        self.apply_bubble(proc, closest_i, scan.angle_increment)

        # Find max gap
        gap = self.find_max_gap(proc)
        if gap is None:
            self.publish_drive(0.0, 0.0)
            return
        start_i, end_i = gap

        # Safety: trim risky gap edges
        edge = int(self.edge_margin_m / scan.angle_increment)
        start_i = min(end_i, start_i + edge)
        end_i = max(start_i, end_i - edge)

        # Best point in gap
        best_i = self.find_best_point(start_i, end_i, proc)
        if best_i is None:
            self.publish_drive(0.0, 0.0)
            return
       

        # Convert best index -> steering angle (global index)
        best_global = s + best_i
        steer = scan.angle_min + best_global * scan.angle_increment
        steer = float(np.clip(steer, -self.steer_limit, self.steer_limit))

        # Steering smoothing (straight vs turn)
        alpha = self.steer_alpha_turn if abs(steer) > 0.18 else self.steer_alpha_straight
        steer = (1 - alpha) * self.prev_steer + alpha * steer

        self.prev_steer = steer

        # Speed control (steer-based)
        speed = self.steering_to_speed(steer)

        # Safety: closest-distance speed cap (prevents long-run crashes)
        closest_global = s + closest_i
        closest_dist = ranges[closest_global] 

        if closest_dist > 0.05:
            if closest_dist < self.cap_close_1:
                speed = min(speed, self.cap_speed_1)
            elif closest_dist < self.cap_close_2:
                speed = min(speed, self.cap_speed_2)
        

        self.publish_drive(speed, steer)


def main(args=None):
    rclpy.init(args=args)
    node = ReactiveFollowGap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()