import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


class ReactiveFollowGap(Node):
    def __init__(self):
        super().__init__('reactive_node')
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # Lidar / FOV
        self.fov_deg = 120
        self.max_range = 6.0
        self.smooth_window = 7
        self.min_gap_dist = 0.70
        self.bubble_radius = 0.55

        # Gap scoring (centered, avoid side-bait)
        self.w_len = 1.0
        self.w_center = 3.0
        self.w_side = 2.2  # penalize both sides

        # Steering
        self.steer_limit = 0.45
        self.lookahead = 1.20
        self.k_steer = 1.1

        # Speed
        self.v_slow = 1.4
        self.v_mid = 2.4
        self.v_fast = 3.8
        self.steer_mid_th = 0.22
        self.steer_slow_th = 0.34

        # Smoothing
        self.prev_steer = 0.0
        self.alpha_straight = 0.06
        self.alpha_turn = 0.22

        # Front blocked 판단
        self.front_block_th = 1.05

        # Spiral breaker (양쪽 다 잡음)
        self.spiral_th = 0.20          # |steer| > th 이면 말림 시작 후보
        self.spiral_count_need = 4     # 연속 프레임
        self.spiral_hold = 10          # 반대 조향 유지 프레임
        self.spiral_push = 0.34        # 반대 조향 크기
        self.spiral_speed = 1.1        # 탈출 중 속도

        self._same_sign_cnt = 0
        self._last_sign = 0
        self._break_cnt = 0
        self._break_steer = 0.0

    def publish(self, speed, steer):
        msg = AckermannDriveStamped()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steer)
        self.drive_pub.publish(msg)

    def preprocess(self, ranges):
        x = np.array(ranges, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=self.max_range, neginf=0.0)
        x[x < 0.0] = 0.0
        x[x > self.max_range] = self.max_range
        w = int(self.smooth_window)
        if w >= 3 and w % 2 == 1:
            k = np.ones(w, dtype=np.float32) / w
            x = np.convolve(x, k, mode='same')
        return x

    def idx_of_angle(self, scan, ang):
        i = int(round((ang - scan.angle_min) / scan.angle_increment))
        return int(np.clip(i, 0, len(scan.ranges) - 1))

    def slice_fov(self, scan, ranges):
        i0 = self.idx_of_angle(scan, 0.0)
        half = int((np.deg2rad(self.fov_deg) * 0.5) / scan.angle_increment)
        s = max(0, i0 - half)
        e = min(len(ranges), i0 + half + 1)
        return s, e, ranges[s:e].copy()

    def find_closest(self, arr):
        valid = arr > 0.05
        if not np.any(valid):
            return None
        m = np.where(valid, arr, np.inf)
        return int(np.argmin(m))

    def apply_bubble(self, arr, closest_i, inc):
        n = int(self.bubble_radius / inc)
        s = max(0, closest_i - n)
        e = min(len(arr) - 1, closest_i + n)
        arr[s:e + 1] = 0.0

    def enumerate_gaps(self, arr):
        free = arr > self.min_gap_dist
        idx = np.where(free)[0]
        if idx.size == 0:
            return []
        gaps = []
        st = int(idx[0])
        pv = int(idx[0])
        for i in idx[1:]:
            i = int(i)
            if i == pv + 1:
                pv = i
            else:
                gaps.append((st, pv))
                st, pv = i, i
        gaps.append((st, pv))
        return gaps

    def front_blocked(self, arr):
        n = len(arr)
        c = n // 2
        w = max(3, n // 10)
        f = arr[max(0, c - w):min(n, c + w)]
        f = f[f > 0.05]
        m = float(np.mean(f)) if f.size else 0.0
        return m < self.front_block_th

    def choose_gap(self, gaps, arr):
        n = len(arr)
        c = n // 2
        denom = max(1, c)

        best = None
        best_score = -1e18

        for a, b in gaps:
            length = (b - a + 1)
            mid = (a + b) // 2
            center_off = abs(mid - c) / denom
            side_off = abs(mid - c) / denom  # 양쪽 다 패널티

            score = (
                self.w_len * float(length)
                - self.w_center * 120.0 * float(center_off)
                - self.w_side * 80.0 * float(side_off)
            )
            if score > best_score:
                best_score = score
                best = (a, b)
        return best

    def pick_aimpoint(self, a, b, arr):
        seg = arr[a:b + 1]
        if seg.size == 0:
            return None
        i_far = a + int(np.argmax(seg))

        n = len(arr)
        c = n // 2
        w = 18
        lo = max(a, i_far - w)
        hi = min(b, i_far + w)
        cand = np.arange(lo, hi + 1)
        vals = arr[lo:hi + 1]
        if vals.size == 0:
            return i_far

        center_bonus = 1.0 - (np.abs(cand - c) / max(1, c))
        score = vals + 0.45 * center_bonus
        return int(cand[int(np.argmax(score))])

    def steer_from_index(self, scan, global_idx):
        ang = scan.angle_min + global_idx * scan.angle_increment
        y = np.sin(ang) * self.lookahead
        steer = self.k_steer * np.arctan2(2.0 * y, self.lookahead**2)
        return float(np.clip(steer, -self.steer_limit, self.steer_limit))

    def speed_from_steer(self, steer):
        a = abs(steer)
        if a > self.steer_slow_th:
            return self.v_slow
        elif a > self.steer_mid_th:
            return self.v_mid
        return self.v_fast

    def update_spiral_breaker(self, steer, blocked):
        if self._break_cnt > 0:
            self._break_cnt -= 1
            return self._break_steer, self.spiral_speed

        a = abs(steer)
        sign = 1 if steer > 0.0 else (-1 if steer < 0.0 else 0)

        if blocked and a > self.spiral_th and sign != 0 and sign == self._last_sign:
            self._same_sign_cnt += 1
        else:
            self._same_sign_cnt = max(0, self._same_sign_cnt - 1)

        if sign != 0:
            self._last_sign = sign

        if self._same_sign_cnt >= self.spiral_count_need:
            self._same_sign_cnt = 0
            self._break_cnt = self.spiral_hold
            self._break_steer = float(np.clip(-sign * self.spiral_push, -self.steer_limit, self.steer_limit))
            return self._break_steer, self.spiral_speed

        return None, None

    def lidar_callback(self, scan: LaserScan):
        ranges = self.preprocess(scan.ranges)
        s, e, arr = self.slice_fov(scan, ranges)

        closest = self.find_closest(arr)
        if closest is None:
            self.publish(0.0, 0.0)
            return

        self.apply_bubble(arr, closest, scan.angle_increment)

        gaps = self.enumerate_gaps(arr)
        if not gaps:
            self.publish(0.0, 0.0)
            return

        chosen = self.choose_gap(gaps, arr)
        if chosen is None:
            self.publish(0.0, 0.0)
            return

        a, b = chosen
        aim = self.pick_aimpoint(a, b, arr)
        if aim is None:
            self.publish(0.0, 0.0)
            return

        global_idx = s + aim
        steer = self.steer_from_index(scan, global_idx)

        blocked = self.front_blocked(arr)

        brk = self.update_spiral_breaker(steer, blocked)
        if brk[0] is not None:
            steer_cmd, speed_cmd = brk
            self.prev_steer = steer_cmd
            self.publish(speed_cmd, steer_cmd)
            return

        alpha = self.alpha_turn if abs(steer) > 0.18 else self.alpha_straight
        steer = (1 - alpha) * self.prev_steer + alpha * steer
        steer = float(np.clip(steer, -self.steer_limit, self.steer_limit))
        self.prev_steer = steer

        speed = self.speed_from_steer(steer)
        if blocked:
            speed = min(speed, self.v_mid)

        self.publish(speed, steer)


def main(args=None):
    rclpy.init(args=args)
    node = ReactiveFollowGap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
