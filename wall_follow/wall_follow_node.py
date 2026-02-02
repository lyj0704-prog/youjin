import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


class WallFollow(Node):
    """
    Implement Wall Following on the car (follow LEFT wall)
    """
    def __init__(self):
        super().__init__('wall_follow_node')

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # TODO: create subscribers and publishers
        self.scan_sub = self.create_subscription(
            LaserScan, lidarscan_topic, self.scan_callback, 10
        )
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, drive_topic, 10
        )

        # ===== Tunable parameters =====
        self.desired_dist = 0.75  # 원하는 벽-차 거리 (m)
        self.theta = np.deg2rad(60) # a,b를 읽을 두 레이 사이 각도 (45deg 권장)
        self.L = 0.57   # lookahead 거리 (m)

        # TODO: set PID gains
        self.kp = 1.4
        self.kd = 0.4
        self.ki = 0.0

        # TODO: store history
        self.integral = 0.0
        self.prev_error = 0.0
        self.error = 0.0

        # TODO: store any necessary values you think you'll need
        self.angle_min = None
        self.angle_inc = None
        self.prev_time = None

        # steering/speed limits
        self.steer_limit = 0.38 # rad (약 24도)

    def scan_callback(self, msg: LaserScan):
        # LiDAR 메타 저장 (get_range에서 사용)
        if self.angle_min is None:
            self.angle_min = msg.angle_min
            self.angle_inc = msg.angle_increment

        # dt 계산
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.prev_time is None:
            self.prev_time = now
            return
        dt = max(now - self.prev_time, 1e-3)
        self.prev_time = now

        ranges = np.array(msg.ranges, dtype=np.float32)

        # error 계산 (왼쪽 벽 기준)
        error = self.get_error(ranges, self.desired_dist)

        # PID 업데이트
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        self.error = error

        steer = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        # 조향각 제한
        steer = float(np.clip(steer, -self.steer_limit, self.steer_limit))

        # 조향이 클수록 속도 낮추기(안정성) - 리듬 일정 + 랩타임 개선
        abs_steer = abs(steer)
        if abs_steer > 0.25:
            speed = 2.3
        elif abs_steer > 0.18:
            speed = 3.8
        else:
            speed = 5.1
        # publish
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = msg.header.stamp
        drive_msg.drive.steering_angle = steer
        drive_msg.drive.speed = float(speed)
        self.drive_pub.publish(drive_msg)

    def get_range(self, range_data, angle):
        """
        Return range at 'angle' (rad). Handles NaN/Inf by falling back to nearest valid.
        Requires self.angle_min, self.angle_inc set from LaserScan.
        """
        if self.angle_min is None or self.angle_inc is None:
            return 0.0

        # angle -> index
        idx = int(round((angle - self.angle_min) / self.angle_inc))
        idx = int(np.clip(idx, 0, len(range_data) - 1))

        r = range_data[idx]

        # NaN/Inf 처리: 주변에서 가장 가까운 finite 값 탐색 (최대 ±10칸)
        if not np.isfinite(r) or r <= 0.0:
            for k in range(1, 11):
                i1 = idx - k
                i2 = idx + k
                if i1 >= 0:
                    v = range_data[i1]
                    if np.isfinite(v) and v > 0.0:
                        return float(v)
                if i2 < len(range_data):
                    v = range_data[i2]
                    if np.isfinite(v) and v > 0.0:
                        return float(v)
            # 끝까지 못 찾으면 큰 값으로 처리(벽이 멀다로 가정)
            return 10.0

        return float(r)

    def get_error(self, range_data, dist):
        """
        Left wall following error using 2 rays and lookahead.

        Use:
          b = range at 90deg (left)
          a = range at 90deg - theta (front-left)
        Compute wall angle alpha, estimate future distance Dt1, then:
          error = Dt1 - dist
        """
        # 왼쪽은 +pi/2
        b = self.get_range(range_data, np.pi / 2.0)
        a = self.get_range(range_data, np.pi / 2.0 - self.theta)

        # 수치 안정성
        a = max(a, 1e-3)
        b = max(b, 1e-3)

        # 벽과 차의 각도 alpha
        numerator = a * np.cos(self.theta) - b
        denominator = a * np.sin(self.theta)
        alpha = np.arctan2(numerator, denominator)

        # 현재 벽까지 수직거리 Dt, lookahead 적용한 미래 거리 Dt1
        Dt = b * np.cos(alpha)
        Dt1 = Dt + self.L * np.sin(alpha)

        error = Dt1 - dist
        return float(error)


def main(args=None):
    rclpy.init(args=args)
    node = WallFollow()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
