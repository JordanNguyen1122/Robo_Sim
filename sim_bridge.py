import math
import sys
 
import rclpy                                      # ROS 2 Python client library
from rclpy.node import Node                       # base class for a ROS 2 node
from nav_msgs.msg import Odometry                 # INPUT:  what the simulator publishes
from booster_interface.msg import Odometer        # OUTPUT: what the brain listens for
 
 
def yaw_from_quat(q):
    """
    Convert a quaternion (x, y, z, w) into a yaw angle in radians.
 
    The simulator gives the robot's orientation as a 3D rotation (quaternion).
    The brain only needs the heading on the flat field ("yaw"):
        0      = facing +x
        +pi/2  = facing +y
        +-pi   = facing -x
    This is the standard quaternion -> yaw formula.
    """
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
 
 
class SimOdomBridge(Node):
    """ROS 2 node: /<robot>/odom (sim)  ->  /odometer_state/<robot> (brain)."""
 
    def __init__(self, robot, odom_factor, flip=False):
        # Give each bridge a unique node name so several robots can run at once.
        super().__init__(f"sim_odom_bridge_{robot}")
 
        # SLIP CORRECTION
        # The brain multiplies odometry x/y by robot.odom_factor (0.8 in config.yaml)
        # because the REAL robot's feet slip and it over-estimates distance.
        # The simulator has no slip, so we divide by the same factor here -
        # the two cancel out and the brain gets the true distance.
        self.odom_factor = odom_factor if odom_factor else 1.0
 
        # FLIP: True / False, or None (= auto, decided from the first message).
        # True rotates this robot's coordinates by 180 deg - needed for the team
        # whose own goal is on the sim's +x side (AWAY), because the brain assumes
        # its own goal is at -x.
        self.flip = flip
 
        # Message counter, only used to print a status line every few seconds.
        self.count = 0
 
        # Topic names for this robot.
        in_topic = f"/{robot}/odom"               # sim publishes here
        out_topic = f"/odometer_state/{robot}"    # brain subscribes here
 
        # Subscribe to the sim: every incoming message calls self.on_odom().
        # The "10" is the queue size (buffer up to 10 messages if we fall behind).
        self.create_subscription(Odometry, in_topic, self.on_odom, 10)
 
        # Publisher towards the brain.
        self.pub = self.create_publisher(Odometer, out_topic, 10)
 
        # First line of <robot>_odom.log - handy to confirm the flip setting.
        self.get_logger().info(
            f"{in_topic} -> {out_topic}  "
            f"(odom_factor compensation: {self.odom_factor}, flip: {self.flip})")
 
    def on_odom(self, msg):
        """Called for every sim odometry message (~500 Hz). Convert and republish."""
        p = msg.pose.pose
        out = Odometer()
 
        # Read position and heading from the sim message.
        x, y, th = p.position.x, p.position.y, yaw_from_quat(p.orientation)
 
        # AUTO FLIP (only when neither --flip nor --no-flip was given).
        # Decides once, from the very first heading: facing +y (th > 0) -> flip.
        # WARNING: this is unreliable. The starting heading only tells you which
        # SIDELINE the robot stands on, not which HALF it is in. Always pass
        # --flip / --no-flip explicitly (the launch scripts do).
        if self.flip is None:
            self.flip = th > 0
            self.get_logger().info(
                f"auto flip decided from start heading {th:.2f} rad -> flip: {self.flip}")
 
        # 180-degree flip: negate x and y, add pi to the heading.
        # atan2(sin, cos) keeps the angle inside -pi..+pi.
        if self.flip:
            x, y, th = -x, -y, math.atan2(math.sin(th + math.pi), math.cos(th + math.pi))
 
        # Fill the brain's message. Divide by odom_factor to cancel the brain's
        # slip correction (see __init__). theta is NOT scaled.
        out.x = float(x / self.odom_factor)
        out.y = float(y / self.odom_factor)
        out.theta = float(th)
        self.pub.publish(out)
 
        # Status line roughly every 5 s (2500 msgs at ~500 Hz) so the log stays readable.
        # Note: the x/y printed here are the RAW sim values, theta is after the flip.
        self.count += 1
        if self.count % 2500 == 0:
            self.get_logger().info(
                f"published {self.count} | sim odom x={p.position.x:.2f} y={p.position.y:.2f} "
                f"theta={out.theta:.2f}")
 
 
def main():
    # Read command-line arguments, ignoring anything ROS adds (--ros-args ...).
    args = [a for a in sys.argv[1:] if not a.startswith("--ros-args")]
 
    # First plain argument is the robot name, e.g. "robot2". Default: robot1.
    robot = args[0] if args and not args[0].startswith("--") else "robot1"
 
    # Must match robot.odom_factor in src/brain/config/config.yaml.
    odom_factor = 0.8
    if "--odom-factor" in args:
        odom_factor = float(args[args.index("--odom-factor") + 1])
 
    rclpy.init()  # start ROS 2
 
    # Flip option: None = auto (not recommended), True = --flip, False = --no-flip.
    flip = None
    if "--flip" in args:
        flip = True
    if "--no-flip" in args:
        flip = False
 
    node = SimOdomBridge(robot, odom_factor, flip=flip)
    try:
        # Run forever, calling on_odom() for every incoming message,
        # until Ctrl+C or the launch script stops it (pkill).
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        # Normal shutdown - swallow the exception instead of printing a traceback.
        pass
 
    # Clean up. The rclpy.ok() check avoids a "shutdown already called" error
    # when ROS was already shut down by the signal.
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
 
 
if __name__ == "__main__":
    main()
 