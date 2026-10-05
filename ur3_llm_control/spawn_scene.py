import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
import math

class SceneSpawner(Node):
    def __init__(self):
        super().__init__('scene_spawner')
        self.client = self.create_client(SpawnEntity, '/spawn_entity')
        self.client.wait_for_service()
        self.spawn_all()

    def make_dynamic_cube_sdf(self, name, r, g, b):
        return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>false</static>
    <link name="link">
      <inertial>
        <mass>0.04</mass>
        <inertia>
          <ixx>0.00001</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>0.00001</iyy><iyz>0</iyz><izz>0.00001</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry><box><size>0.04 0.04 0.04</size></box></geometry>
        <surface>
          <friction>
            <ode><mu>50.0</mu><mu2>50.0</mu2></ode>
          </friction>
        </surface>
      </collision>
      <visual name="visual">
        <geometry><box><size>0.04 0.04 0.04</size></box></geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

    def make_zone_sdf(self, name, r, g, b):
        return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry><box><size>0.12 0.12 0.001</size></box></geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

    def make_camera_sdf(self):
        return """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="overhead_camera">
    <static>true</static>
    <link name="camera_link">
      <sensor name="top_camera" type="camera">
        <camera>
          <horizontal_fov>1.25</horizontal_fov>
          <image>
            <width>640</width>
            <height>480</height>
            <format>R8G8B8</format>
          </image>
          <clip><near>0.1</near><far>10.0</far></clip>
        </camera>
        <always_on>1</always_on>
        <update_rate>15</update_rate>
        <visualize>true</visualize>
        <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
          <ros>
            <namespace>/camera</namespace>
            <remapping>image_raw:=image_raw</remapping>
            <remapping>camera_info:=camera_info</remapping>
          </ros>
        </plugin>
      </sensor>
    </link>
  </model>
</sdf>"""

    def send_spawn_request(self, name, sdf_xml, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
        req = SpawnEntity.Request()
        req.name = name
        req.xml = sdf_xml
        req.initial_pose = Pose()
        req.initial_pose.position.x = float(x)
        req.initial_pose.position.y = float(y)
        req.initial_pose.position.z = float(z)

        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        req.initial_pose.orientation.w = cr * cp * cy + sr * sp * sy
        req.initial_pose.orientation.x = sr * cp * cy - cr * sp * sy
        req.initial_pose.orientation.y = cr * sp * cy + sr * cp * sy
        req.initial_pose.orientation.z = cr * cp * sy - sr * sp * cy

        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

    def spawn_all(self):
        self.get_logger().info("Đang nạp môi trường (Không bàn, 5 Cube, 3 Zone, Camera)...")

        # 1. Camera nhìn từ trên cao xuống
        self.send_spawn_request("overhead_camera", self.make_camera_sdf(), 0.05, 0.0, 1.25, 0.0, 1.5708, 0.0)

        # 2. 03 Zone
        self.send_spawn_request("zone_a", self.make_zone_sdf("zone_a", 0.35, 0.35, 0.35), -0.18, 0.22, 0.001)
        self.send_spawn_request("zone_b", self.make_zone_sdf("zone_b", 0.45, 0.45, 0.45), -0.25, 0.00, 0.001)
        self.send_spawn_request("zone_c", self.make_zone_sdf("zone_c", 0.55, 0.55, 0.55), -0.18, -0.22, 0.001)

        # 3. 05 Cube
        cubes = {
            "red_cube":    (1.0, 0.0, 0.0,  0.26, -0.15, 0.02),
            "yellow_cube": (1.0, 1.0, 0.0,  0.30,  0.00, 0.02),
            "blue_cube":   (0.0, 0.2, 1.0, -0.25,  0.00, 0.02), # Chiếm sẵn zone_b
            "green_cube":  (0.0, 0.9, 0.1,  0.26,  0.15, 0.02),
            "purple_cube": (0.6, 0.1, 0.9,  0.15,  0.25, 0.02)
        }

        for name, (r, g, b, x, y, z) in cubes.items():
            sdf = self.make_dynamic_cube_sdf(name, r, g, b)
            self.send_spawn_request(name, sdf, x, y, z)

        self.get_logger().info("=== NẠP MÔI TRƯỜNG THÀNH CÔNG ===")

def main():
    rclpy.init()
    node = SceneSpawner()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()