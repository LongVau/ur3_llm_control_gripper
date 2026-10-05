import time
import yaml
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Pose
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# Service điều khiển trạng thái vật lý thực tế trong Gazebo
from gazebo_msgs.srv import SetEntityState
from tf2_ros import Buffer, TransformListener

from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath
from moveit_msgs.msg import (
    Constraints, PositionConstraint, OrientationConstraint, 
    JointConstraint, CollisionObject, AttachedCollisionObject,
    PlanningScene
)

class RobotSkills:
    def __init__(self, node: Node, scene_cfg_path: str):
        self.node = node
        with open(scene_cfg_path, 'r', encoding='utf-8') as f:
            self.scene = yaml.safe_load(f)['scene']

        self.bridge = CvBridge()
        self.latest_cv_image = None
        self.held_object = None

        # 1. Đăng ký nhận ảnh Camera
        self._cam_sub = self.node.create_subscription(
            Image, '/camera/image_raw', self._camera_callback, 10
        )

        # 2. Quản lý TF để bám theo tọa độ đầu kẹp tool0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

        # 3. Kết nối service Gazebo để gắp và mang vật thật trong không gian 3D
        self._set_entity_client = self.node.create_client(SetEntityState, '/set_entity_state')
        if not self._set_entity_client.wait_for_service(timeout_sec=1.5):
            self._set_entity_client = self.node.create_client(SetEntityState, '/gazebo/set_entity_state')

        self._marker_pub = self.node.create_publisher(MarkerArray, '/rviz_scene_markers', 10)
        self._apply_scene_client = self.node.create_client(ApplyPlanningScene, '/apply_planning_scene')
        self._planning_scene_pub = self.node.create_publisher(
            AttachedCollisionObject, '/attached_collision_object', 10
        )

        # 4. MoveIt Action Clients
        self._action_client = ActionClient(self.node, MoveGroup, 'move_action')
        self._cartesian_client = self.node.create_client(GetCartesianPath, '/compute_cartesian_path')
        self._execute_client = ActionClient(self.node, ExecuteTrajectory, 'execute_trajectory')

        self.node.get_logger().info("Đang kết nối tới MoveIt 2 Services & Actions...")
        self._action_client.wait_for_server()
        self._execute_client.wait_for_server()
        self.node.get_logger().info("MoveIt 2 & Gazebo Gripper Sync đã sẵn sàng!")

        self.object_poses = dict(self.scene['objects'])

        time.sleep(0.5)
        self._init_planning_scene()
        self._marker_timer = self.node.create_timer(0.1, self._publish_rviz_markers)
        
        # 5. Timer 50Hz liên tục giữ khối hộp bám sát ngón kẹp trong Gazebo khi đang gắp
        self._sync_timer = self.node.create_timer(0.02, self._sync_held_object_in_gazebo)

    def _camera_callback(self, msg):
        try:
            self.latest_cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            pass

    def _sync_held_object_in_gazebo(self):
        """Khóa vị trí khối cube vào giữa 2 ngón kẹp trong Gazebo theo thời gian thực (50Hz)."""
        if self.held_object is not None:
            try:
                t = self.tf_buffer.lookup_transform("world", "tool0", rclpy.time.Time())
                req = SetEntityState.Request()
                req.state.name = self.held_object
                # Khối cube nằm ngay giữa khe 2 ngón kẹp (dưới tool0 7cm)
                req.state.pose.position.x = t.transform.translation.x
                req.state.pose.position.y = t.transform.translation.y
                req.state.pose.position.z = max(0.02, t.transform.translation.z - 0.075)
                req.state.pose.orientation = t.transform.rotation
                req.state.reference_frame = "world"
                self._set_entity_client.call_async(req)
                
                # Đồng bộ tọa độ cho RViz
                self.object_poses[self.held_object] = {
                    'x': req.state.pose.position.x,
                    'y': req.state.pose.position.y,
                    'z': req.state.pose.position.z
                }
            except Exception:
                pass

    def set_gazebo_object_pose(self, name: str, x: float, y: float, z: float):
        """Đặt vị trí ổn định cho vật trong Gazebo."""
        req = SetEntityState.Request()
        req.state.name = name
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = float(z)
        req.state.pose.orientation.w = 1.0
        req.state.reference_frame = "world"
        self._set_entity_client.call_async(req)

    def get_vision_world_state(self) -> dict:
        state = {
            "objects": dict(self.object_poses),
            "zones_status": {},
            "temp_position": self.scene['temp_position']
        }

        if self.latest_cv_image is not None:
            hsv = cv2.cvtColor(self.latest_cv_image, cv2.COLOR_BGR2HSV)
            color_ranges = {
                "red_cube": ((0, 100, 100), (10, 255, 255)),
                "blue_cube": ((100, 100, 100), (130, 255, 255)),
                "yellow_cube": ((20, 100, 100), (35, 255, 255)),
                "green_cube": ((40, 100, 100), (80, 255, 255)),
                "purple_cube": ((135, 100, 100), (160, 255, 255)),
            }
            H, W, _ = self.latest_cv_image.shape
            scale_x = 1.6 / W
            scale_y = 1.2 / H

            for c_name, (lower, upper) in color_ranges.items():
                # Không cập nhật từ camera nếu vật đang được gắp trên tay
                if self.held_object == c_name:
                    continue
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(c) > 50:
                        M = cv2.moments(c)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            world_x = float(np.round(0.05 + (H/2 - cy) * scale_y, 2))
                            world_y = float(np.round((W/2 - cx) * scale_x, 2))
                            self.object_poses[c_name] = {'x': world_x, 'y': world_y, 'z': 0.02}

        for z_name, z_pos in self.scene['zones'].items():
            occupier = None
            for obj_name, o_pos in self.object_poses.items():
                if self.held_object == obj_name:
                    continue
                dist = np.hypot(z_pos['x'] - o_pos['x'], z_pos['y'] - o_pos['y'])
                if dist < 0.08:
                    occupier = obj_name
                    break
            state["zones_status"][z_name] = {
                "occupied": occupier is not None,
                "occupier": occupier
            }
        return state

    def _wait_for_future(self, future, timeout=30.0):
        start_t = time.time()
        while not future.done():
            if time.time() - start_t > timeout:
                return None
            time.sleep(0.02)
        return future.result()

    def _publish_rviz_markers(self):
        ma = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        ma.markers.append(clear_marker)

        m_id = 1
        def make_marker(shape, x, y, z, sx, sy, sz, r, g, b, text=None):
            nonlocal m_id
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp.sec = 0
            m.header.stamp.nanosec = 0
            m.ns = "scene_objects"
            m.id = m_id
            m_id += 1
            m.type = shape
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = float(z)
            m.pose.orientation.w = 1.0
            m.scale.x = float(sx)
            m.scale.y = float(sy)
            m.scale.z = float(sz)
            m.color.r = float(r)
            m.color.g = float(g)
            m.color.b = float(b)
            m.color.a = 0.85
            if text:
                m.text = text
            return m

        for z_name, pos in self.scene['zones'].items():
            ma.markers.append(make_marker(Marker.CUBE, pos['x'], pos['y'], 0.002, 0.12, 0.12, 0.002, 0.4, 0.4, 0.4))
            label = z_name.replace("zone_", "Zone ").upper()
            ma.markers.append(make_marker(Marker.TEXT_VIEW_FACING, pos['x'], pos['y'], 0.08, 0.04, 0.04, 0.04, 1.0, 1.0, 1.0, text=label))

        t_pos = self.scene['temp_position']
        ma.markers.append(make_marker(Marker.CUBE, t_pos['x'], t_pos['y'], 0.002, 0.10, 0.10, 0.002, 0.2, 0.7, 0.2))
        ma.markers.append(make_marker(Marker.TEXT_VIEW_FACING, t_pos['x'], t_pos['y'], 0.08, 0.04, 0.04, 0.04, 0.4, 1.0, 0.4, text="TEMP"))

        colors = {
            "red_cube": (1.0, 0.0, 0.0),
            "yellow_cube": (1.0, 0.9, 0.0),
            "blue_cube": (0.0, 0.3, 1.0),
            "green_cube": (0.0, 0.85, 0.1),
            "purple_cube": (0.7, 0.1, 0.9)
        }
        for name, pos in self.object_poses.items():
            c = colors.get(name, (1.0, 1.0, 1.0))
            ma.markers.append(make_marker(Marker.CUBE, pos['x'], pos['y'], pos['z'], 0.04, 0.04, 0.04, c[0], c[1], c[2]))

        self._marker_pub.publish(ma)

    def _init_planning_scene(self):
        scene_msg = PlanningScene()
        scene_msg.is_diff = True

        floor = CollisionObject()
        floor.header.frame_id = "world"
        floor.id = "floor_plane"
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [2.0, 2.0, 0.02]
        p = Pose()
        p.position.z = -0.05
        p.orientation.w = 1.0
        floor.primitives.append(box)
        floor.primitive_poses.append(p)
        floor.operation = CollisionObject.ADD
        scene_msg.world.collision_objects.append(floor)

        req = ApplyPlanningScene.Request()
        req.scene = scene_msg
        future = self._apply_scene_client.call_async(req)
        self._wait_for_future(future, timeout=5.0)

    def _move_cartesian_linear(self, target_x: float, target_y: float, target_z: float) -> bool:
        req = GetCartesianPath.Request()
        req.header.frame_id = "world"
        req.group_name = "ur_manipulator"
        req.link_name = "tool0"

        target_pose = Pose()
        target_pose.position.x = float(target_x)
        target_pose.position.y = float(target_y)
        target_pose.position.z = float(target_z)
        target_pose.orientation.x = 1.0
        target_pose.orientation.y = 0.0
        target_pose.orientation.z = 0.0
        target_pose.orientation.w = 0.0

        req.waypoints.append(target_pose)
        req.max_step = 0.008
        req.jump_threshold = 0.0
        req.avoid_collisions = True

        future = self._cartesian_client.call_async(req)
        res = self._wait_for_future(future, timeout=5.0)

        if not res or res.fraction < 0.85:
            return False

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = res.solution
        send_future = self._execute_client.send_goal_async(goal)
        gh = self._wait_for_future(send_future, timeout=5.0)
        if not gh or not gh.accepted:
            return False

        res_future = gh.get_result_async()
        exec_res = self._wait_for_future(res_future, timeout=20.0)
        return exec_res is not None and exec_res.result.error_code.val == 1

    def _send_pose_goal(self, x: float, y: float, z: float) -> str:
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = "ur_manipulator"
        goal_msg.request.num_planning_attempts = 15
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 0.20
        goal_msg.request.max_acceleration_scaling_factor = 0.20
        goal_msg.request.start_state.is_diff = True

        c = Constraints()
        pc = PositionConstraint()
        pc.header.frame_id = "world"
        pc.link_name = "tool0"
        pc.weight = 1.0

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [0.015]

        target_pose = PoseStamped()
        target_pose.header.frame_id = "world"
        target_pose.pose.position.x = float(x)
        target_pose.pose.position.y = float(y)
        target_pose.pose.position.z = float(z)

        pc.constraint_region.primitives.append(primitive)
        pc.constraint_region.primitive_poses.append(target_pose.pose)
        c.position_constraints.append(pc)

        oc = OrientationConstraint()
        oc.header.frame_id = "world"
        oc.link_name = "tool0"
        oc.orientation.x = 1.0
        oc.orientation.y = 0.0
        oc.orientation.z = 0.0
        oc.orientation.w = 0.0
        oc.absolute_x_axis_tolerance = 0.15
        oc.absolute_y_axis_tolerance = 0.15
        oc.absolute_z_axis_tolerance = 0.70
        oc.weight = 1.0
        c.orientation_constraints.append(oc)

        goal_msg.request.goal_constraints.append(c)

        send_goal_future = self._action_client.send_goal_async(goal_msg)
        goal_handle = self._wait_for_future(send_goal_future, timeout=8.0)

        if not goal_handle or not goal_handle.accepted:
            return "PLANNING_FAILED"

        res_future = goal_handle.get_result_async()
        res = self._wait_for_future(res_future, timeout=25.0)

        if not res or res.result.error_code.val != 1:
            return "FAILED"
        return "SUCCESS"

    def home(self) -> str:
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = "ur_manipulator"
        goal_msg.request.start_state.is_diff = True
        goal_msg.request.num_planning_attempts = 15
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 0.25
        goal_msg.request.max_acceleration_scaling_factor = 0.25

        c = Constraints()
        joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        for name, val in zip(joint_names, self.scene['home_joints']):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(val)
            jc.tolerance_above = 0.05
            jc.tolerance_below = 0.05
            jc.weight = 1.0
            c.joint_constraints.append(jc)

        goal_msg.request.goal_constraints.append(c)
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        goal_handle = self._wait_for_future(send_goal_future, timeout=8.0)

        if not goal_handle or not goal_handle.accepted:
            return "PLANNING_FAILED"

        res_future = goal_handle.get_result_async()
        res = self._wait_for_future(res_future, timeout=25.0)

        if not res or res.result.error_code.val != 1:
            return "FAILED"
        return "SUCCESS"

    def open_gripper(self):
        time.sleep(0.3)

    def close_gripper(self):
        time.sleep(0.3)

    def pick(self, obj_name: str) -> str:
        if obj_name not in self.object_poses:
            return "INVALID_OBJECT"
        pos = self.object_poses[obj_name]

        # 1. Bay ngang đến phía trên khối hộp
        if self._send_pose_goal(pos['x'], pos['y'], pos['z'] + self.scene['z_offset_approach']) != "SUCCESS":
            return "FAILED"
        self.open_gripper()
        
        # 2. Hạ thẳng đứng xuống sát khối hộp để gắp
        if not self._move_cartesian_linear(pos['x'], pos['y'], pos['z'] + self.scene['z_offset_grasp']):
            if self._send_pose_goal(pos['x'], pos['y'], pos['z'] + self.scene['z_offset_grasp']) != "SUCCESS":
                return "FAILED"
        
        self.close_gripper()
        
        # BẮT ĐẦU GIỮ VẬT THẬT: Gazebo kích hoạt khóa khối cube vào giữa 2 ngón kẹp
        self.held_object = obj_name
        time.sleep(0.3)

        # 3. Nhấc THẲNG ĐỨNG lên cao (Khối cube trong Gazebo sẽ bay theo lên cao!)
        if not self._move_cartesian_linear(pos['x'], pos['y'], pos['z'] + self.scene['z_offset_approach']):
            if self._send_pose_goal(pos['x'], pos['y'], pos['z'] + self.scene['z_offset_approach']) != "SUCCESS":
                return "FAILED"
        return "SUCCESS"

    def place(self, obj_name: str, zone_name: str) -> str:
        target_pos = None
        if zone_name in self.scene['zones']:
            target_pos = self.scene['zones'][zone_name]
        elif zone_name == 'temp_position':
            target_pos = self.scene['temp_position']
        else:
            return "FAILED"

        # 1. Bay ngang qua Zone đích (Khối cube bay theo trên không trung trong Gazebo)
        if self._send_pose_goal(target_pos['x'], target_pos['y'], target_pos['z'] + self.scene['z_offset_approach']) != "SUCCESS":
            return "FAILED"
        
        # 2. Hạ thẳng đứng xuống mặt đất để đặt
        if not self._move_cartesian_linear(target_pos['x'], target_pos['y'], target_pos['z'] + self.scene['z_offset_grasp']):
            if self._send_pose_goal(target_pos['x'], target_pos['y'], target_pos['z'] + self.scene['z_offset_grasp']) != "SUCCESS":
                return "FAILED"

        # THẢ VẬT: Ngừng bám theo tay, đặt khối cube yên vị trên sàn Gazebo
        self.held_object = None
        self.object_poses[obj_name] = dict(target_pos)
        self.set_gazebo_object_pose(obj_name, target_pos['x'], target_pos['y'], 0.02)
        
        self.open_gripper()
        time.sleep(0.3)

        # 3. Rút tay thẳng đứng lên an toàn một mình
        if not self._move_cartesian_linear(target_pos['x'], target_pos['y'], target_pos['z'] + self.scene['z_offset_approach']):
            if self._send_pose_goal(target_pos['x'], target_pos['y'], target_pos['z'] + self.scene['z_offset_approach']) != "SUCCESS":
                return "FAILED"
        return "SUCCESS"