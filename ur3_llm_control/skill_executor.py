import os
import sys
import threading
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from ur3_llm_control.llm_planner import LLMPlanner
from ur3_llm_control.task_validator import PlanValidator
from ur3_llm_control.robot_skills import RobotSkills

try:
    if hasattr(sys.stdin, 'reconfigure'):
        sys.stdin.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

class SkillExecutorNode(Node):
    def __init__(self):
        # BẬT USE_SIM_TIME ĐỂ ĐỒNG BỘ ĐỒNG HỒ VỚI GAZEBO
        super().__init__(
            'skill_executor_node',
            parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)]
        )
        
        pkg_share = get_package_share_directory('ur3_llm_control')
        student_cfg = os.path.join(pkg_share, 'config', 'student_config.yaml')
        scene_cfg = os.path.join(pkg_share, 'config', 'scene.yaml')

        self.planner = LLMPlanner(student_cfg)
        self.skills = RobotSkills(self, scene_cfg)

    def run_command(self, user_command: str):
        print("\n" + "="*50)
        print("USER COMMAND:")
        print(f"{user_command}\n")

        # 1. ĐỌC THÔNG TIN TỪ CAMERA PERCEPTION
        vision_state = self.skills.get_vision_world_state()
        vision_context_str = "- Object locations: " + ", ".join([f"{k} at ({v['x']}, {v['y']})" for k, v in vision_state['objects'].items()]) + "\n"
        for z_name, info in vision_state['zones_status'].items():
            if info['occupied']:
                vision_context_str += f"- {z_name}: OCCUPIED by {info['occupier']}\n"
            else:
                vision_context_str += f"- {z_name}: FREE\n"
        vision_context_str += f"- Available staging area: 'temp_position' at ({vision_state['temp_position']['x']}, {vision_state['temp_position']['y']})\n"

        print("CAMERA & SCENE PERCEPTION:")
        print(vision_context_str)

        # 2. GỬI PROMPT KÈM NGỮ CẢNH VISION CHO LLM
        try:
            plan_json = self.planner.plan(user_command, vision_context=vision_context_str)
        except Exception as e:
            print(f"Lỗi khi gọi LLM: {e}")
            return

        is_valid, reason = PlanValidator.validate(plan_json)
        if not is_valid:
            print(f"TASK REJECTED: Kế hoạch không hợp lệ ({reason})")
            return

        print("LLM PLAN:")
        for step in plan_json.get('plan', []):
            skill = step.get('skill')
            if skill == 'home':
                print("home()")
            elif skill == 'pick':
                print(f"pick({step.get('object')})")
            elif skill == 'place':
                print(f"place({step.get('object')}, {step.get('zone')})")
        print("\nEXECUTION:")

        all_success = True
        for step in plan_json.get('plan', []):
            skill = step.get('skill')
            status = "FAILED"

            if skill == 'home':
                step_str = "home()"
                status = self.skills.home()
            elif skill == 'pick':
                obj = step.get('object')
                step_str = f"pick({obj})"
                status = self.skills.pick(obj)
            elif skill == 'place':
                obj = step.get('object')
                zone = step.get('zone')
                step_str = f"place({obj}, {zone})"
                status = self.skills.place(obj, zone)

            dots = "." * max(2, 35 - len(step_str))
            print(f"{step_str} {dots} {status}")

            if status != "SUCCESS":
                all_success = False
                break

        print()
        if all_success:
            print("TASK SUCCESS")
        else:
            print("TASK FAILED")
        print("="*50 + "\n")

def safe_input(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except UnicodeDecodeError:
        raw = sys.stdin.buffer.readline()
        return raw.decode('utf-8', errors='ignore').strip()

def main(args=None):
    rclpy.init(args=args)
    executor = SkillExecutorNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(executor,), daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            cmd = safe_input("Nhập lệnh ngôn ngữ tự nhiên (hoặc 'exit' để thoát): ")
            if not cmd:
                continue
            if cmd.lower() in ['exit', 'quit']:
                break
            executor.run_command(cmd)
    except KeyboardInterrupt:
        pass
    finally:
        executor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()