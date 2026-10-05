class PlanValidator:
    ALLOWED_SKILLS = ["home", "pick", "place"]
    ALLOWED_OBJECTS = [
        "red_cube", "yellow_cube", "blue_cube", "green_cube", "purple_cube"
    ]
    ALLOWED_ZONES = ["zone_a", "zone_b", "zone_c", "temp_position"]

    @classmethod
    def validate(cls, plan_json: dict) -> tuple[bool, str]:
        if not isinstance(plan_json, dict) or "plan" not in plan_json:
            return False, "Thiếu trường 'plan' ở tầng cao nhất của JSON."

        steps = plan_json.get("plan", [])
        if not isinstance(steps, list) or len(steps) == 0:
            return False, "Kế hoạch (plan) rỗng hoặc không phải dạng danh sách."

        for i, step in enumerate(steps):
            skill = step.get("skill")
            if skill not in cls.ALLOWED_SKILLS:
                return False, f"Bước {i+1}: Skill '{skill}' không hợp lệ."

            if skill == "pick":
                obj = step.get("object")
                if obj not in cls.ALLOWED_OBJECTS:
                    return False, f"Bước {i+1}: Object '{obj}' không hợp lệ cho lệnh pick."
            elif skill == "place":
                obj = step.get("object")
                zone = step.get("zone")
                if obj not in cls.ALLOWED_OBJECTS:
                    return False, f"Bước {i+1}: Object '{obj}' không hợp lệ cho lệnh place."
                if zone not in cls.ALLOWED_ZONES:
                    return False, f"Bước {i+1}: Zone/Target '{zone}' không hợp lệ cho lệnh place."

        return True, "Hợp lệ"