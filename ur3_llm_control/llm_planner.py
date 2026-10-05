import os
import json
import re
import yaml
import urllib.request
import urllib.error

class LLMPlanner:
    def __init__(self, student_cfg_path: str):
        with open(student_cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        self.student_name = cfg['student']['name']
        self.student_id = str(cfg['student']['id'])

        raw_url = os.getenv("ROUTER9_BASE_URL", "http://127.0.0.1:20128/v1").rstrip("/")
        if not raw_url.endswith("/v1"):
            raw_url += "/v1"
        self.base_url = raw_url
        self.api_key = os.getenv("ROUTER9_API_KEY", "9router")

        self.headers = {
            "Content-Type": "application/json",
            "Host": "localhost",
            "Authorization": f"Bearer {self.api_key}"
        }

        self.model = self._detect_model()
        print(f"[LLM Planner] Đang kết nối 9Router tại: {self.base_url}")
        print(f"[LLM Planner] Model được chọn: {self.model}")

        self.system_prompt = self._build_system_prompt()

    def _detect_model(self) -> str:
        custom_model = os.getenv("ROUTER9_MODEL")
        if custom_model:
            return custom_model

        try:
            req = urllib.request.Request(f"{self.base_url}/models", headers=self.headers, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m["id"] for m in data.get("data", [])]
                if "auto" in models:
                    return "auto"
                if models:
                    return models[0]
        except Exception:
            pass

        return "auto"

    def _build_system_prompt(self) -> str:
        last_2_digits = int(self.student_id[-2:])
        p_val = last_2_digits % 6
        
        mapping_rules = {
            0: "Zone A: red_cube, Zone B: yellow_cube, Zone C: blue_cube",
            1: "Zone A: red_cube, Zone B: blue_cube, Zone C: yellow_cube",
            2: "Zone A: yellow_cube, Zone B: red_cube, Zone C: blue_cube",
            3: "Zone A: yellow_cube, Zone B: blue_cube, Zone C: red_cube",
            4: "Zone A: blue_cube, Zone B: red_cube, Zone C: yellow_cube",
            5: "Zone A: blue_cube, Zone B: yellow_cube, Zone C: red_cube",
        }
        id_rule_desc = mapping_rules[p_val]

        return f"""You are a high-level Task Planner for a UR3/UR3e robot arm.
Translate user natural language commands and real-time vision environment state into a structured JSON execution plan.

STUDENT METADATA:
- Name: {self.student_name}
- Student ID: {self.student_id}
- Rule (MSSV mod 6 = {p_val}): {id_rule_desc}

ALLOWED SKILLS:
- home()
- pick(object)
- place(object, zone)

ALLOWED OBJECTS:
- red_cube, yellow_cube, blue_cube, green_cube, purple_cube

ALLOWED ZONES:
- zone_a, zone_b, zone_c, temp_position

JSON SCHEMA:
{{
  "plan": [
    {{"skill": "pick", "object": "<object_name>"}},
    {{"skill": "place", "object": "<object_name>", "zone": "<zone_or_temp>"}},
    {{"skill": "home"}}
  ]
}}

CRITICAL TASK PLANNING RULES:
1. Return ONLY pure valid JSON. No markdown backticks, no comments.
2. Every pick() MUST be paired immediately with a place().
3. Finish the plan with home().
4. DO NOT generate coordinates or joint angles.
5. CONFLICT RESOLUTION: If the destination zone is OCCUPIED by another block (indicated in VISION STATE), you MUST FIRST move that blocking cube to 'temp_position' (pick(blocking_cube) -> place(blocking_cube, 'temp_position')) BEFORE placing the requested cube into the target zone!
6. If the user mentions sorting/arranging by student ID (MSSV), apply Rule: {id_rule_desc}.
"""

    def plan(self, user_command: str, vision_context: str = "") -> dict:
        full_user_content = f"VISION STATE FROM CAMERA:\n{vision_context}\n\nUSER COMMAND: {user_command}"
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": full_user_content}
            ],
            "temperature": 0.1,
            "stream": False
        }

        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode("utf-8"), 
            headers=self.headers, 
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_text = resp.read().decode("utf-8").strip()
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"9Router trả về lỗi {e.code}: {err_body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Không thể kết nối mạng tới 9Router: {e}")

        decoder = json.JSONDecoder()
        try:
            result = json.loads(resp_text)
        except Exception:
            idx = resp_text.find('{')
            if idx != -1:
                result, _ = decoder.raw_decode(resp_text[idx:])
            else:
                raise RuntimeError(f"Phản hồi từ 9Router không chứa JSON: {resp_text[:200]}")

        raw_text = ""
        if isinstance(result, dict) and "choices" in result and len(result["choices"]) > 0:
            raw_text = result["choices"][0].get("message", {}).get("content", "").strip()
        else:
            raw_text = str(result)

        raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()

        idx = raw_text.find('{')
        while idx != -1:
            try:
                obj, end_idx = decoder.raw_decode(raw_text[idx:])
                if isinstance(obj, dict) and "plan" in obj:
                    return obj
                idx = raw_text.find('{', idx + end_idx)
            except Exception:
                idx = raw_text.find('{', idx + 1)

        raise ValueError(f"Không thể bóc tách JSON chứa 'plan' từ nội dung sau:\n{raw_text}")