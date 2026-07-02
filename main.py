import requests
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
# ---- Ollama client (as provided) ----
OLLAMA_HOST = "http://localhost:11434"
CHAT_MODEL = "llama3.2:1b"
EMBED_MODEL = "qwen3-embedding:0.6b"   # change to your exact local tag if different

class OllamaClient:
    def __init__(self, host: str = OLLAMA_HOST,
                 chat_model: str = CHAT_MODEL,
                 embed_model: str = EMBED_MODEL):
        self.host = host.rstrip("/")
        self.chat_model = chat_model
        self.embed_model = embed_model

    def chat(self, messages: List[Dict[str, str]],
             temperature: float = 0.7,
             top_p: float = 0.9,
             stream: bool = False,
             options: Optional[dict] = None) -> str:
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                **(options or {}),
            },
        }
        resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]

    def embed(self, text: str) -> List[float]:
        payload = {"model": self.embed_model, "prompt": text}
        resp = requests.post(f"{self.host}/api/embeddings", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["embedding"]

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]
SYSTEM_PROMPT = """You are a certified fitness coach creating a personalized workout plan.

You will be given five fields: goal, fitness level, equipment, days per week, session duration.

Think through this step by step BEFORE writing the final plan:
Step 1: Restate the goal and what training style suits it best.
Step 2: Consider the fitness level and what intensity/complexity is safe and effective.
Step 3: Consider the equipment and list what exercise categories are possible with it.
Step 4: Consider days per week and session duration, and decide how to split training focus across days.
Step 5: Combine all reasoning above to design the final day-by-day plan.

After completing your reasoning, output the final plan under a line that says exactly:
FINAL PLAN:

The FINAL PLAN section must:
- Have one clearly labeled block per day (Day 1, Day 2, etc.) matching the days per week given.
- Include specific exercises with sets/reps or duration according to given requirements by the user.
- Match difficulty to the given fitness level.
- Be in clean plain text, avoiding the usage markdown symbols like ** or ##.
- Sound Optimistic and realistic, encouraging the user to follow the plan and achieve their goal.
Show your Step 1-5 reasoning above the FINAL PLAN line, then the plan below it."""
client = OllamaClient()

def generate_plan(goal: str, level: str, equipment: str, days: str, duration: str) -> str:
    user_content = (
        f"Goal: {goal}\n"
        f"Fitness level: {level}\n"
        f"Equipment: {equipment}\n"
        f"Days per week: {days}\n"
        f"Session duration: {duration} minutes"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return client.chat(messages, temperature=0.5).strip()

def split_reasoning_and_plan(full_response: str):
    # Case-insensitive check to ensure small local models don't break the split logic
    lower_response = full_response.lower()
    marker = "final plan:"
    
    if marker in lower_response:
        # Find the starting index of the split marker dynamically
        split_idx = lower_response.find(marker)
        reasoning = full_response[:split_idx].strip()
        # Strip out the actual text marker length to get just the plan content
        plan = full_response[split_idx + len(marker):].strip()
        return reasoning, plan
        
    return "", full_response.strip()

def calculate_intensity_schedule(level: str, total_weeks: int = 6) -> pd.DataFrame:
    weeks = np.arange(1, total_weeks + 1)
    base_rpe = {"beginner": 5.5, "intermediate": 6.5, "advanced": 7.0}
    start_rpe = base_rpe.get(level.lower().strip(), 6.0)
    rpe_trend = np.linspace(start_rpe, start_rpe + 1.5, total_weeks)
    rpe_trend[-1] -= 2.0  # Deload drop on the final week
    volume_multiplier = np.clip(1.0 + (weeks - 1) * 0.05, 1.0, 1.3)
    volume_multiplier[-1] = 0.85 # Cut volume significantly for the deload week
    data = {
        "Week": weeks,
        "Target RPE (1-10)": np.round(rpe_trend, 1),
        "Volume Multiplier": np.round(volume_multiplier, 2),
        "Phase Focus": ["Acclimation" if w == 1 else "Deload/Recovery" if w == total_weeks else "Overload" for w in weeks]
    }
    
    return pd.DataFrame(data).set_index("Week")
# ---- Traditional 5 inputs / single output ----
goal = input("Goal: ")
level = input("Fitness level: ")
equipment = input("Equipment: ")
days = input("Days per week: ")
duration = input("Session duration (minutes): ")

full_response = generate_plan(goal, level, equipment, days, duration)
reasoning, plan = split_reasoning_and_plan(full_response)
intensity_df = calculate_intensity_schedule(level, total_weeks=6)
reasoning_clean = reasoning.replace("**", "").replace("* ", "- ")
plan_clean = plan.replace("**", "")

print("\n--- Reasoning ---\n")
print(reasoning_clean)
print("\n--- Workout Plan ---\n")
print(plan_clean)
print("\n--- Intensity Schedule ---\n")
print(intensity_df.to_string())

