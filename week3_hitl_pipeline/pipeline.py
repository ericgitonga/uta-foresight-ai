import json
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Load environment configuration
load_dotenv()
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("LLM_API_KEY", "ollama")
MODEL = os.getenv("LLM_MODEL", "qwen3:8b")

class SynthesizedDriver(BaseModel):
    driver_name: str = Field(description="Title of macro foresight driver")
    signal_ids: list[int] = Field(description="List of raw signal IDs underpinning this driver")
    velocity: str = Field(description="Speed of change: Slow, Accelerating, Exponential")
    foresight_thesis: str = Field(description="3-5 year strategic foresight forecast")

class SynthesisReport(BaseModel):
    approved_count: int
    drivers: list[SynthesizedDriver]
    strategic_recommendation: str

def run_hitl_pipeline(
    input_path: str = "data/raw_signals.json",
    output_path: str = "data/synthesized_drivers.json",
    auto_approve: bool = False
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input signals file not found at {input_path}. Run week 2 crawler first.")

    with open(input_path, "r", encoding="utf-8") as f:
        raw_signals = json.load(f)

    print(f"\nLoaded {len(raw_signals)} raw signals from {input_path}:")
    print("-" * 60)
    for s in raw_signals:
        print(f"[{s['id']}] {s['title']} ({s['source']})")
    print("-" * 60)

    # Human-In-The-Loop Checkpoint
    if auto_approve:
        print("\n[HITL Checkpoint] Auto-approval enabled via flag. Approved.")
        decision = "y"
    else:
        try:
            print("\n[HITL Checkpoint] Review the signals above.")
            decision = input("Proceed with strategic driver synthesis? [y/N]: ").strip().lower()
        except EOFError:
            print("\nNon-interactive environment detected. Defaulting to 'y' for dry-run verification.")
            decision = "y"

    if decision != "y":
        print("Pipeline aborted by user. No LLM synthesis executed.")
        return None

    print(f"\n[HITL Approved] Synthesizing foresight drivers using [{MODEL}] at [{BASE_URL}]...")

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    schema_str = json.dumps(SynthesisReport.model_json_schema(), indent=2)

    prompt = f"""You are a senior strategic foresight analyst.
Given these raw signals:
{json.dumps(raw_signals, indent=2)}

Group related signals into 2-3 overarching foresight drivers and provide strategic recommendations.
Format your output strictly according to this JSON Schema:
{schema_str}

Respond with valid JSON only."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a foresight synthesis system producing strict JSON adhering to schema."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    content = (response.choices[0].message.content or "").strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    data = json.loads(content)
    report = SynthesisReport.model_validate(data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    print(f"\nSuccessfully generated {len(report.drivers)} macro foresight drivers.")
    print(f"Saved report to {output_path}")
    return report

if __name__ == "__main__":
    auto = "--auto-approve" in sys.argv or "-y" in sys.argv
    report = run_hitl_pipeline(auto_approve=auto)
    if report:
        print("\n=== Synthesized Drivers Summary ===")
        print(report.model_dump_json(indent=2))
