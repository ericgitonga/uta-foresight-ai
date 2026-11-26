import json
import os
from typing import Literal
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("LLM_API_KEY", "ollama")
MODEL = os.getenv("LLM_MODEL", "qwen3:8b")

class ForesightSignal(BaseModel):
    topic: str = Field(description="Core subject or signal title")
    trend_category: Literal["Technology", "Economy", "Society", "Environment", "Policy", "Defense"] = Field(
        description="Foresight STEEP category"
    )
    horizon: Literal["Short-term (1-3 yrs)", "Medium-term (3-5 yrs)", "Long-term (5-10+ yrs)"] = Field(
        description="Expected time horizon of impact"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    key_drivers: list[str] = Field(description="Primary forces or technological drivers accelerating this trend")
    implications: str = Field(description="Strategic implications for foresight planning")

def extract_signal(text: str) -> ForesightSignal:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    schema_json = json.dumps(ForesightSignal.model_json_schema(), indent=2)

    prompt = f"""You are an expert strategic foresight analyst.
Analyze the following text and extract a structured foresight signal according to this JSON Schema:
{schema_json}

Input Text:
\"\"\"{text}\"\"\"

Respond ONLY with a valid JSON object matching the schema. Do not enclose in markdown blocks if possible, or provide raw JSON."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a precise foresight extraction system that outputs valid JSON conforming to the requested schema."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content or ""
    # Strip potential markdown formatting if returned
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    data = json.loads(content)
    return ForesightSignal.model_validate(data)

if __name__ == "__main__":
    sample_text = (
        "Global shipping operators and municipal ports in Northern Europe have initiated commercial trials of "
        "fully autonomous, ammonia-powered feeder vessels scheduled for phased coastal deployment by 2028. "
        "Regulatory harmonization from the International Maritime Organization (IMO) remains the decisive bottleneck, "
        "though port terminal automation integration has already achieved 99.4% dockside mooring reliability."
    )

    print(f"Analyzing sample foresight text using model [{MODEL}] at [{BASE_URL}]...\n")
    signal = extract_signal(sample_text)
    print("=== Validated ForesightSignal Output (JSON) ===")
    print(signal.model_dump_json(indent=2))
