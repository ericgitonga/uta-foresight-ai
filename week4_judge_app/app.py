import json
import os
from typing import Literal
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("LLM_API_KEY", "ollama")
MODEL = os.getenv("LLM_MODEL", "qwen3:8b")

class ForecastEvaluation(BaseModel):
    score_plausibility: int = Field(ge=1, le=10, description="Plausibility score from 1 to 10")
    score_depth: int = Field(ge=1, le=10, description="Analytical depth and second-order impact from 1 to 10")
    score_novelty: int = Field(ge=1, le=10, description="Novelty of insight from 1 to 10")
    critique: str = Field(description="Specific analytical critique of the forecast")

class JudgeDecision(BaseModel):
    evaluation_a: ForecastEvaluation
    evaluation_b: ForecastEvaluation
    winner: Literal["Prompt A", "Prompt B", "Tie"]
    rationale: str = Field(description="Comparative synthesis explaining why the winning prompt produced superior foresight")

def get_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)

def generate_forecast(prompt_text: str) -> str:
    client = get_client()
    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a professional foresight scenario planner."},
            {"role": "user", "content": prompt_text}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    return (res.choices[0].message.content or "").strip()

def judge_and_rank(topic: str, prompt_a: str, prompt_b: str):
    client = get_client()

    print(f"Generating Forecast A for topic: {topic}...")
    forecast_a = generate_forecast(f"Topic: {topic}\nForesight Prompt: {prompt_a}\nGenerate a concise 1-paragraph foresight scenario for 2035.")

    print(f"Generating Forecast B for topic: {topic}...")
    forecast_b = generate_forecast(f"Topic: {topic}\nForesight Prompt: {prompt_b}\nGenerate a concise 1-paragraph foresight scenario for 2035.")

    judge_prompt = f"""You are an elite LLM Judge evaluating two strategic foresight scenario forecasts.
Topic: {topic}

Candidate A:
Prompt: {prompt_a}
Generated Scenario:
\"\"\"{forecast_a}\"\"\"

Candidate B:
Prompt: {prompt_b}
Generated Scenario:
\"\"\"{forecast_b}\"\"\"

Evaluate both according to the JSON schema:
{json.dumps(JudgeDecision.model_json_schema(), indent=2)}

Respond ONLY with valid JSON."""

    print("Submitting to LLM Judge...")
    judge_res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an objective foresight judge scoring on plausibility, depth, and novelty."},
            {"role": "user", "content": judge_prompt}
        ],
        temperature=0.2
    )

    content = (judge_res.choices[0].message.content or "").strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    data = json.loads(content)
    decision = JudgeDecision.model_validate(data)

    total_a = (
        decision.evaluation_a.score_plausibility
        + decision.evaluation_a.score_depth
        + decision.evaluation_a.score_novelty
    )
    total_b = (
        decision.evaluation_b.score_plausibility
        + decision.evaluation_b.score_depth
        + decision.evaluation_b.score_novelty
    )

    card_a = f"""### Forecast A
**Total Score: {total_a}/30**
- Plausibility: {decision.evaluation_a.score_plausibility}/10
- Depth: {decision.evaluation_a.score_depth}/10
- Novelty: {decision.evaluation_a.score_novelty}/10

**Critique:** {decision.evaluation_a.critique}

---
**Generated Scenario:**
{forecast_a}
"""

    card_b = f"""### Forecast B
**Total Score: {total_b}/30**
- Plausibility: {decision.evaluation_b.score_plausibility}/10
- Depth: {decision.evaluation_b.score_depth}/10
- Novelty: {decision.evaluation_b.score_novelty}/10

**Critique:** {decision.evaluation_b.critique}

---
**Generated Scenario:**
{forecast_b}
"""

    verdict_badge = f"🏆 Winner: {decision.winner} (A: {total_a} pts vs B: {total_b} pts)\n\n**Judge Rationale:**\n{decision.rationale}"

    return card_a, card_b, verdict_badge

def build_app():
    with gr.Blocks(title="Foresight LLM Judge & Ranking Studio") as demo:
        gr.Markdown("# 🔮 Strategic Foresight Prompt Judge & Ranking Studio")
        gr.Markdown(
            "Compare two prompt strategies side-by-side using an autonomous LLM Judge to evaluate plausibility, depth, and novelty."
        )

        with gr.Row():
            topic_input = gr.Textbox(
                label="Foresight Topic / Domain",
                value="Commercialization of Autonomous Drone Delivery in East Africa",
                lines=1
            )

        with gr.Row():
            with gr.Column():
                prompt_a_input = gr.Textbox(
                    label="Prompt Strategy A (e.g. Incremental / Regulatory focus)",
                    value="Focus strictly on regulatory licensing bottlenecks, air-traffic integration, and incremental logistics cost reduction.",
                    lines=3
                )
            with gr.Column():
                prompt_b_input = gr.Textbox(
                    label="Prompt Strategy B (e.g. Disruptive / Last-mile Leapfrogging)",
                    value="Explore radical leapfrogging in rural healthcare delivery, micro-payload agricultural supply chains, and autonomous swarm charging hubs.",
                    lines=3
                )

        run_btn = gr.Button("⚡ Run Forecasts & LLM Judge", variant="primary")

        verdict_output = gr.Markdown(label="Judge Verdict")

        with gr.Row():
            col_a = gr.Markdown(label="Candidate A")
            col_b = gr.Markdown(label="Candidate B")

        run_btn.click(
            fn=judge_and_rank,
            inputs=[topic_input, prompt_a_input, prompt_b_input],
            outputs=[col_a, col_b, verdict_output]
        )

    return demo

if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
