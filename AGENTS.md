# Course Project Rules
- Always load LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL from .env using python-dotenv.
- Never hardcode localhost:11434 in Python scripts.
- Use Pydantic BaseModel schemas for structured foresight extraction and judge scoring.
- Decouple Playwright crawlers (save raw JSON to data/) from LLM processing scripts.
- Keep human-in-the-loop (HITL) checkpoints explicit with terminal or Gradio approval prompts.
