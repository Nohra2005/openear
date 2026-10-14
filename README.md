---
title: OpenEar Maguy
emoji: 🎧
colorFrom: green
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
---

# OpenEar Business Agent (Maguy)

**Live demo:** [huggingface.co/spaces/TatianaNohra/openear-maguy](https://huggingface.co/spaces/TatianaNohra/openear-maguy)

**Demo video:** [demo/openear_demo_captioned.mp4](demo/openear_demo_captioned.mp4) (4 min, captioned walkthrough of the notebook and the live app)
Maguy is the AI sales assistant of **OpenEar**, a fictional Lebanese online store for open-ear sports headphones.
She answers questions from the store's own business files, recommends a model based on the customer's sport,
collects leads, and logs questions she can't answer for the team.

## How it works

Each customer message goes through four steps:

1. **Router:** a Gemini call labels the message `general`, `lead` or `feedback`.
2. **Agent:** Gemini answers using the persona, instructions and business files, and can call two tools:
   - `record_customer_interest(email, name, message)` saves a lead (the email is validated in code)
   - `record_feedback(question)` logs unanswered questions, complaints and suggestions
3. **Reflection:** a second Gemini call reviews the draft for invented facts, false promises or assumptions.
4. **Rewrite:** if the review rejects the draft, Maguy rewrites it once using the feedback.

## Files

| File | Purpose |
|---|---|
| `business_agent.ipynb` | Step-by-step notebook: setup, persona, instructions, router, tools, chat loop, reflection, Gradio |
| `app.py` | The same agent as a standalone Gradio app (used by the Hugging Face Space) |
| `me/business_summary.txt` | Short business summary |
| `me/about_business.pdf` | Full business profile: products, prices, policies, team, FAQ |
| `requirements.txt` | Python dependencies |
| `demo/openear_demo_captioned.mp4` | Captioned demo video |
| `screenshots/` | Before-the-fixes screenshots (invalid email accepted, invented reservation) |

## Run locally

```bash
pip install -r requirements.txt
# create a .env file containing: GEMINI_API_KEY=your_key
python app.py
```

On Hugging Face, add `GEMINI_API_KEY` as a secret in the Space settings.
