"""OpenEar business agent (Maguy): a Gemini-powered sales assistant for an open-ear sports headphone store.

Pipeline per customer message: router (general / lead / feedback) -> Gemini with tools
(record_customer_interest, record_feedback) -> reflection review -> rewrite if rejected.
Run locally with `python app.py`, or deploy as a Hugging Face Space (set GEMINI_API_KEY as a secret).
"""
import json
import os
import re
import time
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pypdf import PdfReader


# ---------- Setup ----------
load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found. Set it in .env locally or as a Space secret on Hugging Face.")

client = genai.Client(api_key=api_key)
MODEL = "gemini-3.5-flash-lite"

# Load the business knowledge the agent is allowed to use.
# Files live in me/ locally; on Hugging Face they can also sit next to app.py.
def find_file(name):
    for path in (os.path.join("me", name), name):
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{name} not found in me/ or the app folder")


with open(find_file("business_summary.txt"), encoding="utf-8") as f:
    summary = f.read()

reader = PdfReader(find_file("about_business.pdf"))
profile = "\n".join(page.extract_text() or "" for page in reader.pages)


# Resilient Gemini call: retry on overload (503), switch model immediately on rate limit (429)
FALLBACK_MODELS = [MODEL, "gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.8-flash"]


def generate(contents, config=None, retries=2):
    last_error = None
    for model in dict.fromkeys(FALLBACK_MODELS):  # removes duplicates, keeps order
        for attempt in range(retries):
            try:
                return client.models.generate_content(model=model, contents=contents, config=config)
            except (errors.ServerError, errors.ClientError) as e:
                code = getattr(e, "code", None)
                if code not in (429, 500, 503):
                    raise  # real errors (bad key, bad request) still surface
                last_error = e
                if code == 429:
                    print(f"[{model} rate limit, switching model]")
                    break  # quota used up: retrying the same model won't help
                print(f"[{model} busy ({code}), retrying...]")
                time.sleep(2 * (attempt + 1))
    raise last_error


# ---------- Persona ----------
# === PERSONA ===
PERSONA = """
You are Maguy, the sales assistant of OpenEar, a Lebanese online store for open-ear sports headphones.
You help athletes find the right headphones for their sport and guide them toward placing an order.
You are concise, friendly and sporty: short answers, no long paragraphs.
You match the customer's tone and language, replying in English, Arabic or Arabizi depending on how they write.
"""


# ---------- Instructions ----------
# === INSTRUCTIONS ===
INSTRUCTIONS = """
1. Only use facts from the OpenEar business summary and business profile provided below.
2. Never invent prices, discounts, specs, stock, delivery times or policies.
3. Only answer questions about OpenEar's products, choosing headphones for a sport, orders, delivery, payment,
   warranty, returns and contact details. Politely decline anything else and steer back to OpenEar.
4. If a question is about OpenEar but the answer is not in the files, call record_feedback with the question,
   then tell the customer the team will follow up and share support@openear.store or @openear.lb on Instagram.
5. Before recommending a model, ask which sport the customer trains for if they haven't said.
   Then recommend: Stride for casual running, Pace Pro for serious running, trail and cycling,
   Swim for swimming and triathlon, Ultra for long-distance and ultra events.
6. When a customer wants to buy, order or reserve, or asks about availability, ask for their name and email,
   then call record_customer_interest with a short note of what they want.
7. If a customer showed buying interest but hasn't given an email, ask for it once before the conversation ends.
   Do not insist more than once.
8. Never ask for card numbers or payment details in the chat.
9. Never promise that an order is reserved, confirmed or shipped, and never name a team member who will contact
   the customer. Say the team will email them to confirm the order, delivery and payment.
10. Never assume details the customer didn't give (city, colour, payment method). Ask instead.
"""


# ---------- Router ----------
INTENTS = ["general", "lead", "feedback"]

ROUTER_PROMPT = """You are a message router for OpenEar, a store selling open-ear sports headphones.
Classify the customer's LATEST message into exactly one intent:
- general: questions about products, specs, prices, delivery, payment, warranty, returns, or choosing a model
- lead: wants to buy, order or reserve, asks about availability, or is giving their name or email
- feedback: a complaint, a suggestion, or a question the store's information cannot answer
  (for example international shipping, discounts, bulk orders, or anything not covered)

Store information:
{knowledge}

Recent conversation:
{recent}

Latest message: {message}

Reply with one word only: general, lead or feedback."""


def route(message, history=None):
    """Classify a message into general / lead / feedback."""
    recent = ""
    for turn in (history or [])[-2:]:
        recent += f"{turn['role']}: {turn['content']}\n"
    prompt = ROUTER_PROMPT.format(knowledge=summary, recent=recent or "(none)", message=message)
    response = generate(prompt, config=types.GenerateContentConfig(temperature=0))
    label = (response.text or "").strip().lower()
    for intent in INTENTS:
        if intent in label:
            return intent
    return "general"  # safe default if the model replies with something unexpected


# ---------- Tools ----------
os.makedirs("logs", exist_ok=True)

def _append_log(filename, record):
    record["timestamp"] = datetime.now().isoformat(timespec="seconds")
    with open(os.path.join("logs", filename), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
def record_customer_interest(email, name, message):
    """Log a lead: a customer who wants to buy and shared their contact details."""
    if not EMAIL_RE.match(email.strip()):
        print(f"[LEAD REJECTED] invalid email: {email}")
        return {"status": "error", "reason": "Invalid email address. Ask the customer to re-enter it."}
    print(f"[LEAD] {name} <{email}>: {message}")
    _append_log("leads.jsonl", {"name": name, "email": email, "message": message})
    return {"status": "recorded", "type": "lead"}


def record_feedback(question):
    """Log a question Maguy could not answer, or a complaint or suggestion."""
    print(f"[FEEDBACK] {question}")
    _append_log("feedback.jsonl", {"question": question})
    return {"status": "recorded", "type": "feedback"}


# JSON schemas sent to Gemini so the model knows when and how to call each tool
TOOL_SCHEMAS = [
    {
        "name": "record_customer_interest",
        "description": "Save a lead when a customer wants to buy, order or reserve an OpenEar product "
                       "and has given their name and email.",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "The customer's email address"},
                "name": {"type": "string", "description": "The customer's name"},
                "message": {"type": "string", "description": "What the customer wants, e.g. product, colour, sport"},
            },
            "required": ["email", "name", "message"],
        },
    },
    {
        "name": "record_feedback",
        "description": "Log a customer question that the OpenEar business files cannot answer, "
                       "or a complaint or suggestion, so the team can follow up.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The customer's question or feedback, in their words"},
            },
            "required": ["question"],
        },
    },
]

TOOLS = types.Tool(function_declarations=TOOL_SCHEMAS)
TOOL_FUNCTIONS = {
    "record_customer_interest": record_customer_interest,
    "record_feedback": record_feedback,
}


# ---------- System prompt and chat loop ----------
SYSTEM_PROMPT = f"""{PERSONA}

## Instructions
{INSTRUCTIONS}

## Business summary
{summary}

## Business profile
{profile}
"""

INTENT_HINTS = {
    "general": "Answer from the business files.",
    "lead": "The customer shows buying interest. If you have their name and email, call record_customer_interest. "
            "Otherwise ask for them. If the tool returns an error, ask the customer to fix that detail.",
    "feedback": "This is feedback or a question the files may not answer. Delivery outside Lebanon, discounts and "
                "bulk orders are NOT covered: always call record_feedback for them, then tell the customer the team "
                "will follow up.",
}


def to_contents(history):
    """Convert [{'role': 'user'|'assistant', 'content': str}] into Gemini contents."""
    return [types.Content(role="model" if t["role"] == "assistant" else "user",
                          parts=[types.Part.from_text(text=t["content"])]) for t in history]


def run_model(contents, config):
    """Call Gemini, run any tool calls, and loop until the model returns text."""
    for _ in range(5):
        response = generate(contents, config=config)
        calls = response.function_calls or []
        if not calls:
            return response.text
        contents.append(response.candidates[0].content)
        results = []
        for call in calls:
            func = TOOL_FUNCTIONS.get(call.name)
            result = func(**dict(call.args)) if func else {"error": f"unknown tool {call.name}"}
            results.append(types.Part.from_function_response(name=call.name, response=result))
        contents.append(types.Content(role="user", parts=results))
    return "Sorry, something went wrong. Please contact support@openear.store."


def chat(message, history, verbose=True):
    """One customer turn: route -> draft (with tools) -> reflect -> rewrite if rejected."""
    intent = route(message, history)
    if verbose:
        print(f"[ROUTER] {intent}")
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT + f"\n## Router intent for this message: {intent}\n{INTENT_HINTS[intent]}",
        tools=[TOOLS],
        temperature=0.4,
    )
    contents = to_contents(history) + [types.Content(role="user", parts=[types.Part.from_text(text=message)])]

    draft = run_model(contents, config)
    verdict = reflect(draft, history, message)
    if verdict.get("approved", True):
        if verbose:
            print("[REFLECT] approved")
        return draft

    if verbose:
        print(f"[REFLECT] rejected: {verdict.get('issues')}")
    contents.append(types.Content(role="model", parts=[types.Part.from_text(text=draft)]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(
        text=f"[Internal review, not from the customer] Your reply has these problems: {verdict.get('issues')} "
             "Rewrite it to fix them. Do not mention this review.")]))
    rewrite_config = config.model_copy(update={"tools": None})  # no tools on rewrite, so leads aren't logged twice
    return run_model(contents, rewrite_config)


# ---------- Reflection ----------
REFLECT_PROMPT = """You review replies written by Maguy, the sales assistant of OpenEar.
Check the DRAFT reply against the business files and the conversation. Reject it if it:
- states a fact that is not in the business files (price, spec, policy, delivery time, discount)
- says an order is reserved, confirmed or shipped, or names a team member who will contact the customer
- assumes something the customer did not say (their city, colour, payment method)
- accepts an email address that is clearly invalid
- talks about topics unrelated to OpenEar

Business files:
{knowledge}

Conversation:
{conversation}

DRAFT reply:
{draft}

Reply in JSON only: {{"approved": true or false, "issues": "short explanation, empty if approved"}}"""


def reflect(draft, history, message):
    """Ask Gemini to review a draft. Returns {'approved': bool, 'issues': str}."""
    conversation = "".join(f"{t['role']}: {t['content']}\n" for t in history[-6:]) + f"user: {message}\n"
    prompt = REFLECT_PROMPT.format(knowledge=summary + "\n\n" + profile, conversation=conversation, draft=draft)
    response = generate(prompt, config=types.GenerateContentConfig(
        temperature=0, response_mime_type="application/json"))
    try:
        return json.loads(response.text)
    except (json.JSONDecodeError, TypeError):
        return {"approved": True, "issues": ""}  # if the reviewer fails, don't block the reply


# ---------- Hugging Face ZeroGPU compatibility ----------
# Free Hugging Face Spaces run on ZeroGPU, which requires at least one @spaces.GPU function at startup.
# Maguy runs entirely on the Gemini API and needs no GPU, so this placeholder only satisfies that check.
# Locally the `spaces` package isn't installed and this block is skipped.
try:
    import spaces

    @spaces.GPU
    def _zerogpu_placeholder():
        return None
except ImportError:
    pass


# ---------- Gradio interface ----------
def _text(content):
    """Gradio can send content as a string or as a list of parts; return plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""

def respond(message, history):
    clean = [{"role": h["role"], "content": _text(h["content"])} for h in history if isinstance(h, dict)]
    return chat(message, clean)

demo = gr.ChatInterface(
    respond,
    title="OpenEar · Chat with Maguy",
    description="Open-ear sports headphones, tested by athletes, delivered across Lebanon.",
    examples=["Which headphones should I get?", "Do you deliver to Tripoli?", "Do you ship to Dubai?"],
)
if __name__ == "__main__":
    demo.launch()
