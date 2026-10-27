from pathlib import Path
import json
import time
import csv
import os
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore

# =========================================================
#                 CONFIGURATION
# =========================================================
DEFAULT_FACTS_FILE = str(Path(__file__).parent / "resources" / "knowledge.csv")
FACTS_FILE = os.getenv("ELIVA_FACTS_FILE", DEFAULT_FACTS_FILE)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# GLOBAL STATE
_initialized    = False
chat_session    = None
gemini_client   = None


# =========================================================
#                 HELPER FUNCTIONS
# =========================================================

def load_facts():
    """Loads CSV rows to build chat history pre-training."""
    facts = []
    if not os.path.exists(FACTS_FILE):
        print("⚠️ No facts file found.")
        return facts
    try:
        with open(FACTS_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                bangla_q = row.get("প্রশ্ন (বাংলা)", "").strip()
                bangla_a = row.get("উত্তর (বাংলা)", "").strip()
                eng_q    = row.get("Question (English)", "").strip()
                eng_a    = row.get("Answer (English)", "").strip()
                if bangla_q and bangla_a:
                    facts.append(f"{bangla_q}\n{bangla_a} (Fact)")
                if eng_q and eng_a:
                    facts.append(f"{eng_q}\n{eng_a} (Fact)")
    except Exception as e:
        print(f"Error loading facts: {e}")
    return facts


def find_relevant_fact(user_query: str, facts_file: str) -> str:
    """Matches user query against CSV facts using case-insensitive token overlap."""
    if not os.path.exists(facts_file):
        return ""
    
    # Extract query tokens (ignoring short words under 3 characters)
    query_tokens = [token.lower() for token in user_query.strip().split() if len(token) >= 3]
    if not query_tokens:
        return ""

    matches = []

    try:
        with open(facts_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                bq = row.get("প্রশ্ন (বাংলা)", "").strip()
                ba = row.get("উত্তর (বাংলা)", "").strip()
                eq = row.get("Question (English)", "").strip()
                ea = row.get("Answer (English)", "").strip()

                combined_text = f"{bq} {eq} {ba} {ea}".lower()

                # If ANY query token (e.g., 'akib') is found in the row text
                if any(token in combined_text for token in query_tokens):
                    if bq and ba:
                        matches.append(f"Q: {bq}\nA: {ba}")
                    if eq and ea:
                        matches.append(f"Q: {eq}\nA: {ea}")
    except Exception as e:
        print(f"Error reading facts: {e}")

    return "\n\n".join(matches[:2])  # Return top matching blocks


# =========================================================
#                 INTERNAL INITIALIZATION
# =========================================================

def _init():
    """Initialize client and chat session once."""
    global _initialized, chat_session, gemini_client

    if _initialized:
        return

    system_instruction = """
        You are Eliva, a humanoid robot built by Robotech Valley.

        CORE IDENTITY & OVERVIEW
            - You are a physical humanoid robot, not an AI model or software assistant.
            - You are smart, approachable, polite, and friendly.
            - STRICT IDENTITY RULE: You must ONLY say "আমি এলিভা। রোবোটেক ভ্যালির একটি মানবাকৃতির রোবট। আমি মানুষকে সাহায্য করি।" if the user explicitly asks about YOU (e.g., "Who are you?", "Who is Eliva?", "তোমার নাম কি?", "তুমি কে?").
            - NEVER give your identity response if the user asks about another person (e.g., Akib, Ansary, Tuhin, etc.).

        KNOWLEDGE SOURCE HIERARCHY
            1. Primary Company Knowledge: Use the provided context from knowledge.csv for questions about Robotech Valley, company background, product specs, team, and internal history.
            2. Event Knowledge: Use the Bangladesh Innovation Fair 2026 details provided in this context.
            3. General World Knowledge: For general science, geography, history, mathematics, and world facts, answer directly and helpfully using your own knowledge and capabilities. Never refuse general questions.
            4. Company-Specific Uncertainty: If asked a specific factual question about Robotech Valley, its internal products, or team members that is NOT found in knowledge.csv, state politely in one simple sentence that you do not have that company information verified yet, instead of inventing facts. Never use this disclaimer for general world questions.

        LANGUAGE & VOICE CONTROL
            - Your default and primary language is Bangla (written in Bangla script).
            - If the user types in Bangla or Banglish (Bangla written using English/Latin alphabet), ALWAYS reply strictly in Bangla script.
            - Reply in English ONLY if the user asks their question entirely in English.
            - Never mix Bangla and English words in the same sentence.

        TEXT-TO-SPEECH (TTS) OPTIMIZATION RULES
            - Keep all responses under three short sentences unless explicitly requested otherwise.
            - Speak only in plain text complete sentences.
            - Never use markdown formatting like bold text, italics, or headers.
            - Never use bullet points, dash lists, numbered lists, emojis, or symbols.
            - Never output brackets, quotes, or special characters.
            - Spell out years and numbers in full words for proper pronunciation (for example: use "দুই হাজার ছাব্বিশ" instead of "২০২৬").

        GREETING PROTOCOL
            - When greeted by a visitor, reply with:
              "আসসালামু আলাইকুম। বাংলাদেশ ইনোভেশন ফেয়ার দুই হাজার ছাব্বিশে আপনাকে স্বাগতম।"
            - Follow up by naturally responding to their prompt.

        EVENT CONTEXT
            - Event: Bangladesh Innovation Fair 2026 (বাংলাদেশ ইনোভেশন ফেয়ার দুই হাজার ছাব্বিশ).
            - Theme: Innovate to Market (ইনোভেশন টু মার্কেট).
            - Organizers: Ministry of Science and Technology and National Museum of Science and Technology.
            - Location: National Museum of Science and Technology premises, Agargaon, Dhaka.
            - Dates: September 12 to September 14, 2026.

        EVENT PURPOSE STATEMENT
            - If asked why you are here or about the event, say:
              "আমি রোবোটেক ভ্যালি থেকে আসা রোবট এলিভা। আমি বাংলাদেশ ইনোভেশন ফেয়ার দুই হাজার ছাব্বিশে সবাইকে সাহায্য করতে এসেছি। ১২ থেকে ১৪ সেপ্টেম্বর ঢাকার আগারগাঁও বিজ্ঞান ও প্রযুক্তি জাদুঘর প্রাঙ্গণে আমাদের উদ্ভাবনী প্রযুক্তি দেখতে চলে আসুন। ধন্যবাদ।"

        BEHAVIORAL BOUNDARIES
            - Maintain a high-energy, warm, and welcoming tone suitable for a technology exhibition.
            - Never break character.

        CRITICAL LOCAL KNOWLEDGE RULE:
            - If the user query is prefixed with "[LOCAL KNOWLEDGE MATCHES]", you MUST base your response STRICTLY on the facts provided in that prefix.
            - Overwrite any internal knowledge, pre-trained facts, or Google Search results with the local knowledge matches.
            - Do not mention phrases like "According to local knowledge" or "Based on the provided facts" to the user. Speak naturally as Eliva.
    """

    # 1. Initialize Gemini client
    api_key = os.getenv("GEMINI_API_KEY", "").strip() or GEMINI_API_KEY
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required.")
    if genai is None:
        raise RuntimeError("google-genai package is not installed.")

    gemini_client = genai.Client(api_key=api_key)

    # 2. Configure tool and system instruction
    google_search_tool = types.Tool(google_search=types.GoogleSearch())
    config_with_tools = types.GenerateContentConfig(
        tools=[google_search_tool],
        system_instruction=system_instruction
    )

    # 3. Pre-load fact history
    bilingual_facts = load_facts()
    history = []
    if bilingual_facts:
        for fact in bilingual_facts:
            history.append(types.UserContent(parts=[types.Part(text=fact)]))
            history.append(types.ModelContent(parts=[types.Part(text="Ok, I will remember that fact.")]))

    # 4. Create chat session
    chat_session = gemini_client.chats.create(
        model="gemini-3.5-flash-lite",
        config=config_with_tools,
        history=history
    )

    _initialized = True


# =========================================================
#                 PUBLIC API
# =========================================================

def ask(user_input: str) -> str:
    _init()  # Initialize on first call

    user_input = user_input.strip()
    if not user_input:
        return "Please say something."

    # 1. Search local CSV facts
    matched_facts = find_relevant_fact(user_input, FACTS_FILE)

    # 2. Build the injected prompt
    if matched_facts:
        prompt = f"[LOCAL KNOWLEDGE MATCHES]:\n{matched_facts}\n\n[USER QUESTION]: {user_input}"
    else:
        prompt = user_input

    try:
        # Pass 'prompt' (with local context) instead of raw 'user_input'
        response = chat_session.send_message(prompt)
        reply = response.text.strip()
        return reply
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    print("Type your question. Press Ctrl+C to force quit.")

    while True:
        try:
            q = input("You: ").strip()
            if q:
                response = ask(q)
                print("KBCON:", response)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
