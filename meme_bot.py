"""Core meme-finding logic: plan search phrases, fetch GIFs, write a caption."""

import ast
import json
import re

import streamlit as st
from ddgs import DDGS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

MODEL = "gpt-5.4-mini"


@st.cache_resource
def get_llm():
    """One LLM client for the whole app (cached across reruns)."""
    return ChatOpenAI(
        temperature=0.05,
        model=MODEL,
        api_key=st.secrets["OPENAI_API_KEY"],
    )


MODES = {
    "Friendly": {
        "emoji": "🤗",
        "blurb": "warm, supportive, on your side",
        "gif_angle": (
            "Pick wholesome, supportive, encouraging reactions - hugs, cheering, "
            "solidarity, 'you got this' energy."
        ),
        "voice": (
            "a kind friend who is on their side. Be warm and encouraging. "
            "Light humour, never at their expense."
        ),
    },
    "Mean": {
        "emoji": "😈",
        "blurb": "savage roast, zero sympathy",
        "gif_angle": (
            "Pick smug, mocking, savage reactions - laughing at them, judgemental "
            "stares, 'skill issue' energy, popcorn-eating, deal-with-it."
        ),
        "voice": (
            "a savage friend who roasts them. Be cutting, sarcastic and merciless "
            "about the SITUATION. Punch at the circumstances, never at protected "
            "traits or appearance."
        ),
    },
    "Sarcastic": {
        "emoji": "🙃",
        "blurb": "dry, deadpan, heavy eye-roll",
        "gif_angle": (
            "Pick deadpan, unimpressed, eye-rolling reactions - slow blink, "
            "side-eye, fake smile, 'sure, jan' energy."
        ),
        "voice": "bone-dry and deadpan. Heavy sarcasm, understated, zero enthusiasm.",
    },
    "Dramatic": {
        "emoji": "🎭",
        "blurb": "everything is a catastrophe",
        "gif_angle": (
            "Pick wildly over-the-top reactions - explosions, fainting, soap-opera "
            "gasps, world-ending panic, theatrical despair."
        ),
        "voice": (
            "absurdly theatrical. Treat this like the end of the world. "
            "Maximum melodrama."
        ),
    },
    "Chaotic": {
        "emoji": "🤪",
        "blurb": "unhinged, random, no thoughts",
        "gif_angle": (
            "Pick surreal, unhinged, cursed-but-funny reactions - weird animals, "
            "random nonsense, chaotic gremlin energy."
        ),
        "voice": (
            "completely unhinged and random. Non-sequiturs welcome. "
            "Keep it playful, never cruel."
        ),
    },
}

DEFAULT_MODE = "Friendly"


def get_mode(name: str) -> dict:
    """Look up a mode, falling back to the default if it's unknown."""
    return MODES.get(name, MODES[DEFAULT_MODE])


planner_prompt = ChatPromptTemplate.from_template(
    """
You are a Planning Agent for a meme reply bot.

Someone said this in a conversation:
"{input}"

You are picking the perfect reaction GIF to reply with.
First, work out the FEELING behind the message (annoyed? tired? shocked? proud?).

The reply must have this TONE: {mode_name}.
{gif_angle}

Now give exactly 3 reaction-GIF search phrases (2-4 words each), one for each angle:
1. the pure EMOTION
2. a FAMOUS MEME reaction
3. the SCENE or action

Every phrase must match the {mode_name} tone described above.
Use words people actually type into a GIF search. Make all 3 different.
Return ONLY a JSON list of 3 strings, nothing else.
""".strip()
)

caption_prompt = ChatPromptTemplate.from_template(
    """Someone said: "{input}"

Reply with one short line (max 10 words), like a friend texting back.
Your persona: {voice}

Stay funny, not hateful: no slurs, no insults about appearance, identity,
race, gender or religion. Output ONLY the reply line.
""".strip()
)


def _coerce(item) -> str:
    """Flatten whatever the model put in the list into a search phrase."""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        # e.g. [{"phrase": "eye roll"}] or {"query": "..."}
        for key in ("phrase", "query", "search", "text", "gif", "term"):
            if key in item and isinstance(item[key], str):
                return item[key].strip()
        vals = [v for v in item.values() if isinstance(v, str)]
        return vals[0].strip() if vals else ""
    return str(item).strip()


def _clean_phrase(line: str) -> str:
    """Strip bullets, numbering, quotes and JSON punctuation off one line."""
    line = line.strip()
    line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)   # bullets / 1. / 2)
    line = line.strip("[](){},")
    line = line.strip().strip("\"'“”‘’").strip()
    line = re.sub(r"\s+", " ", line)
    return line


def _looks_like_phrase(line: str) -> bool:
    """A usable search phrase: short, has letters, isn't chatty filler."""
    if not line or len(line) > 60:
        return False
    if not re.search(r"[a-z]", line, re.I):
        return False
    if len(line.split()) > 6:
        return False
    # Drop conversational scaffolding ("Sure! Here are 3 phrases:")
    if line.rstrip().endswith(":"):
        return False
    if re.match(r"^(sure|here|okay|ok|certainly|of course|hope|let me)", line, re.I):
        return False
    if re.search(r"(helps?|enjoy|good luck)\s*[!.]?$", line, re.I):
        return False
    if line.endswith("!") and len(line.split()) > 2:
        return False
    return True


def _parse_list(answer: str) -> list:
    """Extract search phrases from the model's reply, however it formatted them.

    Tries, in order: strict JSON, a JSON array/object found anywhere in the
    text, Python-literal syntax (single quotes), then line-by-line. Never
    raises - worst case it returns [] and the caller falls back.
    """
    if not answer or not str(answer).strip():
        return []

    text = str(answer).strip()
    # Normalise smart quotes, which break both JSON and literal_eval
    text = (text.replace("“", '"').replace("”", '"')
                .replace("‘", "'").replace("’", "'"))
    # Drop code fences
    text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "").strip()

    def from_obj(obj):
        """Pull a list of strings out of a parsed JSON/Python object."""
        if isinstance(obj, list):
            return [p for p in (_coerce(x) for x in obj) if p]
        if isinstance(obj, dict):
            for key in ("phrases", "queries", "searches", "results", "gifs", "items"):
                if isinstance(obj.get(key), list):
                    return [p for p in (_coerce(x) for x in obj[key]) if p]
            for value in obj.values():           # any list value will do
                if isinstance(value, list):
                    return [p for p in (_coerce(x) for x in value) if p]
            single = _coerce(obj)
            return [single] if single else []
        return []

    # 1. Straight JSON
    try:
        got = from_obj(json.loads(text))
        if got:
            return got
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. A JSON array or object embedded in prose
    for pattern in (r"\[.*?\]", r"\{.*?\}"):
        for match in re.findall(pattern, text, re.DOTALL):
            snippet = re.sub(r",\s*([\]}])", r"", match)   # kill trailing commas
            for loader in (json.loads, ast.literal_eval):
                try:
                    got = from_obj(loader(snippet))
                    if got:
                        return got
                except Exception:
                    continue

    # 3. Python-literal syntax for the whole string (single quotes)
    try:
        got = from_obj(ast.literal_eval(text))
        if got:
            return got
    except Exception:
        pass

    # 4. Line by line
    phrases = []
    for raw in text.splitlines():
        line = _clean_phrase(raw)
        if not line:
            continue
        # A single line may itself be a comma-separated list
        chunks = [_clean_phrase(c) for c in line.split(",")] if "," in line else [line]
        for chunk in chunks:
            if _looks_like_phrase(chunk) and chunk not in phrases:
                phrases.append(chunk)

    return phrases


FALLBACK_FLAVOUR = {
    "Friendly": "supportive hug",
    "Mean": "laughing at you",
    "Sarcastic": "eye roll",
    "Dramatic": "dramatic gasp",
    "Chaotic": "chaotic confused",
}


def _keyword_fallback(user_text: str, mode: str = DEFAULT_MODE) -> list:
    """No usable plan? Build a crude query from the message's own words."""
    stop = {
        "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "my",
        "me", "i", "we", "you", "he", "she", "it", "they", "them", "this",
        "that", "of", "to", "in", "on", "at", "for", "and", "but", "so", "just",
        "now", "then", "there", "has", "have", "had", "did", "do", "does",
    }
    words = re.findall(r"[a-zA-Z']+", user_text.lower())
    keep = [w for w in words if w not in stop and len(w) > 2]
    phrase = " ".join(keep[:3]) if keep else " ".join(words[:3])
    flavour = FALLBACK_FLAVOUR.get(mode, "")
    if not phrase:
        return [flavour or "confused reaction"]
    if not flavour:
        return [phrase]
    # Flavour first: it carries the tone, and the caller may only use the
    # first result. Combine so the GIF still relates to what they said.
    return [f"{flavour} {phrase}", flavour, phrase]


def plan_searches(user_text: str, mode: str = DEFAULT_MODE) -> list:
    """Turn a sentence into up to 3 short GIF search phrases in the given mode.

    Never raises: if the LLM call or its output is unusable, falls back to
    keywords pulled from the user's own message.
    """
    cfg = get_mode(mode)
    try:
        answer = (planner_prompt | get_llm()).invoke({
            "input": user_text,
            "mode_name": mode,
            "gif_angle": cfg["gif_angle"],
        }).content
        phrases = _parse_list(answer)
    except Exception:
        phrases = []
    return phrases[:3] or _keyword_fallback(user_text, mode)


def search_gifs(query: str, limit: int = 4) -> list:
    """Search the web for GIFs matching a short phrase."""
    results = DDGS().images(query + " gif", type_image="gif", max_results=12)
    urls = [r["image"] for r in results if ".gif" in r["image"].lower()]
    return urls[:limit]


class LoopGuard:
    """Stops the agent repeating near-identical searches, or running too long."""

    def __init__(self, max_steps: int = 4):
        self.max_steps = max_steps
        self.history = []

    def is_max_reached(self) -> bool:
        return len(self.history) >= self.max_steps

    def is_repeat(self, step: str) -> bool:
        """Repeat = shares 2+ words with a step we already did."""
        words = set(step.lower().split())
        return any(len(words & set(old.lower().split())) >= 2 for old in self.history)

    def remember(self, step: str):
        self.history.append(step)


def find_memes(message: str, how_many: int = 2, mode: str = DEFAULT_MODE) -> tuple:
    """Return (caption, gif_urls) for a user message, in the chosen tone."""
    cfg = get_mode(mode)
    plan = plan_searches(message, mode)

    guard = LoopGuard(max_steps=4)
    gifs = []
    for step in plan:
        if guard.is_max_reached():
            break
        if guard.is_repeat(step):
            continue
        guard.remember(step)
        try:
            gifs += search_gifs(step, limit=2)
        except Exception:
            continue  # one dead search shouldn't sink the reply

    # Keep order, drop repeats
    gifs = list(dict.fromkeys(gifs))[:how_many]

    try:
        caption = (caption_prompt | get_llm()).invoke({
            "input": message,
            "voice": cfg["voice"],
        }).content.strip()
    except Exception:
        caption = ""
    return caption or "Here's what that felt like:", gifs
