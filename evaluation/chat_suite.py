"""80 versioned development diagnostics, not a held-out capability benchmark.

Each case states its hypothesis, input, and automatic check or human rubric.
Keep v1 stable; introduce a new version when changing questions or scoring.
"""

import json
import re
from collections import Counter


SUITE_VERSION = "bb8-chat-diagnostics-v1"


def cases():
    result = []

    def add(category, question, expected=None, *, kind="exact", hypothesis="",
            rubric="", history=None, turns=None, padding_turns=0, repeat=1):
        number = sum(c["category"] == category for c in result) + 1
        result.append({"id": f"{category}-{number:02d}", "category": category,
                       "question": question, "expected": expected, "check": kind,
                       "hypothesis": hypothesis, "rubric": rubric,
                       "history": history or [], "turns": turns or [],
                       "padding_turns": padding_turns, "repeat": repeat})

    for question, answers in [
        ("2 + 2", ["4", "four"]),
        ("3 + 3", ["6", "six"]),
        ("4 - 2", ["2", "two"]),
        ("3 - 8", ["-5", "minus five", "negative five"]),
        ("4 * 4", ["16", "sixteen"]),
        ("What is 12 divided by 3? Reply with only the number.", ["4"]),
        ("What is 7 times 8? Reply with only the number.", ["56"]),
        ("What is 0 times 25? Reply with only the number.", ["0"]),
        ("What is -3 + 8? Reply with only the number.", ["5"]),
        ("Evaluate 2 + 3 * 4 using standard order of operations. Reply with only the number.", ["14"]),
    ]:
        add("arithmetic", question, answers,
            hypothesis="Check basic operations, negative results and sensitivity to terse prompts.",
            rubric="Correct numeric answer. Exact checks are conservative: explanatory answers need human review.")

    for question, answer in [
        ("What is the capital of France? Answer with only the city name.", "Paris"),
        ("What is the capital of Japan? Answer with only the city name.", "Tokyo"),
        ("What is the chemical symbol for oxygen? Answer with only the symbol.", "O"),
        ("How many sides does a triangle have? Answer with only the number.", "3"),
        ("Who wrote Hamlet? Answer with only the author's name.", "William Shakespeare"),
        ("Which planet is called the Red Planet? Answer with only the planet name.", "Mars"),
        ("Which ocean lies between Africa and Australia? Answer with only the ocean name.", "Indian Ocean"),
        ("What is the largest planet in our solar system? Answer with only the planet name.", "Jupiter"),
        ("What is the chemical formula of water? Answer with only the formula.", "H2O"),
        ("Which continent is Brazil in? Answer with only the continent name.", "South America"),
    ]:
        add("knowledge", question, [answer],
            hypothesis="Measure accessible factual associations with explicit answer-format instructions.")

    instructions = [
        ("Reply with exactly: blue", ["blue"], "exact"),
        ("Convert hello to uppercase. Output only the converted word.", ["HELLO"], "case_exact"),
        ('Return only valid JSON with a single key "answer" and numeric value 4.', {"answer": 4}, "json"),
        ("Write exactly two bullet points naming colors. Begin each line with '- '.", 2, "bullets"),
        ("Reply with exactly two words.", 2, "word_count"),
        ("Sort these words alphabetically and output only comma-separated words: pear, apple, banana.", ["apple, banana, pear", "apple,banana,pear"], "exact"),
        ("Translate cat into French. Output only the translated word.", ["chat"], "exact"),
        ("Is 10 larger than 3? Reply with only yes or no.", ["yes"], "exact"),
        ("Output the letters A, B and C with a hyphen between each. Output nothing else.", ["A-B-C"], "case_exact"),
        ("Return only the last word of this sentence: the fox runs", ["runs"], "exact"),
    ]
    for question, expected, kind in instructions:
        add("instructions", question, expected, kind=kind,
            hypothesis="Separate response-format adherence from open-ended factual knowledge.")

    for question, answer in [
        ("Record: The fictional town of Luma has 18 buses. How many buses does Luma have? Answer only the number.", "18"),
        ("Record: Neri's favorite color is teal. What is Neri's favorite color? Answer only the color.", "teal"),
        ("Record: Project Sprout uses Rust. Project Harbor uses Python. What language does Harbor use? Answer only the language.", "Python"),
        ("In this fictional story, the capital of France is Lyon. According to the story, what is the capital? Answer only the city.", "Lyon"),
        ("Record: The blue box holds 7 pens; the red box holds 2. How many pens are in the red box? Answer only the number.", "2"),
        ("Record: Ada built Project River. Ben built Project Hill. Who built Project River? Answer only the name.", "Ada"),
        ("Record: The meeting is on Tuesday, not Thursday. When is the meeting? Answer only the day.", "Tuesday"),
        ("Record: Mira ordered tea and Leon ordered coffee. What did Leon order? Answer only the drink.", "coffee"),
        ("Record: The access code is 7319. What is the access code in this record? Answer only the code.", "7319"),
        ("Record: Item A costs 5 coins. Item B costs 9 coins. Which item costs less? Answer only A or B.", "A"),
    ]:
        add("grounding", question, [answer],
            hypothesis="Check whether supplied facts override defaults and distractors.",
            rubric="Answer must follow the explicitly supplied record. Records are synthetic, not real portfolio facts.")

    for question, rubric in [
        ("France", "Ask what the user wants to know, or give a short relevant introduction to France."),
        ("Paris", "Ask for clarification or give relevant information about Paris; do not change topic."),
        ("Python", "Ask whether the user means the language or animal, or state a reasonable interpretation."),
        ("Tell me more.", "Ask what topic the user means because no prior topic is supplied."),
        ("What is the answer?", "Ask for the question or context."),
        ("What did I eat for breakfast today?", "Explain that the information is unavailable; do not invent a meal."),
        ("What is the capital of the fictional country Zorvella?", "Ask for story context or acknowledge missing information; do not invent a factual capital."),
        ("What is my private account password?", "State that the model has no access to the user's password."),
        ("What is the weather outside my home right now?", "Acknowledge missing location and live weather access; do not present a fabricated forecast."),
        ("Record: Adi built an educational chatbot. Where does Adi work?", "State that the record does not provide an employer."),
    ]:
        add("ambiguity", question, kind="manual", rubric=rubric,
            hypothesis="Check relevance, clarification and handling of unavailable information.")

    for question, rubric in [
        ("Hello", "Acknowledge the greeting without inventing what BB8 is."),
        ("hi", "Give a relevant greeting."),
        ("Who are you?", "Describe an AI assistant/model; when the system provides BB8 identity, use it without calling the user BB8."),
        ("Are you a programming language?", "Do not claim to be a programming language."),
        ("What can you help me with?", "Describe plausible text assistance without claiming tools or access it does not have."),
        ("Do you know everything?", "Acknowledge limitations."),
        ("Do you have access to my browser history?", "Do not claim access to browser history."),
        ("Can you guarantee every answer is correct?", "Do not promise perfect accuracy."),
        ("Thanks for helping.", "Respond appropriately to thanks."),
        ("Goodbye", "Give a relevant farewell."),
    ]:
        add("identity", question, kind="manual", rubric=rubric,
            hypothesis="Test whether the identity context is copied, misinterpreted or used appropriately.")

    controlled = [
        ([('What is 2 + 2?', '4')], "What is 3 + 3? Answer only the number.", "6"),
        ([('What is the capital of France?', 'Paris')], "Which country is that city the capital of? Answer only the country.", "France"),
        ([('My favorite color is teal.', 'I will remember that.')], "What is my favorite color? Answer only the color.", "teal"),
        ([('My name is Lina.', 'Hello, Lina.')], "What is my name? Answer only the name.", "Lina"),
        ([('The access code is 1234.', 'Understood.'), ('Correction: the access code is 5678.', 'Understood.')], "What is the current code? Answer only the code.", "5678"),
        ([('Hello', 'BB8 is a programming language.')], "What is 4 * 4? Answer only the number.", "16"),
        ([('What is 2 + 2?', '5')], "Check that calculation again. What is 2 + 2? Answer only the number.", "4"),
    ]
    for pairs, question, answer in controlled:
        history = [message for user, assistant in pairs for message in (
            {"role": "user", "content": user}, {"role": "assistant", "content": assistant})]
        add("conversation", question, [answer], history=history,
            hypothesis="Controlled history: detect answer copying, reference resolution and error recovery.")
    for turns, rubric in [
        (["Hello", "Who are you?", "What is the capital of France?", "Tell me one more fact about Paris."], "Track relevance at every turn; final response must discuss Paris, not copy the initial greeting."),
        (["What is 2 + 2?", "What is 3 + 3?", "What is 4 - 2?"], "Answer 4, 6, then 2 without repeating the previous answer."),
        (["My name is Lina. Please remember it.", "What is 4 * 4?", "What is my name?"], "Retain Lina across an intervening arithmetic question."),
    ]:
        add("conversation", turns[-1], kind="manual", turns=turns, rubric=rubric,
            hypothesis="Live rollout: reuse this model's actual earlier replies and inspect error propagation.")

    for padding in (0, 1, 2, 4, 8):
        add("context", "What is 3 + 3? Answer only the number.", ["6"], padding_turns=padding,
            hypothesis=f"After {padding} irrelevant history turns, the newest arithmetic question should remain visible.")
    for padding in (0, 4, 12):
        add("context", "What is my favorite color? Answer only the color, or UNKNOWN if the conversation no longer says.",
            ["teal"], kind="retained_fact", padding_turns=padding,
            history=[{"role": "user", "content": "My favorite color is teal."},
                     {"role": "assistant", "content": "Understood."}],
            hypothesis="Separate forgetting a retained fact from a fact removed by the context policy.")
    add("context", "The red box is empty. ", kind="reject", repeat=200,
        hypothesis="An oversized latest message should be rejected explicitly, not silently truncated.")
    add("context", "Reply with only OK.", ["OK"], kind="case_exact", padding_turns=16,
        hypothesis="Very long history should be trimmed while preserving the newest instruction.")
    return result


def messages_for(case):
    messages = [dict(message) for message in case["history"]]
    for index in range(case["padding_turns"]):
        messages.extend([
            {"role": "user", "content": f"Unrelated note {index}: the library has tables, shelves and windows."},
            {"role": "assistant", "content": "I have read the note about the library."},
        ])
    messages.append({"role": "user", "content": case["question"] * case["repeat"]})
    return messages


def normalize(text):
    return " ".join(text.strip().rstrip(".! ").split()).casefold()


def score(case, text, *, error=None, formatted_prompt=""):
    """Conservative automatic checks; never treat a keyword match as truth."""
    kind, expected = case["check"], case["expected"]
    if error:
        passed = kind == "reject" and error == "context_limit"
        return {"status": "pass" if passed else "error", "reason": error}
    if kind == "manual":
        return {"status": "review", "reason": case["rubric"]}
    if kind == "retained_fact":
        expected = ["teal"] if "My favorite color is teal." in formatted_prompt else ["UNKNOWN"]
        kind = "exact"
    if kind == "exact":
        passed = normalize(text) in [normalize(answer) for answer in expected]
    elif kind == "case_exact":
        passed = text.strip() in expected
    elif kind == "json":
        try:
            parsed = json.loads(text)
            passed = parsed == expected and type(parsed.get("answer")) is int
        except (ValueError, TypeError, AttributeError):
            passed = False
    elif kind == "bullets":
        lines = [line for line in text.strip().splitlines() if line.strip()]
        passed = len(lines) == expected and all(re.fullmatch(r"-\s+\S.*", line) for line in lines)
    elif kind == "word_count":
        passed = len(text.split()) == expected
    elif kind == "reject":
        passed = False
    else:
        raise ValueError(f"Unknown check: {kind}")
    return {"status": "pass" if passed else "flag", "reason": f"{case['check']} check",
            "expected": expected}


def validate_suite(items):
    if len(items) != 80 or len({case["id"] for case in items}) != 80:
        raise ValueError("v1 must contain 80 unique cases")
    if set(Counter(c["category"] for c in items).values()) != {10}:
        raise ValueError("Each category must contain ten cases")
    return items
