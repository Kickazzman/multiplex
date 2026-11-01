import json
import sys
import urllib.request

URL = "http://127.0.0.1:8206/score_steps"


def score(question, steps):
    req = urllib.request.Request(
        URL,
        data=json.dumps({"question": question, "steps": steps}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


# --- Validation 1: the model card's own worked example (Janet's ducks).
# Card's published expected scores: correct -> ~[0.9955, 0.9958, 0.9983, 0.9957]
#                                    wrong  -> ~[0.9955, 0.9958, 0.9983, 0.0240]
Q1 = ("Janet’s ducks lay 16 eggs per day. She eats three for breakfast every "
      "morning and bakes muffins for her friends every day with four. She sells "
      "the remainder at the farmers' market daily for $2 per fresh duck egg. "
      "How much in dollars does she make every day at the farmers' market?")
BASE = [
    "Janet's ducks lay 16 eggs per day.",
    "She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left.",
    "She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left.",
]
GOOD = BASE + ["She sells the remainder at the farmers' market daily for $2 per "
               "fresh duck egg, so she makes 9 * $2 = $18 every day at the "
               "farmers' market. The answer is: 18"]
BAD = BASE + ["She sells the remainder at the farmers' market daily for $2 per "
              "fresh duck egg, so she makes 9 * $2 = $17 every day at the "
              "farmers' market. The answer is: 17"]

g = score(Q1, GOOD)
b = score(Q1, BAD)
print("CARD-EXAMPLE correct:", g)
print("CARD-EXAMPLE wrong  :", b)

# --- Validation 2 (the assigned test): a 4-step solution with step 2 wrong.
Q2 = ("A store sells pencils in packs of 6. Maria buys 4 packs, gives away 5 "
      "pencils, then buys 2 more packs. How many pencils does she have now?")
WRONG2 = [
    "Maria buys 4 packs of 6 pencils, so she starts with 4 * 6 = 24 pencils.",
    "She gives away 5 pencils, so she has 24 - 5 = 21 pencils left.",
    "She buys 2 more packs, which is 2 * 6 = 12 more pencils.",
    "In total she has 21 + 12 = 33 pencils. The answer is: 33",
]
RIGHT2 = [
    "Maria buys 4 packs of 6 pencils, so she starts with 4 * 6 = 24 pencils.",
    "She gives away 5 pencils, so she has 24 - 5 = 19 pencils left.",
    "She buys 2 more packs, which is 2 * 6 = 12 more pencils.",
    "In total she has 19 + 12 = 31 pencils. The answer is: 31",
]
w = score(Q2, WRONG2)
r = score(Q2, RIGHT2)
print("ASSIGNED wrong-step2:", w)
print("ASSIGNED all-correct:", r)

checks = {
    "card correct all > 0.9": all(s > 0.9 for s in g["step_scores"]),
    "card wrong last step < 0.2": b["step_scores"][-1] < 0.2,
    # The wrong step is where the collapse starts: step 1 (pre-error) stays
    # ~1.0, step 2 craters vs its correct twin. Later steps inherit the wrong
    # state, so the global min can sit on the conclusion step - the flag is
    # the collapse AT step 2 relative to the same step done correctly.
    "assigned: wrong step 2 flagged (<0.5, twin >0.9, step1 clean)":
        w["step_scores"][1] < 0.5 and r["step_scores"][1] > 0.9
        and w["step_scores"][0] > 0.9,
    "assigned: wrong step 2 < correct step 2":
        w["step_scores"][1] < r["step_scores"][1],
}
for k, v in checks.items():
    print(("PASS " if v else "FAIL ") + k)
print("SHEPHERD_TEST", "PASS" if all(checks.values()) else "FAIL")
sys.exit(0 if all(checks.values()) else 1)
