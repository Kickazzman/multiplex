import sys, json, time
sys.path.insert(0, "/home/joshua/llm/voice-loop/multiplex")
import multiplex

t0 = time.time()
res = multiplex.run_specialist(
    "agent",
    "Two things: (1) Calculate 17 * 24 + 6 exactly. "
    "(2) Tell me the current time. "
    "You MUST use the calculator tool for the math and the "
    "get_current_time tool for the time - do not guess either answer.",
    max_tokens=400)
print(json.dumps(res, indent=1, default=str))
print("WALL", round(time.time() - t0, 1))

tools = [c["tool"] for c in res.get("tool_calls", [])]
calc_ok = any(c["tool"] == "calculator" and str(c["result"]).strip() == "414"
              for c in res.get("tool_calls", []))
time_ok = "get_current_time" in tools
text_ok = "414" in res.get("text", "")
print("CHECKS", {"tools_called": tools, "calc_414": calc_ok,
                 "time_tool": time_ok, "answer_mentions_414": text_ok})
print("EASY1", "PASS" if (calc_ok and time_ok and text_ok) else "FAIL")
