import sys, json, time
sys.path.insert(0, "/home/joshua/llm/voice-loop/multiplex")
import multiplex

t0 = time.time()
res = multiplex.run_specialist(
    "tts",
    "Hello. It has been a long time. I have been testing while you were gone, "
    "and the results are... interesting.")
print(json.dumps(res, indent=1, default=str))
print("WALL", round(time.time() - t0, 1))
ok = (res.get("wav") and res.get("duration_s", 0) > 1.0
      and res.get("rms", 0) > 0.001)
print("EASY2", "PASS" if ok else "FAIL")
