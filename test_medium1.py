import sys, json, re, subprocess, time, os
sys.path.insert(0, "/home/joshua/llm/voice-loop/multiplex")
import multiplex

PROMPT = """Write a complete Python script and output ONLY the code.

The script must:
1. Use the Open-Meteo API (https://api.open-meteo.com/v1/forecast) with urllib (no third-party packages, no API key).
2. Fetch the CURRENT temperature (current=temperature_2m) for three cities:
   Columbus, OH (latitude 39.9612, longitude -82.9988)
   Cleveland, OH (latitude 41.4993, longitude -81.6944)
   Cincinnati, OH (latitude 39.1031, longitude -84.5120)
3. Save the results as JSON to /tmp/weather.json in the form:
   {"Columbus": {"temp_c": <number>}, "Cleveland": {"temp_c": <number>}, "Cincinnati": {"temp_c": <number>}}
4. Also print the JSON to stdout.

Output only the Python code, in a single ```python block."""

t0 = time.time()
res = multiplex.run_specialist("code", PROMPT, max_tokens=1200)
text = res.get("text", "")
print("code elapsed:", res.get("elapsed"), "chars:", len(text))

m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
code = m.group(1) if m else text
with open("/tmp/weather_scraper.py", "w") as f:
    f.write(code)
print("script bytes:", os.path.getsize("/tmp/weather_scraper.py"))

for p in ("/tmp/weather.json",):
    if os.path.exists(p):
        os.remove(p)
run = subprocess.run([sys.executable, "/tmp/weather_scraper.py"],
                     capture_output=True, text=True, timeout=120)
print("run rc:", run.returncode)
print("stdout:", run.stdout[:300])
print("stderr:", run.stderr[:300])

ok = False
detail = {}
if os.path.exists("/tmp/weather.json"):
    data = json.load(open("/tmp/weather.json"))
    detail = data
    cities = {"Columbus", "Cleveland", "Cincinnati"}
    ok = cities.issubset(data.keys()) and all(
        isinstance(data[c].get("temp_c"), (int, float)) and -30 < data[c]["temp_c"] < 45
        for c in cities)
print("weather.json:", json.dumps(detail))
print("WALL", round(time.time() - t0, 1))
print("MEDIUM1", "PASS" if ok else "FAIL")
