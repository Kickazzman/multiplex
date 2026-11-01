import sys, json, re, time, os, urllib.request
sys.path.insert(0, "/home/joshua/llm/voice-loop/multiplex")
import multiplex

SOURCES = [
    ("Wikipedia: Mamba architecture",
     "https://en.wikipedia.org/wiki/Mamba_(deep_learning_architecture)"),
    ("GitHub: state-spaces/mamba README",
     "https://raw.githubusercontent.com/state-spaces/mamba/main/README.md"),
    ("Hugging Face API: state-spaces models",
     "https://huggingface.co/api/models?author=state-spaces&limit=100"),
]


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read().decode("utf-8", "ignore")
    raw = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw,
                 flags=re.DOTALL)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()


t0 = time.time()
extracts = {}
for name, url in SOURCES:
    try:
        extracts[name] = fetch(url)[:2600]
        print(f"fetched {name}: {len(extracts[name])} chars")
    except Exception as e:
        print(f"FETCH FAIL {name}: {e}")

src_block = "\n\n".join(f"=== SOURCE: {n} ({u}) ===\n{t}"
                        for (n, u), t in
                        zip(SOURCES, (extracts.get(n, "") for n, _ in SOURCES)))
prompt = (
    "Write a factual research brief about the Mamba-3 state-space model "
    "architecture: who created it, when it appeared, what is new in it "
    "versus Mamba-2 and the transformer baseline, and what results are "
    "reported. Use ONLY the sources below; cite the source name after "
    "each claim. If the sources do not say something, do not invent it. "
    "Aim for 400-600 words.\n\n" + src_block)

criteria = ("Every factual claim is supported by one of the provided "
            "sources and cites it; no invented dates, numbers, or names; "
            "covers creators, timing, what's new, and reported results.")
res = multiplex.run_verified("quicklook", prompt, criteria=criteria,
                             max_tokens=1300)
text = res.get("text", "")
out_path = "/home/joshua/llm/voice-loop/multiplex/out/medium2-mamba3-brief.md"
with open(out_path, "w") as f:
    f.write(text)
size = os.path.getsize(out_path)
print("attempts:", json.dumps(res.get("attempts"), default=str)[:600])
print("best score:", res.get("score"), "verified:", res.get("verified"))
print("file:", out_path, "bytes:", size)
print("WALL", round(time.time() - t0, 1))

# spot-check material: numbers present in sources
src_all = " ".join(extracts.values())
nums = set(re.findall(r"\b\d+(?:\.\d+)?\s?(?:B|M|%|billion|million)\b", src_all))
shared = [n for n in nums if n in text]
print("source numbers:", sorted(nums)[:12])
print("numbers shared with report:", shared)
ok = (size > 2000 and res.get("score") is not None
      and len(shared) >= 1 and "Mamba" in text)
print("MEDIUM2", "PASS" if ok else "FAIL")
