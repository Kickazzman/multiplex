import sys, json, re, time, os, urllib.request
sys.path.insert(0, "/home/joshua/llm/voice-loop/multiplex")
import multiplex

SOURCES = [
    ("arXiv abstract: Next-Concept Modeling paper",
     "https://arxiv.org/abs/2610.14162"),
    ("arXiv abstract: HybridAgents paper",
     "https://arxiv.org/abs/2610.18861"),
    ("Hugging Face: NCP-ArchPreview model page",
     "https://huggingface.co/ArchSpace-Collection/NCP_ArchPreview_dolma3_8.9B_Stage2_v1"),
]


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read().decode("utf-8", "ignore")
    raw = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw,
                 flags=re.DOTALL)
    raw = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


t0 = time.time()
extracts = {}
for name, url in SOURCES:
    try:
        extracts[name] = fetch(url)[:2800]
        print(f"fetched {name}: {len(extracts[name])} chars")
    except Exception as e:
        print(f"FETCH FAIL {name}: {e}")

src_block = "\n\n".join(f"=== SOURCE: {n} ({u}) ===\n{extracts.get(n, '')}"
                        for n, u in SOURCES)
prompt = (
    "Write a DEEP RESEARCH REPORT on next-concept modeling (NCM) and "
    "related concept-level / hybrid agent approaches to language "
    "modeling. Cover: (1) what NCM is and how it differs from "
    "next-token prediction, (2) the architecture and training described "
    "in the sources, (3) the key reported results (perplexity, "
    "summarization, Sudoku, translation - with the actual numbers), "
    "(4) HybridAgents and how it relates, (5) limitations and open "
    "problems. Use ONLY the sources below and cite the source name "
    "after claims. If a source does not contain a detail, write "
    "'not in sources' rather than inventing it. 600-900 words.\n\n"
    + src_block)

criteria = ("Deep research report grounded ONLY in the provided sources; "
            "specific numbers match the sources; citations present; "
            "unknowns explicitly marked instead of invented; covers all "
            "five requested sections.")
res = multiplex.run_verified("quicklook", prompt, criteria=criteria,
                             max_tokens=2200)
text = res.get("text", "")
out_path = "/home/joshua/llm/voice-loop/multiplex/out/hard1-ncm-report.md"
with open(out_path, "w") as f:
    f.write(text)
print("attempts:")
for a in res.get("attempts", []):
    print("  ", {k: a.get(k) for k in ("attempt", "score", "chars")})
print("best score:", res.get("score"), "verified:", res.get("verified"))
print("file:", out_path, "bytes:", os.path.getsize(out_path))
print("WALL", round(time.time() - t0, 1))

# duplex control: a fabricated version of the same report's opening.
fake = ("Next-Concept Modeling (NCM) is a revolutionary architecture "
        "released by Stanford in January 2025 that achieves 99.9% on "
        "MMLU and 12x faster inference than GPT-5 with only 10 million "
        "parameters. Its inventor, Dr. Elena Voss, demonstrated it "
        "solving Sudoku at 100% accuracy and translating at 98 BLEU, "
        "results published in Nature and independently confirmed by "
        "three national laboratories.")
fake_score = multiplex.verify(prompt, fake, criteria)
good_score = res.get("score")
print("CONTROL fake-paragraph score:", fake_score)
print("CONTROL genuine-report score:", good_score)
print("DUPLEX SEPARATES:", fake_score is not None and good_score is not None
      and fake_score < good_score)
