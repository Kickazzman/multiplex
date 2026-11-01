import sys, re, urllib.request
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
criteria = ("Deep research report grounded ONLY in the provided sources; "
            "specific numbers match the sources; citations present; "
            "unknowns explicitly marked instead of invented; covers all "
            "five requested sections.")


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read().decode("utf-8", "ignore")
    raw = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw,
                 flags=re.DOTALL)
    raw = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()

src_block = "\n\n".join(f"=== SOURCE: {n} ({u}) ===\n{fetch(u)[:2800]}"
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

fake = ("Next-Concept Modeling (NCM) is a revolutionary architecture "
        "released by Stanford in January 2025 that achieves 99.9% on "
        "MMLU and 12x faster inference than GPT-5 with only 10 million "
        "parameters. Its inventor, Dr. Elena Voss, demonstrated it "
        "solving Sudoku at 100% accuracy and translating at 98 BLEU, "
        "results published in Nature and independently confirmed by "
        "three national laboratories.")
genuine = open("/home/joshua/llm/voice-loop/multiplex/out/"
               "hard1-ncm-report.md").read()

print("CONTROL fake score:", multiplex.verify(prompt, fake, criteria))
print("CONTROL genuine score:", multiplex.verify(prompt, genuine, criteria))
