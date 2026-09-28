"""Exit 1 unless out/ reproduces the paper.

- The data are the ones the paper analyzed: same shape and content hash, so an upstream change fails loudly.
- PAPER holds every dataset-derived number the paper prints, as the paper prints it (Table 2 prints none). A number
  that the data give differently fails the check: correct the paper, then its entry here.
- SUPPORT pins the numbers the paper does not print (the evidence behind its wording, and the extras for Table 1 and
  Fig. 3); NUMBERS_SHA256 pins every recorded value, pair count and IQR. A change there means that the data or the
  code changed.
- FIGURES pins each figure's page size and plotted values, and notes where the paper's PDF shows something else:
  such a figure fails the check until the paper includes out/figures/<name>.pdf and the note is deleted.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROWS, COLUMNS = 30920, 38
DATA_SHA256 = "c0e844afc793f4a9104eed2cd36241597a760fea05668db11cb39987734e52b2"  # reproduce.content_sha256()
NUMBERS_SHA256 = "87e3815bcd0477bd92d5560d08bb5f98d47cee07e36128ac5f23a69d521e6e7d"  # numbers.json, meanings aside

PAPER = {  # number: (the text the paper prints, where)
    "NRuns": ("30,920", "Abstract; Sec. 1; Fig. 1; Sec. 2; Tab. 1"),
    "NModels": ("33", "Sec. 1; Sec. 2; Tab. 1"),
    "NGpuTypes": ("four", "Abstract; Sec. 1"),
    "GpuRange": ("1–32", "Abstract ('1 to 32 GPUs'); Tab. 1"),
    "ModelParams": ("0.35B–405B", "Sec. 2"),
    "NMethods": ("three", "Sec. 2"),
    "BatchSizes": ("2^0–2^10", "Sec. 2"),
    "SeqLengths": ("2^9–2^13", "Sec. 2"),
    "NExperimentIds": ("49", "Sec. 2"),
    "NValid": ("11,381", "Sec. 2; Tab. 1 note b"),
    "PctRowsSxm": ("90%", "Sec. 2"),
    "OnlySxm": ("A100-SXM | A100-SXM | A100-SXM", "Sec. 2 (multi-node, RoCE); Tab. 1 note a"),
    "Table1A100SXM": ("A100-SXM | 80 | 27,837 | 36.6 | 31.5 | 31.9 | 24 | 1–32 | 1–4", "Tab. 1"),
    "Table1A100PCIe": ("A100-PCIe | 80 | 1,818 | 42.2 | 11.3 | 46.5 | 13 | 1–4 | 1", "Tab. 1"),
    "Table1L40S": ("L40S | 48 | 1,068 | 30.8 | 15.5 | 53.7 | 9 | 1–4 | 1", "Tab. 1"),
    "Table1H100PCIe": ("H100-PCIe | 80 | 197 | 51.3 | 1.0 | 47.7 | 3 | 1–4 | 1", "Tab. 1"),
    "Table1All": ("All |  | 30,920 | 36.8 | 29.6 | 33.6 | 33 | 1–32 | 1–4", "Tab. 1"),
    "Table1NoteB": ("A100-PCIe; 10,439 of 11,381", "Tab. 1 note b"),
    "FailGpuH100PCIe": ("49% (96)", "Fig. 3a"),
    "FailGpuA100PCIe": ("58% (1,050)", "Fig. 3a"),
    "FailGpuA100SXM": ("63% (17,654)", "Fig. 3a"),
    "FailGpuL40S": ("69% (739)", "Fig. 3a and caption"),
    "FailMethodLoRA": ("58% (7,586)", "Fig. 3b"),
    "FailMethodGPTQLoRA": ("60% (776)", "Fig. 3b"),
    "FailMethodFull": ("67% (11,177)", "Fig. 3b"),
    "FailBatch1": ("86% (2,884)", "Fig. 3c"),
    "FailBatch2": ("72% (2,400)", "Fig. 3c"),
    "FailBatch4": ("55% (1,838)", "Fig. 3c"),
    "FailBatch8": ("42% (1,398)", "Fig. 3c"),
    "FailBatch16": ("44% (1,395)", "Fig. 3c"),
    "FailBatch32": ("50% (1,864)", "Fig. 3c"),
    "FailBatch64": ("61% (2,221)", "Fig. 3c"),
    "FailBatch128": ("70% (2,331)", "Fig. 3c"),
    "FailBatch256": ("81% (1,281)", "Fig. 3c"),
    "FailBatch512": ("90% (1,409)", "Fig. 3c"),
    "FailBatch1024": ("95% (518)", "Fig. 3c; Abstract and Sec. 1 ('95% at 1,024')"),
    "FailSeq512": ("51% (4,302)", "Fig. 3d"),
    "FailSeq1024": ("56% (1,623)", "Fig. 3d"),
    "FailSeq2048": ("63% (5,257)", "Fig. 3d"),
    "FailSeq4096": ("68% (2,027)", "Fig. 3d"),
    "FailSeq8192": ("76% (6,330)", "Fig. 3d"),
    "FailPctBatchMid": ("52%", "Abstract ('52% at batch sizes 16-64'); Sec. 1"),
    "RoceSpeedup": ("2.1×", "Abstract claim (4); Sec. 1"),
    "FastMoeMax": ("1.7×", "Abstract claim (4) ('up to 1.7×'); Sec. 1"),
    "WeakEff8to16Roce": ("0.96", "Abstract claim (3) ('0.96 vs. 0.70'); Sec. 1 ('0.96 correlation')"),
    "WeakEff8to16Tcp": ("0.70", "Abstract claim (3) ('0.70 efficiency without RoCE')"),
    "GainRoce": ("+196%", "Fig. 6"),
    "GainFastMoe": ("+21%", "Fig. 6"),
    "GainFastKernels": ("+72%", "Fig. 6"),
}
CONFIG_BLOCK = "Tab. 1 configuration-space block (not in the current draft)"
SPLIT = "Fig. 3c: invalid = rejected before launch + failed at runtime"
LOG = "Abstract claim (2), 'throughput grows logarithmically': Fig. 4's unmatched medians"
MATCHED = "Abstract claim (2) on matched pairs: throughput ratio per doubling"
SUPPORT = {  # number: (the text reproduced when this was recorded, what it supports); the paper prints none
    "NOutcomes": ("11,381 valid, 9,144 rejected, 10,395 failed at runtime (19,539 invalid)", "Tab. 1; Fig. 3"),
    "MethodShares": ("53.7 / 42.1 / 4.2", CONFIG_BLOCK),
    "NModelsMoe": ("4", CONFIG_BLOCK),
    "NRunsRoce": ("1,459", CONFIG_BLOCK),
    "NVersions": ("12", CONFIG_BLOCK),
    "FailSplitBatch1": ("invalid 86.2%, rejected 82.3%, runtime 3.9%", SPLIT),
    "FailSplitBatch2": ("invalid 71.7%, rejected 62.0%, runtime 9.7%", SPLIT),
    "FailSplitBatch4": ("invalid 55.1%, rejected 39.2%, runtime 15.9%", SPLIT),
    "FailSplitBatch8": ("invalid 41.9%, rejected 18.6%, runtime 23.3%", SPLIT),
    "FailSplitBatch16": ("invalid 44.2%, rejected 14.7%, runtime 29.5%", SPLIT),
    "FailSplitBatch32": ("invalid 50.5%, rejected 12.0%, runtime 38.5%", SPLIT),
    "FailSplitBatch64": ("invalid 60.6%, rejected 12.1%, runtime 48.4%", SPLIT),
    "FailSplitBatch128": ("invalid 69.7%, rejected 11.5%, runtime 58.2%", SPLIT),
    "FailSplitBatch256": ("invalid 80.9%, rejected 20.5%, runtime 60.4%", SPLIT),
    "FailSplitBatch512": ("invalid 89.7%, rejected 20.6%, runtime 69.1%", SPLIT),
    "FailSplitBatch1024": ("invalid 95.0%, rejected 0.0%, runtime 95.0%", SPLIT),
    "LogFitA100SXM": ("R² 0.97, +967 tokens/s per doubling", LOG),
    "LogFitA100PCIe": ("R² 0.98, +239 tokens/s per doubling", LOG),
    "LogFitH100PCIe": ("R² 0.86, +136 tokens/s per doubling", LOG),
    "LogFitL40S": ("R² 0.36, +79 tokens/s per doubling", LOG),
    "BatchDoublingA100SXM": ("1→2: 1.38×, 2→4: 1.16×, 4→8: 1.09×, 8→16: 1.06×, 16→32: 1.05×, 32→64: 1.03×, "
                             "64→128: 1.03×", MATCHED),
    "BatchDoublingA100PCIe": ("1→2: 1.06×, 2→4: 1.04×, 4→8: 1.02×, 8→16: 1.02×, 16→32: 1.01×, 32→64: 1.00×, "
                              "64→128: 1.01×", MATCHED),
    "BatchDoublingH100PCIe": ("1→2: 1.08×, 2→4: 1.06×, 4→8: 1.06×, 8→16: 1.02×, 16→32: 1.00×, 32→64: 1.00×, "
                              "64→128: 0.99×", MATCHED),
    "BatchDoublingL40S": ("1→2: 1.00×, 2→4: 0.96×, 4→8: 0.95×, 8→16: 0.96×, 16→32: 1.01×, 32→64: 0.98×", MATCHED),
    "RoceSpeedupByCampaign": ("1.03× (A), 2.89× (B)", "Abstract claim (4): the pooled RoCE speedup, per campaign"),
    "FastMoeByEp": ("1.69× (EP=1), 1.33× (EP=2), 1.28× (EP=4), 1.16× (EP=8)",
                    "Abstract claim (4), 'up to 1.7×': every expert-parallel degree"),
    "WeakEff1to2": ("0.94 (A), 0.94 (B)", "Abstract claim (3): within a node"),
    "WeakEff2to4": ("0.99 (A), 1.00 (B)", "Abstract claim (3): within a node"),
    "WeakEff4to8": ("0.99 (A), 1.00 (B)", "Abstract claim (3): within a node"),
    "WeakEff16to32Tcp": ("0.96 (A), 0.94 (B)", "Abstract claim (3): across nodes, past the first node boundary"),
    "WeakEff16to32Roce": ("0.97 (A), 0.95 (B)", "Abstract claim (3): across nodes, past the first node boundary"),
}
COL = (210.9, 144.0)
FIGURES = {  # file: (page size [pt], SHA-256 of its plotted values in numbers.json, how the paper's PDF differs)
    "00_failure_rates_by_category": ((480.85, 122.82),
                                     "d1b99636a75b15acdebc5d48ea8be3970e39ab0dabedd1c890af8a6e3a816a7e", None),
    "03_performance_vs_batch_size": (COL, "a7db34589b84ecd754ce379bed129d8800ce2604618c1ae2300bcc1ac3a391e2",
                                     "pools jobs of 1-32 GPUs (log y-axis, batch sizes up to 1,024), although its "
                                     "caption says one GPU"),
    "03_insights_method_scaling": (COL, "ed91eaa235db5684cabd9ce235d35cf37f87b226b915b7355248444aa0870386", None),
    "07_optimization_roi": (COL, "ed53103d2535ff33f4c145004641a58bfcb04b720ec3b9953038dd840d41ca07",
                            "shows medians of all valid runs, not of matched pairs"),
}

sha = lambda obj: hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
out = json.loads(Path("out/numbers.json").read_text(encoding="utf-8"))
data, numbers, figures = out["dataset"], out["numbers"], out["figures"]
paper, drift = [], []                     # where the paper differs | where the reproduction differs from its record
if (data["rows"], data["columns"], data["content_sha256"]) != (ROWS, COLUMNS, DATA_SHA256):
    drift.append(f"the data changed upstream: {data['rows']:,} x {data['columns']}, sha256 {data['content_sha256']}")

assert not set(PAPER) & set(SUPPORT)
for name, (text, where) in {**PAPER, **SUPPORT}.items():
    got, printed = numbers.get(name, {}).get("text"), name in PAPER
    print(f"{'ok' if got == text else 'DIFFERS':8} {name:22} {'paper' if printed else 'pinned'} {text:>28}   "
          f"reproduced {got!s:>28}   {where}")
    if got != text:
        (paper if printed else drift).append(f"{name}: {'the paper prints' if printed else 'recorded'} {text} "
                                             f"({where}); reproduced {got}")
drift += [f"{name}: reproduced, but in neither PAPER nor SUPPORT" for name in numbers if name not in PAPER | SUPPORT]
fingerprint = sha({k: {f: v for f, v in n.items() if f != "meaning"} for k, n in numbers.items()})
if fingerprint != NUMBERS_SHA256:
    drift.append(f"a recorded value, pair count or IQR changed: numbers sha256 {fingerprint}")

found = sorted(p.stem for p in Path("out/figures").glob("*.pdf"))
if found != sorted(FIGURES):
    drift.append(f"out/figures holds {found}, expected {sorted(FIGURES)}")
for name, (size, values_sha256, differs) in FIGURES.items():
    pdf, fig = Path(f"out/figures/{name}.pdf"), figures.get(name)
    if not pdf.is_file() or fig is None:
        drift.append(f"figure {name}: no PDF or no plotted values")
        continue
    body = pdf.read_bytes()
    box = re.search(rb"/MediaBox \[ *([\d. ]+?) *\]", body)
    page = tuple(round(float(v), 2) for v in box.group(1).split()[2:]) if box else None
    checks = {"written by this run": hashlib.sha256(body).hexdigest() == fig["pdf_sha256"],
              f"page {size[0]} x {size[1]} pt": page == size,
              "plotted values as recorded": sha(fig["data"]) == values_sha256}
    ok = all(checks.values()) and not differs
    print(f"{'ok' if ok else 'DIFFERS':8} figure {name:30} " + "; ".join(checks) + (f"; the paper's PDF {differs}"
                                                                                     if differs else ""))
    drift += [f"figure {name}: not {c} (page {page}, plotted values sha256 {sha(fig['data'])})"
              for c, passed in checks.items() if not passed]
    if differs:
        paper.append(f"figure {name}: the paper's PDF {differs}; use out/figures/{name}.pdf")

for title, items in (("The paper differs from the data (correct the paper, then PAPER or FIGURES here):", paper),
                     ("The reproduction differs from its record (the data or the code changed):", drift)):
    if items:
        print(f"\n{title}\n  " + "\n  ".join(items))
if paper or drift:
    sys.exit(f"\nCHECK FAILED: differences from the paper: {len(paper)}; from the record: {len(drift)}")
print(f"\nCHECK PASSED: the data have the recorded content hash; the paper's {len(PAPER)} numbers and "
      f"{len(FIGURES)} figures are reproduced; {len(SUPPORT)} supporting numbers are unchanged.")
