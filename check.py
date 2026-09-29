"""Exit 1 unless out/ reproduces the paper.

- The data are the ones the paper analyzed: same shape and content hash, so an upstream change fails loudly.
- PAPER holds every dataset-derived number the paper prints, in its text, tables and captions and inside its figures
  (Table 2 prints none), as a reader sees it (the times sign as ×, a double hyphen as an en dash, a thin space as a
  space). A number that the data give differently fails the check: the paper must print the reproduced text instead;
  correct the paper, then its entry here.
- SUPPORT pins the numbers the paper does not print (the evidence behind its wording); NUMBERS_SHA256 pins every
  recorded value, pair count and IQR. A change there means that the data or the code changed.
- FIGURES pins each figure's page size and plotted values. Every figure must also embed the paper's fonts (Times New
  Roman, regular and bold) and be written by the matplotlib version that drew the paper's figures (3.11.0): another
  version or font moves the text although the page and the plotted values stay the same.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROWS, COLUMNS = 30920, 38
# reproduce.content_sha256() of the data as load_dataset() parses them. pandas' CSV parser reads about 5% of the floats
# one or two units in the last place differently on Linux (the Docker image, on x86-64 and arm64 alike) than on macOS,
# so the hash has one value per platform; every reproduced number and figure is the same on both.
DATA_SHA256 = {"cf8c758cb87335581e6986aee2d369bb6e75f59211c52e1cf010626e9cfeef1c": "Linux, as in the Docker image",
               "c0e844afc793f4a9104eed2cd36241597a760fea05668db11cb39987734e52b2": "macOS"}
NUMBERS_SHA256 = "7b7c17b89740010b2bc2b75b3a08a0949f842c8464d79e3725c8107e462e8dea"  # numbers.json, meanings aside

PAPER = {  # number: (the text the paper prints, where)
    "NRuns": ("30,920", "Abstract; Sec. 1; Sec. 2 'Data processing'; Tab. 1; Fig. 1"),
    "NModels": ("33", "Sec. 1; Sec. 2; Tab. 1"),
    "NGpuTypes": ("four", "Abstract; Sec. 1"),
    "GpuRange": ("1 to 32", "Abstract ('1 to 32 GPUs'); Sec. 3.3 ('(1 to 32)')"),
    "ModelParams": ("0.35B-405B", "Sec. 2"),
    "NMethods": ("3", "Sec. 2 ('3 fine-tuning methods'; 'three' in Sec. 2 and Sec. 3.3)"),
    "BatchSizes": ("2^0-2^10", "Sec. 2"),
    "SeqLengths": ("2^9-2^13", "Sec. 2"),
    "NExperimentIds": ("49", "Sec. 2 'Data processing'"),
    "NRejectedBatchRule": ("5,360", "Sec. 2 'Data processing'"),
    "NRejectedEpRule": ("3,784", "Sec. 2 'Data processing'"),
    "PctRowsSxm": ("90%", "Sec. 2 'Threats to validity'"),
    "OnlySxm": ("A100-SXM | A100-SXM | A100-SXM", "Sec. 2 'Threats to validity'; Tab. 1 note a; Sec. 3.1 (GPTQ-LoRA)"),
    "NOptimizations": ("three", "Sec. 2 'Data Analysis'; Sec. 3.4"),
    "Table1A100SXM": ("A100-SXM | 80 | 27,837 | 36.6 | 31.5 | 31.9 | 24 | 1–32 | 1–4", "Tab. 1"),
    "Table1A100PCIe": ("A100-PCIe | 80 | 1,818 | 42.2 | 11.3 | 46.5 | 13 | 1–4 | 1", "Tab. 1"),
    "Table1L40S": ("L40S | 48 | 1,068 | 30.8 | 15.5 | 53.7 | 9 | 1–4 | 1", "Tab. 1"),
    "Table1H100PCIe": ("H100-PCIe | 80 | 197 | 51.3 | 1.0 | 47.7 | 3 | 1–4 | 1", "Tab. 1"),
    "Table1Dataset": ("Dataset |  | 30,920 | 36.8 | 29.6 | 33.6 | 33 | 1–32 | 1–4", "Tab. 1"),
    "Table1NoteB": ("A100-PCIe", "Tab. 1 note b"),
    "NInvalid": ("19,539", "Sec. 3.1 'Overall'"),
    "PctInvalid": ("63.2%", "Sec. 3.1 'Overall'; Finding 1"),
    "PctInvalidInt": ("63%", "Tab. 3"),
    "InvalidToValid": ("63:37", "Sec. 3.1 'Overall'"),
    "NRejected": ("9,144", "Sec. 3.1 'Overall'"),
    "NRuntime": ("10,395", "Sec. 3.1 'Overall'"),
    "FailBatch1": ("86%", "Finding 1; Sec. 3.1 'Failures vs. Batch sizes'"),
    "FailBatch2": ("72%", "Sec. 3.1 'Failures vs. Batch sizes'"),
    "FailBatch4": ("55%", "Sec. 3.1 'Failures vs. Batch sizes'"),
    "FailBatch8": ("42%", "Finding 1; Sec. 3.1 'Failures vs. Batch sizes'"),
    "FailBatch16": ("44%", "Sec. 3.1 'Failures vs. Batch sizes'"),
    "FailBatch1024": ("95%", "Abstract; Sec. 1; Finding 1; Sec. 3.1 'Failures vs. Batch sizes'"),
    "FailPctBatchMid": ("52%", "Abstract; Sec. 1"),
    "FailGpuL40S": ("69% (739)", "Fig. 2 caption; Sec. 3.1 'Failures vs. GPU Type'"),
    "NRunsL40S": ("1,068", "Sec. 3.1 'Failures vs. GPU Type'"),
    "RuntimePctL40S": ("54%", "Sec. 3.1 'Failures vs. GPU Type'"),
    "FailGpuA100PCIe": ("58% (1,050)", "Sec. 3.1 'Failures vs. GPU Type'"),
    "NRunsA100PCIe": ("1,818", "Sec. 3.1 'Failures vs. GPU Type'"),
    "FailMethodFull": ("67%", "Sec. 3.1 'Failures vs. fine-tuning method'"),
    "RuntimePctLoRA": ("30%", "Sec. 3.1 'Failures vs. fine-tuning method'"),
    "RuntimePctFull": ("36%", "Sec. 3.1 'Failures vs. fine-tuning method'"),
    "RuntimePctGPTQLoRA": ("40%", "Sec. 3.1 'Failures vs. fine-tuning method'"),
    "NValidOneGpu": ("2,320", "Sec. 3.2"),
    "BatchGainPerDoubling": ("about 1,000 tokens/s", "Sec. 3.2 'Throughput vs. batch size'"),
    "TpsSxmBatch1": ("2,847", "Sec. 3.2 'Throughput vs. batch size'"),
    "TpsSxmBatch8": ("6,616", "Sec. 3.2 'Throughput vs. batch size'"),
    "TpsSxmBatch128": ("9,627", "Sec. 3.2 'Throughput vs. batch size'"),
    "MaxBatchOneGpu": ("128", "Sec. 3.2 'Throughput vs. batch size'"),
    "SxmBatch8Over1": ("2.3×", "Finding 3"),
    "SxmBatch128Over8": ("1.5×", "Finding 3; Tab. 3"),
    "PairsL40SA100PCIe": ("103", "Sec. 3.2 'Throughput vs. GPU type'"),
    "L40SLowerThroughput": ("32%", "Sec. 3.2 'Throughput vs. GPU type'"),
    "PairsSxmPcie": ("49", "Sec. 3.2 'Throughput vs. GPU type'"),
    "SxmHigherThroughput": ("18%", "Sec. 3.2 'Throughput vs. GPU type'"),
    "HeatmapMin": ("1.4k", "Sec. 3.2 'Throughput vs. sequence length'"),
    "HeatmapMax": ("75k", "Sec. 3.2 'Throughput vs. sequence length'"),
    "HeatmapMaxGpus": ("16 or 32", "Sec. 3.2 'Throughput vs. sequence length'"),
    "NValid": ("11,381", "Fig. 4 caption"),
    "GpusPerNode": ("8", "Fig. 5 caption ('1 node = 8 GPUs'); Sec. 3.3; Finding 4"),
    "TpsOneGpuRange": ("3,100-4,300", "Sec. 3.3 'Within a node'"),
    "TpsEightGpuRange": ("28,700-33,500", "Sec. 3.3 'Within a node'"),
    "WeakEffIntraNode": ("0.94-1.00", "Sec. 3.3 'Within a node' ('between 0.94 and 1.00'); Finding 4; Tab. 3"),
    "WeakEff8to16Uncapped": ("0.98", "Finding 4"),
    "Gain8to16Uncapped": ("1.96×", "Sec. 3.3 'Across nodes': 2.4.0, with and without RoCE"),
    "WeakEff8to16CappedTcp": ("0.26", "Abstract; Sec. 1; Finding 4"),
    "WeakEff8to16CappedRoce": ("0.72", "Abstract; Sec. 1; Finding 4"),
    "Gain8to16CappedRoce": ("1.45×", "Sec. 3.3 'Across nodes'"),
    "Gain8to16CappedTcp": ("0.52×", "Sec. 3.3 'Across nodes'"),
    "VersionUncapped": ("2.4.0", "Sec. 3.3 'Across nodes'"),
    "VersionCapped": ("2.7.1", "Sec. 3.3 'Across nodes'"),
    "TimeCap": ("600 s", "Sec. 3.3 'Across nodes'; Sec. 3.4; Findings 4 and 5"),
    "MedianParamsLoRAAbove8": ("47B", "Sec. 3.3 'Scaling vs. fine-tuning method'"),
    "MedianParamsFullAbove8": ("8.1B", "Sec. 3.3 'Scaling vs. fine-tuning method'"),
    "LoRAOverFull": ("9%", "Sec. 3.3 'Scaling vs. fine-tuning method'"),
    "PairsLoRAFull": ("3,744", "Sec. 3.3 'Scaling vs. fine-tuning method'"),
    "RoceSpeedup": ("2.16×", "Sec. 3.4 'System-level, RoCE'"),
    "PairsRoce": ("511", "Sec. 3.4 'System-level, RoCE'"),
    "RoceNodes": ("2-4", "Sec. 3.4 'System-level, RoCE'"),
    "RoceUncapped": ("1.03×", "Sec. 3.4 'System-level, RoCE'; Finding 5"),
    "RoceUncappedPct": ("3%", "Sec. 3.4 'System-level, RoCE'"),
    "PairsRoceUncapped": ("241", "Sec. 3.4 'System-level, RoCE'"),
    "RoceUncapped2Nodes": ("1%", "Sec. 3.4 'System-level, RoCE'"),
    "RoceUncapped4NodesLlama70B": ("2.85×", "Sec. 3.4 'System-level, RoCE'"),
    "RoceUncapped4NodesMixtral": ("3.42×", "Sec. 3.4 'System-level, RoCE'"),
    "RoceCapped": ("2.89×", "Sec. 3.4 'System-level, RoCE'; Finding 5; Tab. 3"),
    "RoceCappedQ3": ("3.5×", "Sec. 3.4 'System-level, RoCE'"),
    "RoceUpTo": ("2.9×", "Abstract; Sec. 1; Sec. 5"),
    "FastMoeEp1": ("1.69×", "Sec. 3.4 'Fast MoE'; Finding 6"),
    "FastMoeUpTo": ("1.7×", "Abstract; Sec. 1"),
    "FastMoeEp1Full": ("2.43×", "Sec. 3.4 'Fast MoE'"),
    "FastMoeEp1LoRA": ("1.44×", "Sec. 3.4 'Fast MoE'"),
    "FastMoeEp8LoRA": ("0.92×", "Sec. 3.4 'Fast MoE'; Sec. 3.4 'Implications'"),
    "FastMoeEp2to8": ("1.16-1.33×", "Finding 6"),
    "FastMoeEpDegrees": ("2, 4, and 8", "Finding 6"),
    "FastKernels": ("1.12×", "Sec. 3.4 'Fast Kernels'; Finding 6"),
    "FastKernelsPct": ("12%", "Finding 6"),
    "PairsFastKernels": ("2,418", "Sec. 3.4 'Fast Kernels'"),
    "FastKernelsModels": ("seven", "Sec. 3.4 'Fast Kernels'"),
    "FastKernelsMaxParams": ("8B", "Sec. 3.4 'Fast Kernels'"),
    "FastKernelsIQR": ("5-15%", "Sec. 3.4 'Fast Kernels'"),
    "GpuHoursFastKernels": ("11%", "Sec. 3.4 'Implications'"),
    "GpuHoursRoceCapped": ("65%", "Sec. 3.4 'Implications'"),
    "GpuHoursRange": ("11-65%", "Tab. 3"),
    "Fig2a_Batch1": ("86% (2,884)", "Fig. 2a"),
    "Fig2a_Batch2": ("72% (2,400)", "Fig. 2a"),
    "Fig2a_Batch4": ("55% (1,838)", "Fig. 2a"),
    "Fig2a_Batch8": ("42% (1,398)", "Fig. 2a"),
    "Fig2a_Batch16": ("44% (1,395)", "Fig. 2a"),
    "Fig2a_Batch32": ("50% (1,864)", "Fig. 2a"),
    "Fig2a_Batch64": ("61% (2,221)", "Fig. 2a"),
    "Fig2a_Batch128": ("70% (2,331)", "Fig. 2a"),
    "Fig2a_Batch256": ("81% (1,281)", "Fig. 2a"),
    "Fig2a_Batch512": ("90% (1,409)", "Fig. 2a"),
    "Fig2a_Batch1024": ("95% (518)", "Fig. 2a"),
    "Fig2b_Seq512": ("51% (4,302)", "Fig. 2b"),
    "Fig2b_Seq1024": ("56% (1,623)", "Fig. 2b"),
    "Fig2b_Seq2048": ("63% (5,257)", "Fig. 2b"),
    "Fig2b_Seq4096": ("68% (2,027)", "Fig. 2b"),
    "Fig2b_Seq8192": ("76% (6,330)", "Fig. 2b"),
    "Fig2c_L40S": ("69% (739)", "Fig. 2c"),
    "Fig2c_A100SXM": ("63% (17,654)", "Fig. 2c"),
    "Fig2c_A100PCIe": ("58% (1,050)", "Fig. 2c"),
    "Fig2c_H100PCIe": ("49% (96)", "Fig. 2c"),
    "Fig2d_Full": ("67% (11,177)", "Fig. 2d"),
    "Fig2d_GPTQ": ("60% (776)", "Fig. 2d"),
    "Fig2d_LoRA": ("58% (7,586)", "Fig. 2d"),
    "Fig4_b1_s512": ("1.4k", "Fig. 4"),
    "Fig4_b1_s1024": ("2.4k", "Fig. 4"),
    "Fig4_b1_s2048": ("3.3k", "Fig. 4"),
    "Fig4_b1_s4096": ("2.9k", "Fig. 4"),
    "Fig4_b1_s8192": ("4.6k", "Fig. 4"),
    "Fig4_b2_s512": ("1.9k", "Fig. 4"),
    "Fig4_b2_s1024": ("2.5k", "Fig. 4"),
    "Fig4_b2_s2048": ("4.2k", "Fig. 4"),
    "Fig4_b2_s4096": ("4k", "Fig. 4"),
    "Fig4_b2_s8192": ("6.7k", "Fig. 4"),
    "Fig4_b4_s512": ("2.6k", "Fig. 4"),
    "Fig4_b4_s1024": ("3.3k", "Fig. 4"),
    "Fig4_b4_s2048": ("7.2k", "Fig. 4"),
    "Fig4_b4_s4096": ("5.1k", "Fig. 4"),
    "Fig4_b4_s8192": ("11k", "Fig. 4"),
    "Fig4_b8_s512": ("4.2k", "Fig. 4"),
    "Fig4_b8_s1024": ("5k", "Fig. 4"),
    "Fig4_b8_s2048": ("13k", "Fig. 4"),
    "Fig4_b8_s4096": ("7.5k", "Fig. 4"),
    "Fig4_b8_s8192": ("20k", "Fig. 4"),
    "Fig4_b16_s512": ("7.6k", "Fig. 4"),
    "Fig4_b16_s1024": ("5.3k", "Fig. 4"),
    "Fig4_b16_s2048": ("17k", "Fig. 4"),
    "Fig4_b16_s4096": ("9.1k", "Fig. 4"),
    "Fig4_b16_s8192": ("29k", "Fig. 4"),
    "Fig4_b32_s512": ("12k", "Fig. 4"),
    "Fig4_b32_s1024": ("8k", "Fig. 4"),
    "Fig4_b32_s2048": ("22k", "Fig. 4"),
    "Fig4_b32_s4096": ("18k", "Fig. 4"),
    "Fig4_b32_s8192": ("38k", "Fig. 4"),
    "Fig4_b64_s512": ("17k", "Fig. 4"),
    "Fig4_b64_s1024": ("12k", "Fig. 4"),
    "Fig4_b64_s2048": ("33k", "Fig. 4"),
    "Fig4_b64_s4096": ("29k", "Fig. 4"),
    "Fig4_b64_s8192": ("55k", "Fig. 4"),
    "Fig4_b128_s512": ("22k", "Fig. 4"),
    "Fig4_b128_s1024": ("16k", "Fig. 4"),
    "Fig4_b128_s2048": ("42k", "Fig. 4"),
    "Fig4_b128_s4096": ("44k", "Fig. 4"),
    "Fig4_b128_s8192": ("75k", "Fig. 4"),
    "Fig4_b256_s512": ("39k", "Fig. 4"),
    "Fig4_b256_s2048": ("70k", "Fig. 4"),
    "Fig4_b512_s512": ("56k", "Fig. 4"),
    "Fig4_b1024_s512": ("64k", "Fig. 4"),
    "Fig4NoValidRun": (
        "(256, 4,096), (256, 8,192), (512, 2,048), (512, 4,096), (512, 8,192), (1,024, 2,048), (1,024, 8,192)",
        "Fig. 4 (blank cells) and caption"),
    "Fig4NotTested": ("(256, 1,024), (512, 1,024), (1,024, 1,024), (1,024, 4,096)",
                      "Fig. 4 (hatched cells) and caption"),
    "Fig6a_NoTimeLimit": ("1.03× (241)", "Fig. 6a"),
    "Fig6a_600sLimit": ("2.89× (270)", "Fig. 6a"),
    "Fig6b_Full_EP1": ("2.43× (174)", "Fig. 6b"),
    "Fig6b_Full_EP2": ("1.86× (514)", "Fig. 6b"),
    "Fig6b_Full_EP4": ("1.78× (364)", "Fig. 6b"),
    "Fig6b_Full_EP8": ("1.71× (199)", "Fig. 6b"),
    "Fig6b_LoRA_EP1": ("1.44× (153)", "Fig. 6b"),
    "Fig6b_LoRA_EP2": ("1.08× (446)", "Fig. 6b"),
    "Fig6b_LoRA_EP4": ("1.02× (295)", "Fig. 6b"),
    "Fig6b_LoRA_EP8": ("0.92× (146)", "Fig. 6b"),
    "Fig6c_AllPairs": ("1.12× (2,418)", "Fig. 6c"),
}
SPLIT = "Finding 2; Fig. 2a and 2b: rejected before launch (hatched) and failed at runtime (solid)"
LOG = "Abstract (2), Finding 3, 'grows logarithmically': Fig. 3's unmatched medians"
MATCHED = "Abstract (2), Finding 3, 'grows logarithmically': matched pairs per doubling"
WITHIN = "Sec. 3.3 'Within a node', Finding 4: 'between 0.94 and 1.00 for every intra-node doubling'"
SUPPORT = {  # number: (the text reproduced when this was recorded, what it supports); the paper prints none
    "FailSplitBatch1": ("failing 86.2%, rejected 82.3%, runtime 3.9%", SPLIT),
    "FailSplitBatch2": ("failing 71.7%, rejected 62.0%, runtime 9.7%", SPLIT),
    "FailSplitBatch4": ("failing 55.1%, rejected 39.2%, runtime 15.9%", SPLIT),
    "FailSplitBatch8": ("failing 41.9%, rejected 18.6%, runtime 23.3%", SPLIT),
    "FailSplitBatch16": ("failing 44.2%, rejected 14.7%, runtime 29.5%", SPLIT),
    "FailSplitBatch32": ("failing 50.5%, rejected 12.0%, runtime 38.5%", SPLIT),
    "FailSplitBatch64": ("failing 60.6%, rejected 12.1%, runtime 48.4%", SPLIT),
    "FailSplitBatch128": ("failing 69.7%, rejected 11.5%, runtime 58.2%", SPLIT),
    "FailSplitBatch256": ("failing 80.9%, rejected 20.5%, runtime 60.4%", SPLIT),
    "FailSplitBatch512": ("failing 89.7%, rejected 20.6%, runtime 69.1%", SPLIT),
    "FailSplitBatch1024": ("failing 95.0%, rejected 0.0%, runtime 95.0%", SPLIT),
    "FailSplitSeq512": ("failing 51.2%, rejected 30.3%, runtime 20.9%", SPLIT),
    "FailSplitSeq1024": ("failing 56.1%, rejected 25.7%, runtime 30.5%", SPLIT),
    "FailSplitSeq2048": ("failing 62.9%, rejected 30.5%, runtime 32.5%", SPLIT),
    "FailSplitSeq4096": ("failing 68.3%, rejected 25.6%, runtime 42.7%", SPLIT),
    "FailSplitSeq8192": ("failing 76.2%, rejected 30.7%, runtime 45.5%", SPLIT),
    "FailSeqRange": ("51% → 76%",
                     "Sec. 3.1 'Failures vs. Sequence Lengths': 'failures grow with sequence length' (Fig. 2b)"),
    "RuntimeSeqRange": ("21% → 46%", "Sec. 3.1 'Failures vs. Sequence Lengths': the growth is in runtime failures"),
    "RejectedSeqRange": ("26-31%", "Sec. 3.1 'Failures vs. Sequence Lengths': rejections stay flat"),
    "FailMethodAll": ("Full 67%, LoRA 58%, GPTQ-LoRA 60%",
                      "Sec. 3.1 'Failures vs. fine-tuning method': 'LoRA and GPTQ-LoRA ... fail less often'"),
    "LogFitA100SXM": ("R² 0.97, +967 tokens/s per doubling", LOG),
    "BatchDoublingA100SXM": (
        "1→2: 1.38×, 2→4: 1.16×, 4→8: 1.09×, 8→16: 1.06×, 16→32: 1.05×, 32→64: 1.03×, 64→128: 1.03×", MATCHED),
    "LogFitA100PCIe": ("R² 0.98, +239 tokens/s per doubling", LOG),
    "BatchDoublingA100PCIe": (
        "1→2: 1.06×, 2→4: 1.04×, 4→8: 1.02×, 8→16: 1.02×, 16→32: 1.01×, 32→64: 1.00×, 64→128: 1.01×", MATCHED),
    "LogFitL40S": ("R² 0.36, +79 tokens/s per doubling", LOG),
    "BatchDoublingL40S": ("1→2: 1.00×, 2→4: 0.96×, 4→8: 0.95×, 8→16: 0.96×, 16→32: 1.01×, 32→64: 0.98×", MATCHED),
    "LogFitH100PCIe": ("R² 0.86, +136 tokens/s per doubling", LOG),
    "BatchDoublingH100PCIe": (
        "1→2: 1.08×, 2→4: 1.06×, 4→8: 1.06×, 8→16: 1.02×, 16→32: 1.00×, 32→64: 1.00×, 64→128: 0.99×", MATCHED),
    "GpuTypeRatios": ("L40S/A100-PCIe 0.68× (103 pairs), A100-SXM/A100-PCIe 1.18× (49 pairs)",
                      "Sec. 3.2 'Throughput vs. GPU type': '32% lower', '18% higher'"),
    "L40SSlowerAllPairs": ("103 of 103 pairs below 1× (largest 0.90×)",
                           "Sec. 3.2 'Throughput vs. GPU type': 'In all 103 pairs, ... L40S is slower'"),
    "SxmFasterAllPairs": ("49 of 49 pairs above 1× (smallest 1.07×)",
                          "Sec. 3.2 'Throughput vs. GPU type': 'faster than A100-PCIe in all 49 pairs'"),
    "NoH100Pairs": ("0 with A100-SXM, 0 with A100-PCIe, 0 with L40S",
                    "Sec. 3.2 'Throughput vs. GPU type': 'no identical job on H100 and another device'"),
    "HeatmapCorners": ("smallest (1, 512), largest (128, 8,192)",
                       "Sec. 3.2 'Throughput vs. sequence length': from (1, 512) to (128, 8,192)"),
    "Cell512x8192": ("505 experiments, 0 valid",
                     "Fig. 4 (blank cell); Sec. 3.2: 'increasing further leads to unfeasible, failing jobs'"),
    "EightGpusTwoNodes": ("36 experiments (19 valid) in 1 experiment_id",
                          "Fig. 5 caption '1 node = 8 GPUs'; Sec. 3.3 'up to 8 GPUs, the jobs run intra-node'"),
    "Fig5Above8": ("16 GPUs: Full 26,391, LoRA 14,274; 32 GPUs: Full 32,407, LoRA 13,097",
                   "Sec. 3.3: 'above 8 GPUs, the LoRA curve ... drops below full fine-tuning'"),
    "LoRAOverFullRatio": ("1.09× (3,744 pairs)",
                          "Sec. 3.3 'Scaling vs. fine-tuning method': 'only 9% faster (median for 3,744 pairs)'"),
    "WeakEff1to2": ("0.94 (2.4.0, 75 pairs), 0.94 (2.7.1, 54 pairs)", WITHIN),
    "WeakEff2to4": ("0.99 (2.4.0, 107 pairs), 1.00 (2.7.1, 86 pairs)", WITHIN),
    "WeakEff4to8": ("0.99 (2.4.0, 124 pairs), 1.00 (2.7.1, 125 pairs)", WITHIN),
    "Eff8to16UncappedArms": ("without RoCE 0.98 (1.95×), with RoCE 0.98 (1.96×), 108 pairs each",
                             "Finding 4 '0.98'; Sec. 3.3 '1.96×, with and without RoCE': per network"),
    "Gain16to32Capped": ("1.88× without RoCE, 1.90× with RoCE (102 pairs each)",
                         "Sec. 3.3 'Across nodes': the next doubling (2 to 4 nodes), runs capped at 600 s"),
    "Gain16to32Uncapped": ("1.92× without RoCE, 1.94× with RoCE (68 pairs each)",
                           "Sec. 3.3 'Across nodes': the next doubling (2 to 4 nodes), runs without a time limit"),
    "RoceUncapped4Nodes8B": ("granite-3.1-8b-instruct 1.01× (30 pairs), llama3.1-8b 1.03× (29 pairs)",
                             "Sec. 3.4 'System-level, RoCE': 'On 4 nodes, only Llama-3.1-70B and Mixtral-8x7B gain'"),
    "RoceByNodesUncapped": ("2 nodes 1.01× (112 pairs), 4 nodes 1.40× (129 pairs)",
                            "Sec. 3.4 'System-level, RoCE': runs without a time limit per number of nodes"),
    "RoceByNodesCapped": ("2 nodes 2.62× (152 pairs), 4 nodes 3.00× (118 pairs)",
                          "Sec. 3.4 'System-level, RoCE': runs capped at 600 s per number of nodes"),
    "RoceCappedIQR": ("2.13-3.51× (270 pairs)", "Fig. 6a (whisker); Sec. 3.4: 'a quarter of them above 3.5×'"),
    "PairsFastMoeEp1": ("327", "Sec. 3.4 'Fast MoE', Finding 6: the pairs behind 1.69× (Fig. 6b: 174 + 153)"),
    "FastMoeEpNoValidRun": ("EP=16: 70 experiments, EP=32: 30 experiments",
                            "Finding 6 'spread over 2, 4, and 8 GPUs': EP 16 and 32 have no valid run"),
    "FastMoePooledByEp": (
        "1.69× (EP=1, 327 pairs), 1.33× (EP=2, 960 pairs), 1.28× (EP=4, 659 pairs), 1.16× (EP=8, 345 pairs)",
        "Finding 6 '1.16-1.33×'; Sec. 3.4: 'a decreasing trend' (EP > 1)"),
    "FastMoeFullEp2to8": ("1.71-1.86×", "Fig. 6b and Sec. 3.4 'a decreasing trend': full fine-tuning at EP 2-8"),
    "FastMoeLoRAEp2to8": ("0.92-1.08×", "Fig. 6b and Sec. 3.4 'a decreasing trend': LoRA at EP 2-8"),
    "FastKernelsIQRRatio": ("1.05-1.15×", "Sec. 3.4 'Fast Kernels': 'Half of these pairs gain 5-15%'"),
    "FastKernelsAbove1": ("94.9% of 2,418 pairs above 1×", "Sec. 3.4 'Fast Kernels': 'a smaller, but steady speed-up'"),
}
FONTS = {"TimesNewRomanPSMT", "TimesNewRomanPS-BoldMT"}      # PostScript names of the fonts the figures embed
PRODUCER = "Matplotlib pdf backend v3.11.0"                   # the writer of the paper's figure PDFs
COLUMN = 240.04                                               # [pt] the paper's column width
FIGURES = {  # file: (page size [pt], SHA-256 of its plotted values in numbers.json)
    "00_failure_rates_by_category": ((457.86, 129.01),
                                     "0b3d6f7db79dd52d91d5b0c5ccebdc580de18183810101edaca27477074d0fa4"),
    "03_performance_vs_batch_size": ((COLUMN, 100.8),
                                     "393977e9be09b8470b53f8f7f30643346194a3bec09d3ae6a6b53462160a88de"),
    "08_workload_characteristics": ((COLUMN, 115.2),
                                    "900bac53f4b1907dccdcfb94b53d7c73dcaf8af608d39b28d1185138be2ca2d3"),
    "03_insights_method_scaling": ((COLUMN, 100.8),
                                   "81cbd8a734907808a8c8976478b6c6ec1fee80c0fe88e2c2d5edcc542fd502f4"),
    "07_optimization_roi": ((COLUMN, 183.6),
                            "50f816094b5dccab3e57852ca0a26be8b92ce792d0dcb16880077c99dd2f650e"),
}

sha = lambda obj: hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
out = json.loads(Path("out/numbers.json").read_text(encoding="utf-8"))
data, numbers, figures = out["dataset"], out["numbers"], out["figures"]
paper, drift = [], []                     # where the paper differs | where the reproduction differs from its record
if (data["rows"], data["columns"]) != (ROWS, COLUMNS) or data["content_sha256"] not in DATA_SHA256:
    drift.append(f"the data changed upstream: {data['rows']:,} x {data['columns']}, sha256 {data['content_sha256']}")

assert not set(PAPER) & set(SUPPORT)
for name, (text, where) in {**PAPER, **SUPPORT}.items():
    got, printed = numbers.get(name, {}).get("text"), name in PAPER
    print(f"{'ok' if got == text else 'DIFFERS':8} {name:26} {'paper' if printed else 'pinned'} {text:>28}   "
          f"reproduced {got!s:>28}   {where}")
    if got != text:
        (paper if printed else drift).append(
            f"{name}: the paper prints {text} ({where}); the data give {got}, so the paper must print {got}"
            if printed else f"{name}: recorded {text} ({where}); reproduced {got}")
drift += [f"{name}: reproduced, but in neither PAPER nor SUPPORT" for name in numbers if name not in PAPER | SUPPORT]
fingerprint = sha({k: {f: v for f, v in n.items() if f != "meaning"} for k, n in numbers.items()})
if fingerprint != NUMBERS_SHA256:
    drift.append(f"a recorded value, pair count or IQR changed: numbers sha256 {fingerprint}")

found = sorted(p.stem for p in Path("out/figures").glob("*.pdf"))
if found != sorted(FIGURES):
    drift.append(f"out/figures holds {found}, expected {sorted(FIGURES)}")
for name, (size, values_sha256) in FIGURES.items():
    pdf, fig = Path(f"out/figures/{name}.pdf"), figures.get(name)
    if not pdf.is_file() or fig is None:
        drift.append(f"figure {name}: no PDF or no plotted values")
        continue
    body = pdf.read_bytes()
    box = re.search(rb"/MediaBox \[ *([\d. ]+?) *\]", body)
    page = tuple(round(float(v), 2) for v in box.group(1).split()[2:]) if box else None
    fonts = {f.decode() for f in re.findall(rb"/BaseFont /(?:[A-Z]{6}\+)?([\w-]+)", body)}
    producer = re.search(rb"/Producer \(([^)]*)\)", body)
    checks = {"written by this run": hashlib.sha256(body).hexdigest() == fig["pdf_sha256"],
              f"page {size[0]} x {size[1]} pt": page == size,
              "fonts Times New Roman": fonts == FONTS,
              "written by matplotlib 3.11.0": producer is not None and producer.group(1).decode() == PRODUCER,
              "plotted values as recorded": sha(fig["data"]) == values_sha256}
    print(f"{'ok' if all(checks.values()) else 'DIFFERS':8} figure {name:30} " + "; ".join(checks))
    drift += [f"figure {name}: not {c} (page {page}, fonts {sorted(fonts)}, "
              f"{producer.group(1).decode() if producer else 'no producer'}, plotted values sha256 {sha(fig['data'])})"
              for c, passed in checks.items() if not passed]

for title, items in (("The paper differs from the data (correct the paper, then its entry in PAPER):", paper),
                     ("The reproduction differs from its record (the data or the code changed):", drift)):
    if items:
        print(f"\n{title}\n  " + "\n  ".join(items))
if paper or drift:
    sys.exit(f"\nCHECK FAILED: differences from the paper: {len(paper)}; from the record: {len(drift)}")
print(f"\nCHECK PASSED: the data have the recorded content hash ({DATA_SHA256[data['content_sha256']]}); the "
      f"paper's {len(PAPER)} numbers and {len(FIGURES)} figures are reproduced; {len(SUPPORT)} supporting numbers are "
      "unchanged.")
