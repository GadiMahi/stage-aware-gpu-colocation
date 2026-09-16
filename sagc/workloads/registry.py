"""The workload zoo.

Twelve GPU workloads chosen to span the occupancy / DRAM-bandwidth plane, so a
slowdown model trained across them has genuinely varied behaviour to learn from.
Four are stages of the motivating AI-for-science pipeline; the other eight widen
the training distribution, which is what makes leave-one-workload-out
cross-validation a substantive experiment rather than a gesture. Twelve
workloads give C(12,2) = 66 distinct unordered pairings plus 12 self-pairings.

Every workload is a FIXED-WORK runner: it performs a predetermined number of
iterations rather than running for a fixed wall-clock duration. That is the
single most consequential design decision in the project. Fixed work makes
slowdown a clean ratio and bounds the pairwise sweep to roughly seven GPU-hours
instead of forty-five.

The `sim_*` fields describe each workload's position on the roofline. On a GPU
they are *predictions* the profiler overwrites with measurements; with no GPU
present they drive the analytic interference model. They are calibrated for an
NVIDIA T4 (16 GB, 320 GB/s, 40 SMs), and the kernel statistics are internally
consistent: count x duration x iterations equals the intended duty cycle times
the solo duration.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

# workload families, used for grouping in figures and for held-out CV splits
FAM_TRANSFORMER = "transformer"
FAM_GNN = "gnn"
FAM_CNN = "cnn"
FAM_MLP = "mlp"
FAM_DIFFUSION = "diffusion"
FAM_FINETUNE = "finetune"

# the four stages of the motivating pipeline
STAGE_FEATURISE = "S1_featurise"
STAGE_EMBED = "S2_embed"
STAGE_SCREEN = "S3_screen"
STAGE_REFINE = "S4_refine"


@dataclass(frozen=True)
class WorkloadSpec:
    """Static description of one workload."""

    name: str
    family: str
    description: str

    # --- fixed-work configuration ------------------------------------------
    iterations: int              # timed iterations
    warmup: int                  # untimed warm-up iterations
    batch_size: int

    # --- roofline position (T4-calibrated) ----------------------------------
    sim_occupancy: float         # achieved SM occupancy, fraction of peak
    sim_dram_bw: float           # DRAM bandwidth used, fraction of peak
    sim_vram_mb: int             # peak device memory footprint
    sim_kernel_us: float         # mean kernel duration
    sim_kernel_count: int        # kernels per iteration
    sim_solo_seconds: float      # solo wall time for the full fixed work
    sim_h2d_mb: float            # host to device transfer per run

    # --- interference sensitivities -----------------------------------------
    # How much this workload SUFFERS when the corresponding resource is
    # contended. Bubble-Up's "sensitivity"; the pressure side is given by
    # sim_dram_bw / sim_occupancy above.
    #
    # These are deliberately NOT part of the signature vector the predictor
    # sees. The predictor must infer sensitivity from counter features, which is
    # exactly the inference problem posed on real hardware.
    sens_bandwidth: float
    sens_compute: float
    sens_cache: float

    # --- pipeline membership -------------------------------------------------
    pipeline_stage: Optional[str] = None
    gpu: bool = True             # False for the CPU-only featurisation stage

    # --- how to build it on real hardware ------------------------------------
    torch_builder: str = ""
    torch_kwargs: Optional[Dict] = None

    @property
    def is_pipeline_stage(self) -> bool:
        return self.pipeline_stage is not None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# The zoo
# ---------------------------------------------------------------------------
# Design notes on the spread:
#   bandwidth-bound, low-to-mid occupancy : transformer inference (1, 2, 3, 12)
#   occupancy-starved, tiny kernels       : GNN and MLP inference (4, 5, 6)
#   compute-bound, high occupancy         : CNN / ViT / LoRA / diffusion (7-11)
#   one deliberate VRAM capacity stressor : 12
# Without the capacity stressor, memory exhaustion never appears in the dataset
# as a labelled outcome and the scheduler's VRAM constraint is never exercised.

_SPECS: List[WorkloadSpec] = [
    # ---------------- bandwidth-bound transformers --------------------------
    WorkloadSpec(
        name="esm2_35m_infer",
        family=FAM_TRANSFORMER,
        description="ESM-2 35M protein language model inference over target sequences",
        iterations=4388, warmup=3, batch_size=8,
        sim_occupancy=0.42, sim_dram_bw=0.55, sim_vram_mb=900,
        sim_kernel_us=34.0, sim_kernel_count=148, sim_solo_seconds=24.0,
        sim_h2d_mb=42.0,
        sens_bandwidth=0.72, sens_compute=0.28, sens_cache=0.40,
        pipeline_stage=STAGE_EMBED,
        torch_builder="build_hf_encoder",
        torch_kwargs={"model_id": "facebook/esm2_t12_35M_UR50D", "seq_len": 256},
    ),
    WorkloadSpec(
        name="bert_base_s128_b32",
        family=FAM_TRANSFORMER,
        description="BERT-base inference, sequence length 128, batch 32",
        iterations=3469, warmup=3, batch_size=32,
        sim_occupancy=0.48, sim_dram_bw=0.60, sim_vram_mb=1400,
        sim_kernel_us=41.0, sim_kernel_count=170, sim_solo_seconds=26.0,
        sim_h2d_mb=58.0,
        sens_bandwidth=0.75, sens_compute=0.30, sens_cache=0.45,
        torch_builder="build_hf_encoder",
        torch_kwargs={"model_id": "bert-base-uncased", "seq_len": 128},
    ),
    WorkloadSpec(
        name="bert_base_s512_b8",
        family=FAM_TRANSFORMER,
        description="BERT-base inference, sequence length 512, batch 8 (longer context)",
        iterations=2408, warmup=3, batch_size=8,
        sim_occupancy=0.40, sim_dram_bw=0.68, sim_vram_mb=2100,
        sim_kernel_us=62.0, sim_kernel_count=170, sim_solo_seconds=27.0,
        sim_h2d_mb=36.0,
        sens_bandwidth=0.82, sens_compute=0.26, sens_cache=0.55,
        torch_builder="build_hf_encoder",
        torch_kwargs={"model_id": "bert-base-uncased", "seq_len": 512},
    ),
    # ---------------- occupancy-starved small-kernel work -------------------
    WorkloadSpec(
        name="gcn_molecule_infer",
        family=FAM_GNN,
        description="Graph convolutional network scoring molecular graphs",
        iterations=34118, warmup=10, batch_size=64,
        sim_occupancy=0.12, sim_dram_bw=0.22, sim_vram_mb=400,
        sim_kernel_us=11.0, sim_kernel_count=34, sim_solo_seconds=22.0,
        sim_h2d_mb=18.0,
        sens_bandwidth=0.30, sens_compute=0.22, sens_cache=0.18,
        pipeline_stage=STAGE_SCREEN,
        torch_builder="build_gnn",
        torch_kwargs={"arch": "gcn", "hidden": 128, "layers": 3},
    ),
    WorkloadSpec(
        name="gin_small_infer",
        family=FAM_GNN,
        description="Graph isomorphism network inference, small batch",
        iterations=40444, warmup=10, batch_size=16,
        sim_occupancy=0.09, sim_dram_bw=0.18, sim_vram_mb=350,
        sim_kernel_us=9.0, sim_kernel_count=30, sim_solo_seconds=21.0,
        sim_h2d_mb=9.0,
        sens_bandwidth=0.26, sens_compute=0.20, sens_cache=0.15,
        torch_builder="build_gnn",
        torch_kwargs={"arch": "gin", "hidden": 96, "layers": 4},
    ),
    WorkloadSpec(
        name="mlp_admet_infer",
        family=FAM_MLP,
        description="Small multilayer perceptron over tabular molecular descriptors",
        iterations=98901, warmup=15, batch_size=256,
        sim_occupancy=0.07, sim_dram_bw=0.12, sim_vram_mb=200,
        sim_kernel_us=6.5, sim_kernel_count=14, sim_solo_seconds=20.0,
        sim_h2d_mb=6.0,
        sens_bandwidth=0.20, sens_compute=0.16, sens_cache=0.10,
        torch_builder="build_mlp",
        torch_kwargs={"width": 512, "depth": 6, "in_dim": 200},
    ),
    # ---------------- compute-bound -----------------------------------------
    WorkloadSpec(
        name="resnet50_infer_b64",
        family=FAM_CNN,
        description="ResNet-50 inference, batch 64",
        iterations=3927, warmup=5, batch_size=64,
        sim_occupancy=0.68, sim_dram_bw=0.42, sim_vram_mb=1800,
        sim_kernel_us=48.0, sim_kernel_count=126, sim_solo_seconds=25.0,
        sim_h2d_mb=120.0,
        sens_bandwidth=0.34, sens_compute=0.78, sens_cache=0.30,
        torch_builder="build_vision",
        torch_kwargs={"model_id": "resnet50", "train": False, "res": 224},
    ),
    WorkloadSpec(
        name="resnet50_train_b32",
        family=FAM_CNN,
        description="ResNet-50 training step, batch 32",
        iterations=1577, warmup=5, batch_size=32,
        sim_occupancy=0.78, sim_dram_bw=0.55, sim_vram_mb=4200,
        sim_kernel_us=55.0, sim_kernel_count=310, sim_solo_seconds=28.0,
        sim_h2d_mb=60.0,
        sens_bandwidth=0.46, sens_compute=0.85, sens_cache=0.40,
        torch_builder="build_vision",
        torch_kwargs={"model_id": "resnet50", "train": True, "res": 224},
    ),
    WorkloadSpec(
        name="vit_b16_infer_b32",
        family=FAM_TRANSFORMER,
        description="Vision transformer ViT-B/16 inference, batch 32",
        iterations=3045, warmup=5, batch_size=32,
        sim_occupancy=0.62, sim_dram_bw=0.50, sim_vram_mb=2200,
        sim_kernel_us=52.0, sim_kernel_count=150, sim_solo_seconds=25.0,
        sim_h2d_mb=90.0,
        sens_bandwidth=0.48, sens_compute=0.66, sens_cache=0.38,
        torch_builder="build_vision",
        torch_kwargs={"model_id": "vit_base_patch16_224", "train": False, "res": 224},
    ),
    WorkloadSpec(
        name="lora_finetune_step",
        family=FAM_FINETUNE,
        description="LoRA fine-tuning step of a small transformer surrogate scorer",
        iterations=1827, warmup=4, batch_size=16,
        sim_occupancy=0.72, sim_dram_bw=0.48, sim_vram_mb=3400,
        sim_kernel_us=58.0, sim_kernel_count=260, sim_solo_seconds=29.0,
        sim_h2d_mb=28.0,
        sens_bandwidth=0.44, sens_compute=0.80, sens_cache=0.42,
        pipeline_stage=STAGE_REFINE,
        torch_builder="build_lora_finetune",
        torch_kwargs={"model_id": "distilbert-base-uncased", "rank": 8, "seq_len": 128},
    ),
    WorkloadSpec(
        name="diffusion_unet_sample",
        family=FAM_DIFFUSION,
        description="Diffusion U-Net sampling loop, 20 denoising steps",
        iterations=1651, warmup=3, batch_size=4,
        sim_occupancy=0.75, sim_dram_bw=0.52, sim_vram_mb=3200,
        sim_kernel_us=61.0, sim_kernel_count=280, sim_solo_seconds=30.0,
        sim_h2d_mb=14.0,
        sens_bandwidth=0.47, sens_compute=0.82, sens_cache=0.44,
        torch_builder="build_diffusion",
        torch_kwargs={"steps": 20, "res": 64, "channels": 128},
    ),
    # ---------------- VRAM capacity stressor ---------------------------------
    WorkloadSpec(
        name="vit_large_batch_infer",
        family=FAM_TRANSFORMER,
        description="ViT inference at batch 128, chosen to stress VRAM capacity",
        iterations=2157, warmup=3, batch_size=128,
        sim_occupancy=0.58, sim_dram_bw=0.62, sim_vram_mb=12200,
        sim_kernel_us=92.0, sim_kernel_count=150, sim_solo_seconds=31.0,
        sim_h2d_mb=360.0,
        sens_bandwidth=0.70, sens_compute=0.58, sens_cache=0.60,
        torch_builder="build_vision",
        torch_kwargs={"model_id": "vit_base_patch16_224", "train": False, "res": 224},
    ),
]

# ---------------------------------------------------------------------------
# The CPU-only pipeline stage.
#
# Featurisation issues no GPU work at all. It is deliberately NOT a member of
# the co-location zoo (there is nothing to co-locate) but IS a member of the
# pipeline, because it is the single largest source of held-but-idle GPU time
# under whole-device allocation. Excluding it from the zoo while including it in
# the pipeline is the whole point of the project.
# ---------------------------------------------------------------------------
FEATURISE = WorkloadSpec(
    name="rdkit_featurise",
    family="cpu",
    description="RDKit featurisation of ligand SMILES to molecular graphs (host only)",
    iterations=2000, warmup=0, batch_size=64,
    sim_occupancy=0.0, sim_dram_bw=0.0, sim_vram_mb=0,
    sim_kernel_us=0.0, sim_kernel_count=0, sim_solo_seconds=18.0,
    sim_h2d_mb=0.0,
    sens_bandwidth=0.0, sens_compute=0.0, sens_cache=0.0,
    pipeline_stage=STAGE_FEATURISE,
    gpu=False,
    torch_builder="build_cpu_featurise",
    torch_kwargs={},
)

ZOO: Dict[str, WorkloadSpec] = {w.name: w for w in _SPECS}
ALL: Dict[str, WorkloadSpec] = {**ZOO, FEATURISE.name: FEATURISE}

#: Ordered stages of the motivating pipeline.
PIPELINE: List[WorkloadSpec] = [
    FEATURISE,
    ZOO["esm2_35m_infer"],
    ZOO["gcn_molecule_infer"],
    ZOO["lora_finetune_step"],
]


def zoo_names() -> List[str]:
    return list(ZOO.keys())


def get(name: str) -> WorkloadSpec:
    if name not in ALL:
        raise KeyError(f"unknown workload {name!r}; known: {sorted(ALL)}")
    return ALL[name]


def pipeline_stage_names() -> List[str]:
    return [w.name for w in PIPELINE]


def unordered_pairs(names: Optional[List[str]] = None, include_self: bool = True):
    """All co-location combinations to measure."""
    names = names or zoo_names()
    out = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j < i:
                continue
            if j == i and not include_self:
                continue
            out.append((a, b))
    return out


def families() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for w in ZOO.values():
        out.setdefault(w.family, []).append(w.name)
    return out


# ---------------------------------------------------------------------------
# Derived device metrics
# ---------------------------------------------------------------------------
def duty_cycle(spec: WorkloadSpec) -> float:
    """Fraction of wall time during which a kernel is executing."""
    if not spec.gpu or spec.sim_solo_seconds <= 0:
        return 0.0
    total = spec.sim_kernel_count * spec.iterations * spec.sim_kernel_us / 1e6
    return float(min(1.0, total / spec.sim_solo_seconds))


def reported_utilisation(spec: WorkloadSpec) -> float:
    """What `nvidia-smi utilization.gpu` would report for this workload.

    This is deliberately NOT a function of occupancy. NVML defines utilisation
    as the fraction of the sampling window during which at least one kernel was
    resident, so any workload submitting work back to back reads high regardless
    of how much of the device that work actually fills. The compression below
    (roughly 0.95 to 0.99 across the entire zoo, against an occupancy range of
    0.07 to 0.78) is the whole phenomenon the project argues about.

    Modelling utilisation as proportional to occupancy, as a draft of this file
    did, quietly makes it an excellent proxy and INVERTS the result of ablation
    A1. It is simply wrong about how NVML works. Guarded by
    tests/test_sagc.py::test_utilisation_does_not_track_occupancy.
    """
    if not spec.gpu:
        return 0.0
    return float(min(0.99, 0.80 + 0.19 * (duty_cycle(spec) ** 0.25)))


def sm_active_level(spec: WorkloadSpec) -> float:
    """Modelled DCGM SM_ACTIVE (field 1002).

    SM_ACTIVE is the fraction of time at least one warp is resident, averaged
    over SMs. It therefore sits between reported utilisation (which ignores how
    much of the device is used) and achieved occupancy (which measures how full
    each SM is). A workload whose kernels reach only a few SMs reads low here
    even while utilisation reads 99 per cent, which is the intermediate case the
    profiler needs to be able to distinguish.
    """
    if not spec.gpu:
        return 0.0
    sm_coverage = min(1.0, spec.sim_occupancy * 2.5)
    return float(min(0.99, duty_cycle(spec) * sm_coverage))
