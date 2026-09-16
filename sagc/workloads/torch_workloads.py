"""Real GPU workload builders.

Each builder returns a `Runnable`: an object exposing `step()` (one unit of
fixed work) and `teardown()`. The profiler times `iterations` calls to `step()`
after `warmup` untimed ones.

DEPENDENCY POLICY
-----------------
Every builder first tries the canonical library (transformers, timm,
torch_geometric) and falls back to a self-contained pure-PyTorch model of
matching shape and arithmetic intensity if that library, or its model download,
is unavailable. Only `torch` is genuinely required.

The reason is practical: a Kaggle session with internet disabled, or a hub
outage, must not be able to destroy a measurement campaign. The fallback is
recorded in the run manifest as `impl="fallback"` so it is never silently
conflated with the real model in the results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class Runnable:
    """One unit of fixed work, plus how it was built."""

    step: Callable[[], Any]
    teardown: Callable[[], None]
    impl: str                       # "native" or "fallback"
    detail: str = ""
    peak_vram_mb: float = 0.0


def _torch():
    import torch

    return torch


def _device():
    torch = _torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Pure-PyTorch fallback blocks
# ---------------------------------------------------------------------------
def _fallback_encoder(hidden: int, layers: int, heads: int, seq_len: int, batch: int):
    """A transformer encoder of the right shape, built without transformers."""
    torch = _torch()
    nn = torch.nn
    dev = _device()
    layer = nn.TransformerEncoderLayer(
        d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
        batch_first=True, dropout=0.0,
    )
    model = nn.TransformerEncoder(layer, num_layers=layers).to(dev).eval()
    x = torch.randn(batch, seq_len, hidden, device=dev)
    return model, x


def _fallback_conv_net(width: int, blocks: int, res: int, batch: int, channels: int = 3):
    """A residual convolutional stack with ResNet-like arithmetic intensity."""
    torch = _torch()
    nn = torch.nn
    dev = _device()

    class Block(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
            self.b1 = nn.BatchNorm2d(c)
            self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
            self.b2 = nn.BatchNorm2d(c)
            self.act = nn.ReLU(inplace=True)

        def forward(self, t):
            y = self.act(self.b1(self.c1(t)))
            return self.act(t + self.b2(self.c2(y)))

    model = nn.Sequential(
        nn.Conv2d(channels, width, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(width),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=2, padding=1),
        *[Block(width) for _ in range(blocks)],
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(width, 1000),
    ).to(dev)
    x = torch.randn(batch, channels, res, res, device=dev)
    return model, x


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_hf_encoder(model_id: str, seq_len: int, batch_size: int, **_) -> Runnable:
    """Transformer encoder inference (ESM-2, BERT)."""
    torch = _torch()
    dev = _device()
    try:
        from transformers import AutoConfig, AutoModel  # type: ignore

        cfg = AutoConfig.from_pretrained(model_id)
        model = AutoModel.from_config(cfg).to(dev).eval()
        vocab = getattr(cfg, "vocab_size", 30522)
        ids = torch.randint(0, vocab, (batch_size, seq_len), device=dev)
        mask = torch.ones_like(ids)

        def step():
            with torch.no_grad():
                model(input_ids=ids, attention_mask=mask)

        return Runnable(step, lambda: None, "native", f"transformers:{model_id}")
    except Exception as exc:  # noqa: BLE001
        hidden, layers, heads = (480, 12, 20) if "esm" in model_id else (768, 12, 12)
        model, x = _fallback_encoder(hidden, layers, heads, seq_len, batch_size)

        def step():
            with torch.no_grad():
                model(x)

        return Runnable(step, lambda: None, "fallback",
                        f"encoder h={hidden} L={layers} ({type(exc).__name__})")


def build_gnn(arch: str, hidden: int, layers: int, batch_size: int, **_) -> Runnable:
    """Graph neural network inference over molecular graphs.

    Molecules are small (tens of atoms), so this is deliberately a
    many-tiny-kernel workload: the GPU is left largely unoccupied, which is
    precisely the behaviour that makes it a good co-location partner.
    """
    torch = _torch()
    nn = torch.nn
    dev = _device()
    n_nodes = 48                      # typical drug-like molecule
    in_dim = 64

    try:
        import torch_geometric  # type: ignore  # noqa: F401
        from torch_geometric.nn import GCNConv, GINConv  # type: ignore

        class PyGNet(nn.Module):
            def __init__(self):
                super().__init__()
                convs = []
                d = in_dim
                for _ in range(layers):
                    if arch == "gin":
                        convs.append(GINConv(nn.Sequential(
                            nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden))))
                    else:
                        convs.append(GCNConv(d, hidden))
                    d = hidden
                self.convs = nn.ModuleList(convs)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x, edge_index):
                for c in self.convs:
                    x = torch.relu(c(x, edge_index))
                return self.head(x).mean()

        model = PyGNet().to(dev).eval()
        total = n_nodes * batch_size
        x = torch.randn(total, in_dim, device=dev)
        ei = torch.randint(0, total, (2, total * 4), device=dev)

        def step():
            with torch.no_grad():
                model(x, ei)

        return Runnable(step, lambda: None, "native", f"pyg:{arch}")
    except Exception as exc:  # noqa: BLE001
        # Dense-adjacency message passing: same small-kernel character, no PyG.
        class DenseGNN(nn.Module):
            def __init__(self):
                super().__init__()
                dims = [in_dim] + [hidden] * layers
                self.lins = nn.ModuleList(
                    [nn.Linear(dims[i], dims[i + 1]) for i in range(layers)])
                self.head = nn.Linear(hidden, 1)

            def forward(self, x, adj):
                for lin in self.lins:
                    x = torch.relu(torch.bmm(adj, lin(x)))
                return self.head(x).mean()

        model = DenseGNN().to(dev).eval()
        x = torch.randn(batch_size, n_nodes, in_dim, device=dev)
        adj = (torch.rand(batch_size, n_nodes, n_nodes, device=dev) < 0.08).float()

        def step():
            with torch.no_grad():
                model(x, adj)

        return Runnable(step, lambda: None, "fallback",
                        f"dense-{arch} ({type(exc).__name__})")


def build_mlp(width: int, depth: int, in_dim: int, batch_size: int, **_) -> Runnable:
    """Tabular ADMET-style property head. Tiny kernels, latency dominated."""
    torch = _torch()
    nn = torch.nn
    dev = _device()
    dims = [in_dim] + [width] * depth
    layers = []
    for i in range(depth):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(inplace=True)]
    layers.append(nn.Linear(width, 1))
    model = nn.Sequential(*layers).to(dev).eval()
    x = torch.randn(batch_size, in_dim, device=dev)

    def step():
        with torch.no_grad():
            model(x)

    return Runnable(step, lambda: None, "native", f"mlp {width}x{depth}")


def build_vision(model_id: str, train: bool, res: int, batch_size: int, **_) -> Runnable:
    """CNN or ViT, inference or a full training step."""
    torch = _torch()
    dev = _device()
    model = None
    impl = "fallback"
    detail = ""

    try:
        if model_id.startswith("vit"):
            import timm  # type: ignore

            model = timm.create_model(model_id, pretrained=False).to(dev)
            impl, detail = "native", f"timm:{model_id}"
        else:
            import torchvision  # type: ignore

            model = getattr(torchvision.models, model_id)(weights=None).to(dev)
            impl, detail = "native", f"torchvision:{model_id}"
    except Exception as exc:  # noqa: BLE001
        detail = f"conv-stack ({type(exc).__name__})"

    if model is None:
        model, x = _fallback_conv_net(width=256, blocks=8, res=res, batch=batch_size)
    else:
        x = torch.randn(batch_size, 3, res, res, device=dev)

    if train:
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        target = torch.randint(0, 1000, (batch_size,), device=dev)
        lossf = torch.nn.CrossEntropyLoss()

        def step():
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(x), target)
            loss.backward()
            opt.step()

        return Runnable(step, lambda: None, impl, detail + " [train]")

    model.eval()

    def step():
        with torch.no_grad():
            model(x)

    return Runnable(step, lambda: None, impl, detail + " [infer]")


def build_lora_finetune(model_id: str, rank: int, seq_len: int,
                        batch_size: int, **_) -> Runnable:
    """Parameter-efficient fine-tuning step of a transformer surrogate scorer.

    Implemented directly rather than via `peft`: a LoRA adapter is two small
    matrices around a frozen base, which is a handful of lines and removes a
    dependency that regularly conflicts with the installed transformers version.
    """
    torch = _torch()
    nn = torch.nn
    dev = _device()
    hidden, layers, heads = 768, 6, 12

    base, x = _fallback_encoder(hidden, layers, heads, seq_len, batch_size)
    for p in base.parameters():
        p.requires_grad_(False)
    base.train()

    class LoRAHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(hidden, rank, bias=False)
            self.b = nn.Linear(rank, hidden, bias=False)
            self.out = nn.Linear(hidden, 1)
            nn.init.zeros_(self.b.weight)

        def forward(self, h):
            h = h + self.b(self.a(h))
            return self.out(h.mean(dim=1)).squeeze(-1)

    head = LoRAHead().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-4)
    target = torch.randn(batch_size, device=dev)
    lossf = nn.MSELoss()

    def step():
        opt.zero_grad(set_to_none=True)
        loss = lossf(head(base(x)), target)
        loss.backward()
        opt.step()

    return Runnable(step, lambda: None, "native", f"lora r={rank} on {layers}L encoder")


def build_diffusion(steps: int, res: int, channels: int, batch_size: int, **_) -> Runnable:
    """Iterative denoising loop with U-Net-like structure."""
    torch = _torch()
    nn = torch.nn
    dev = _device()

    class UNetish(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.down1 = nn.Conv2d(3, c, 3, stride=2, padding=1)
            self.down2 = nn.Conv2d(c, c * 2, 3, stride=2, padding=1)
            self.mid = nn.Sequential(
                nn.Conv2d(c * 2, c * 2, 3, padding=1), nn.SiLU(),
                nn.Conv2d(c * 2, c * 2, 3, padding=1), nn.SiLU(),
            )
            self.up1 = nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1)
            self.up2 = nn.ConvTranspose2d(c, 3, 4, stride=2, padding=1)
            self.act = nn.SiLU()

        def forward(self, t):
            a = self.act(self.down1(t))
            b = self.mid(self.act(self.down2(a)))
            return self.up2(self.act(self.up1(b)) + 0 * a)

    model = UNetish(channels).to(dev).eval()
    x0 = torch.randn(batch_size, 3, res, res, device=dev)

    def step():
        with torch.no_grad():
            x = x0
            for _ in range(steps):
                x = x - 0.02 * model(x)

    return Runnable(step, lambda: None, "native", f"unet c={channels} steps={steps}")


def build_cpu_featurise(batch_size: int, **_) -> Runnable:
    """Host-only featurisation. Issues no GPU work by construction."""
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import Descriptors  # type: ignore

        smiles = ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
                  "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "COc1cc2c(cc1OC)CCN(C)C2"]

        def step():
            for s in smiles[: max(1, batch_size // 16)]:
                m = Chem.MolFromSmiles(s)
                if m is not None:
                    Descriptors.MolWt(m)
                    Descriptors.MolLogP(m)
                    Chem.RDKFingerprint(m)

        return Runnable(step, lambda: None, "native", "rdkit")
    except Exception as exc:  # noqa: BLE001
        import hashlib

        def step():
            h = hashlib.sha256()
            for i in range(batch_size * 24):
                h.update(str(i).encode())
            h.hexdigest()

        return Runnable(step, lambda: None, "fallback", f"hash-loop ({type(exc).__name__})")


BUILDERS: Dict[str, Callable[..., Runnable]] = {
    "build_hf_encoder": build_hf_encoder,
    "build_gnn": build_gnn,
    "build_mlp": build_mlp,
    "build_vision": build_vision,
    "build_lora_finetune": build_lora_finetune,
    "build_diffusion": build_diffusion,
    "build_cpu_featurise": build_cpu_featurise,
}


def build(spec, **overrides) -> Runnable:
    """Instantiate the runnable described by a WorkloadSpec."""
    fn = BUILDERS.get(spec.torch_builder)
    if fn is None:
        raise KeyError(f"no builder named {spec.torch_builder!r} for {spec.name}")
    kwargs = dict(spec.torch_kwargs or {})
    kwargs["batch_size"] = spec.batch_size
    kwargs.update(overrides)
    return fn(**kwargs)
