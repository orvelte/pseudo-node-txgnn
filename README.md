# TxGNN: Pseudo-Disease Node for Sympathetic Adrenergic TME


Adds a synthetic disease node — "sympathetic_adrenergic_tme" — to the TxGNN knowledge graph, wired to the biological mechanism of sympathetic nerve-driven tumor proliferation (β-adrenergic → NE → tumor TME). Then fine-tunes TxGNN on the augmented graph and queries for drug candidates that could disrupt this axis.

Standard TxGNN queries ask: "what drugs treat cancer X?" The disease pooling
module then borrows signal from related diseases. But sympathetic adrenergic
TME signaling is a *mechanism* that cuts across breast, pancreatic, and prostate cancer — querying any one cancer type conflates this mechanism with all others.

Directly encodes the mechanism as a node, so the model reasons about
"what disrupts this specific biological process" rather than "what treats cancer
in general."

## Structure

```
txgnn_sympathetic_tme/
├── README.md                   ← this file
├── 01_audit_kg.py              ← inspect PrimeKG structure, find relevant nodes
├── 02_build_pseudo_node.py     ← define new node + edges, validate, export
├── 03_inject_and_split.py      ← merge into TxGNN data format, create splits
├── 04_pretrain.py              ← pretrain R-GCN on augmented graph
├── 05_finetune.py              ← finetune on indication/contraindication
├── 06_query.py                 ← run inference, get ranked drug list
├── 07_explain.py               ← pull multi-hop explanatory paths
├── 08_validate_with_lincs.py   ← cross-validate top hits against LINCS L1000
├── edges/
│   └── sympathetic_tme_edges.tsv   ← hand-curated + mined edge list
├── data/                       ← TxGNN data goes here (download separately)
└── results/
    ├── drug_ranking.csv
    ├── explanation_paths.json
    └── lincs_crossval.csv
```

## Setup

```bash
conda create -n txgnn_env python=3.8
conda activate txgnn_env

# PyTorch — match your CUDA version
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# DGL 0.5.2 — TxGNN requires this exact version
pip install dgl-cu117==0.5.2 -f https://data.dgl.ai/wheels/repo.html

pip install TxGNN pandas numpy scipy requests tqdm

# Download PrimeKG + pretrained weights from Harvard Dataverse
# https://doi.org/10.7910/DVN/IXA7BM
# Place all files in ./data/
```

## Key design decisions

1. **Map new edges to existing relation types** — TxGNN's pretrained R-GCN has
   weight matrices for 29 fixed relation types. We don't add new types; we map
   our biology to the closest existing relation. This means we can load pretrained
   weights and only fine-tune, rather than training from scratch.

2. **Pseudo-disease node type = "disease"** — TxGNN's decoder scores
   (drug, disease) pairs. Our node must be type "disease" so the metric-learning
   head can score drugs against it.

3. **Edge density target: ≥40 edges** — Disease pooling similarity is computed
   from neighborhood overlap (all_nodes_profile mode). With <20 edges the
   embedding is too sparse for meaningful pooling. We target 40–80 edges
   covering genes, pathways, biological processes, and phenotypes.

4. **Positive control set** — We include 5 drugs with known β-adrenergic
   cancer biology (propranolol, carvedilol, atenolol, metoprolol, nadolol)
   as held-in training edges. If these don't rank near the top after fine-tuning,
   something is wrong with the edge wiring.
