"""
03_inject_and_split.py — disk-efficient injection (no full KG copy)
Appends pseudo-node edges directly to kg.csv and saves a small delta file.
Idempotent: skips append if pseudo-node already present in kg.csv.
Run after 01_audit_kg.py and 02_build_pseudo_node.py.
"""
import os, json
import pandas as pd
import numpy as np

DATA_DIR  = "./data"
EDGES_DIR = "./edges"
OUT_DIR   = "./results"
os.makedirs(OUT_DIR, exist_ok=True)

PSEUDO_ID   = "PSEUDO:sympathetic_adrenergic_tme"
PSEUDO_NAME = "sympathetic adrenergic tumor microenvironment signaling"

new_edges_path = os.path.join(EDGES_DIR, "sympathetic_tme_edges.tsv")
assert os.path.exists(new_edges_path), "Run 02_build_pseudo_node.py first."
new_edges = pd.read_csv(new_edges_path, sep="\t")
print(f"Pseudo-node edges to inject: {len(new_edges)}")

kg_path = os.path.join(DATA_DIR, "kg.csv")
if not os.path.exists(kg_path):
    kg_path = os.path.join(DATA_DIR, "kg_raw.csv")

kg_cols = list(pd.read_csv(kg_path, nrows=1).columns)
print(f"KG columns: {kg_cols}")

print("Checking if pseudo-node already present in kg.csv...")
kg_ids = pd.read_csv(kg_path, low_memory=False, usecols=["x_id","y_id"], dtype=str)
already_present = (kg_ids["x_id"] == PSEUDO_ID).any() or (kg_ids["y_id"] == PSEUDO_ID).any()

if already_present:
    print("  Pseudo-node already in kg.csv — skipping append (idempotent re-run)")
else:
    print("Building node metadata lookup...")
    kg_meta = pd.read_csv(kg_path, low_memory=False,
                          usecols=["x_id","x_index","x_source","y_id","y_index","y_source"],
                          dtype={"x_id":str,"y_id":str})
    node_lookup = {}
    for _, r in pd.concat([
        kg_meta[["x_id","x_index","x_source"]].rename(columns={"x_id":"id","x_index":"idx","x_source":"src"}),
        kg_meta[["y_id","y_index","y_source"]].rename(columns={"y_id":"id","y_index":"idx","y_source":"src"}),
    ]).drop_duplicates(subset=["id"]).iterrows():
        node_lookup[str(r["id"])] = (int(r["idx"]), str(r["src"]))

    pseudo_idx = max(v[0] for v in node_lookup.values()) + 1
    display_rel_map = {
        "disease_protein":            "associated with",
        "disease_phenotype_positive": "phenotype present",
        "indication":                 "indication",
        "contraindication":           "contraindication",
    }

    def node_meta(nid):
        if str(nid) == PSEUDO_ID:
            return pseudo_idx, "curated_synthetic"
        m = node_lookup.get(str(nid))
        return (m[0], m[1]) if m else (-1, "unknown")

    ec = new_edges.drop(columns=["evidence"], errors="ignore").copy()
    ec["display_relation"] = ec["relation"].map(display_rel_map)
    xm = ec["x_id"].apply(node_meta).tolist()
    ym = ec["y_id"].apply(node_meta).tolist()
    ec["x_index"]  = [m[0] for m in xm]
    ec["x_source"] = [m[1] for m in xm]
    ec["y_index"]  = [m[0] for m in ym]
    ec["y_source"] = [m[1] for m in ym]

    delta_path = os.path.join(DATA_DIR, "kg_pseudo_edges.csv")
    ec[kg_cols].to_csv(delta_path, index=False)
    print(f"  Delta saved to {delta_path} ({len(ec)} rows)")

    print(f"  Appending {len(ec)} rows to {kg_path}...")
    ec[kg_cols].to_csv(kg_path, mode="a", header=False, index=False)
    print(f"  Appended to {kg_path}")

print("\nBuilding disease node index...")
kg_d = pd.read_csv(kg_path, low_memory=False,
                   usecols=["x_id","x_type","y_id","y_type"], dtype=str)
disease_nodes = pd.concat([
    kg_d[kg_d["x_type"]=="disease"]["x_id"],
    kg_d[kg_d["y_type"]=="disease"]["y_id"],
]).drop_duplicates().tolist()
print(f"Disease node count: {len(disease_nodes)}")
print(f"Pseudo-node present: {PSEUDO_ID in disease_nodes}")

split_path     = os.path.join(DATA_DIR, "train_val_test_disease_idx.json")
aug_split_path = os.path.join(DATA_DIR, "train_val_test_disease_idx_augmented.json")

if os.path.exists(split_path):
    with open(split_path) as f:
        splits = json.load(f)
    print("\nExisting splits: " + ", ".join(f"{k}={len(v)}" for k,v in splits.items()))
    if PSEUDO_ID not in splits.get("train", []):
        splits["train"].append(PSEUDO_ID)
        print("  Appended pseudo-node to train split")
else:
    other = [d for d in disease_nodes if d != PSEUDO_ID]
    np.random.seed(42); np.random.shuffle(other); n = len(other)
    splits = {
        "train": other[:int(0.70*n)] + [PSEUDO_ID],
        "val":   other[int(0.70*n):int(0.85*n)],
        "test":  other[int(0.85*n):],
    }
    print(f"Created new split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

with open(aug_split_path, "w") as f:
    json.dump(splits, f)
print(f"Saved augmented split to {aug_split_path}")

label_path = os.path.join(DATA_DIR, "drug_disease_train_labels.json")
ind_edges  = new_edges[new_edges["relation"] == "indication"]
print(f"\nFine-tuning indication labels: {len(ind_edges)}")
for _, row in ind_edges.iterrows():
    print(f"  + {row['x_name']} -> {PSEUDO_NAME}")

labels = [{"drug_id":r["x_id"],"disease_id":PSEUDO_ID,"label":1,"split":"train"}
          for _,r in ind_edges.iterrows()]
if os.path.exists(label_path):
    with open(label_path) as f:
        existing = json.load(f)
    seen = {(l["drug_id"],l["disease_id"]) for l in existing}
    labels = existing + [l for l in labels if (l["drug_id"],l["disease_id"]) not in seen]

with open(label_path, "w") as f:
    json.dump(labels, f, indent=2)
print(f"Saved {len(labels)} fine-tuning labels to {label_path}")

print("\n" + "="*60)
print(f"DONE  Pseudo-node injected into kg.csv (no full copy written)")
print(f"  Delta:  {os.path.join(DATA_DIR, 'kg_pseudo_edges.csv')}")
print(f"  Split:  {aug_split_path}")
print(f"  Labels: {label_path}")
print("\nNEXT STEP: Run 04_05_pretrain_finetune.py")
