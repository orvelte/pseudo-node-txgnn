"""
01_audit_kg.py
==============
Inspect the PrimeKG knowledge graph to:
  1. Print all node types and relation types
  2. Find exact node IDs for the genes, pathways, and phenotypes we need
  3. Confirm which relation types we'll re-use for our new edges

Run this FIRST before building the pseudo-node.
All outputs are printed + saved to results/kg_audit.json.

EXPECTED OUTPUT (approximate, from PrimeKG):
  Node types: ['gene/protein', 'disease', 'drug', 'effect/phenotype',
               'biological_process', 'molecular_function', 'cellular_component',
               'exposure', 'pathway', 'anatomy']
  Relation types: 29 types including 'carrier', 'enzyme', 'target', 'transporter',
                  'contraindication', 'indication', 'off-label use',
                  'synergistic interaction', 'associated with', 'parent-child',
                  'phenotype present', 'phenotype absent', 'side effect',
                  'interacts with', 'linked to', ...
"""

import os
import json
import pandas as pd

DATA_DIR = './data'
OUT_DIR  = './results'
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 1. Load KG files ─────────────────────────────────────────────────────────
# PrimeKG ships as: nodes.tsv, edges.tsv (or kg.csv / kg_raw.csv in some builds)
# TxGNN's TxData loader expects specific filenames — audit what's in ./data/

print("Files in ./data/:")
for f in sorted(os.listdir(DATA_DIR)):
    size_mb = os.path.getsize(os.path.join(DATA_DIR, f)) / 1e6
    print(f"  {f:40s}  {size_mb:.1f} MB")

# Load the main edge file — TxGNN uses 'kg.csv' or 'kg_raw.csv'
kg_path = os.path.join(DATA_DIR, 'kg.csv')
if not os.path.exists(kg_path):
    kg_path = os.path.join(DATA_DIR, 'kg_raw.csv')

print(f"\nLoading KG from {kg_path}...")
kg = pd.read_csv(kg_path, low_memory=False)
print(f"KG shape: {kg.shape}")
print(f"\nColumns: {list(kg.columns)}")
print(f"\nFirst 3 rows:\n{kg.head(3)}")

# ─── 2. Node types and relation types ─────────────────────────────────────────
print("\n" + "="*60)
print("NODE TYPES:")
if 'x_type' in kg.columns:
    node_types = pd.concat([
        kg[['x_id', 'x_type']].rename(columns={'x_id':'id', 'x_type':'type'}),
        kg[['y_id', 'y_type']].rename(columns={'y_id':'id', 'y_type':'type'})
    ]).drop_duplicates()
    type_counts = node_types['type'].value_counts()
    for t, c in type_counts.items():
        print(f"  {t:35s}  {c:7,d} nodes")

print("\nRELATION TYPES:")
rel_counts = kg['relation'].value_counts()
for r, c in rel_counts.items():
    print(f"  {r:40s}  {c:7,d} edges")

# ─── 3. Find our target nodes ─────────────────────────────────────────────────
# Genes/proteins relevant to sympathetic adrenergic TME signaling:
TARGET_GENES = [
    'ADRB1',   # beta-1 adrenergic receptor
    'ADRB2',   # beta-2 adrenergic receptor — main tumor TME target
    'ADRB3',   # beta-3 adrenergic receptor
    'ADRA1A',  # alpha-1A adrenergic receptor
    'ADRA2A',  # alpha-2A adrenergic receptor (guanfacine target)
    'TH',      # tyrosine hydroxylase — rate-limiting NE synthesis
    'DBH',     # dopamine beta-hydroxylase — NE synthesis
    'SLC6A2',  # NET (norepinephrine transporter) — NE reuptake
    'STAT3',   # downstream of beta-AR signaling in tumor cells
    'PRKA',    # PKA — cAMP/PKA axis downstream of beta-AR
    'PRKACA',  # PKA catalytic subunit
    'CREB1',   # CREB — downstream of PKA in stress response
    'MYC',     # c-Myc — upregulated via beta-AR/CREB in tumors
    'BCL2',    # anti-apoptotic, upregulated by beta-AR in tumors
    'VEGFA',   # angiogenesis, promoted by sympathetic signaling
    'MMP9',    # invasion/metastasis marker
    'PTPRC',   # CD45 — immune cell marker (for TME immune suppression)
    'CD274',   # PD-L1 — immune checkpoint (upregulated by NE signaling)
    'FOXP3',   # Treg marker — NE promotes Treg expansion
    'TGFB1',   # TGF-beta — immune suppression, promoted by NE
]

# Pathways:
TARGET_PATHWAYS = [
    'adrenergic signaling in cardiomyocytes',  # closest Reactome pathway
    'cAMP signaling pathway',
    'MAPK signaling pathway',
    'PI3K-Akt signaling pathway',
    'JAK-STAT signaling pathway',
]

print("\n" + "="*60)
print("SEARCHING FOR TARGET GENES/PROTEINS:")

# The node name column varies by KG version — try common ones
name_col = None
for candidate in ['x_name', 'node_name', 'name']:
    if candidate in kg.columns:
        name_col = candidate
        break

if name_col:
    gene_mask = kg['x_type'].isin(['gene/protein']) & kg[name_col].isin(TARGET_GENES)
    found_genes = kg[gene_mask][['x_id', name_col, 'x_type']].drop_duplicates()
    print(f"Found {len(found_genes)} target gene nodes:")
    print(found_genes.to_string(index=False))

    # Also check y_name
    y_name_col = name_col.replace('x_', 'y_')
    if y_name_col in kg.columns:
        gene_mask2 = kg['y_type'].isin(['gene/protein']) & kg[y_name_col].isin(TARGET_GENES)
        found_genes2 = kg[gene_mask2][['y_id', y_name_col, 'y_type']].drop_duplicates()
        found_genes2.columns = ['x_id', name_col, 'x_type']
        found_genes = pd.concat([found_genes, found_genes2]).drop_duplicates()

print(f"\nTotal unique target gene nodes found: {len(found_genes)}")
missing = set(TARGET_GENES) - set(found_genes[name_col])
if missing:
    print(f"NOT FOUND (will need manual ID lookup): {sorted(missing)}")

# ─── 4. Find which relation types connect drugs to genes (for edge mapping) ──
print("\n" + "="*60)
print("DRUG→GENE RELATION TYPES (for edge mapping reference):")
drug_gene = kg[
    (kg['x_type'] == 'drug') & (kg['y_type'] == 'gene/protein')
]['relation'].value_counts()
print(drug_gene)

print("\nDISEASE→GENE RELATION TYPES:")
dis_gene = kg[
    (kg['x_type'] == 'disease') & (kg['y_type'] == 'gene/protein')
]['relation'].value_counts()
print(dis_gene)

print("\nDISEASE→PHENOTYPE RELATION TYPES:")
dis_phen = kg[
    (kg['x_type'] == 'disease') & (kg['y_type'] == 'effect/phenotype')
]['relation'].value_counts()
print(dis_phen)

print("\nDISEASE→PATHWAY RELATION TYPES:")
dis_path = kg[
    (kg['x_type'] == 'disease') & (kg['y_type'] == 'pathway')
]['relation'].value_counts()
print(dis_path)

# ─── 5. Save audit results ────────────────────────────────────────────────────
audit = {
    'kg_shape': list(kg.shape),
    'node_types': type_counts.to_dict() if 'type_counts' in dir() else {},
    'relation_types': rel_counts.to_dict(),
    'found_target_genes': found_genes.to_dict(orient='records') if name_col else [],
    'missing_target_genes': list(missing) if 'missing' in dir() else [],
}
with open(os.path.join(OUT_DIR, 'kg_audit.json'), 'w') as f:
    json.dump(audit, f, indent=2)

print(f"\n✓ Audit saved to {OUT_DIR}/kg_audit.json")
print("\nNEXT STEP: Run 02_build_pseudo_node.py")
print("  Update TARGET_GENE_IDS in that file using the node IDs found here.")
