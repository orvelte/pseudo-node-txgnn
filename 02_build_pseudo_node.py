"""
02_build_pseudo_node.py
=======================
Defines the "sympathetic_adrenergic_tme" pseudo-disease node and all its
edges to existing KG nodes.

BIOLOGY ENCODED HERE:
  The sympathetic nervous system releases norepinephrine (NE) into the tumor
  microenvironment. NE binds β2-adrenergic receptors (ADRB2) on tumor cells,
  activating the cAMP→PKA→CREB axis, which upregulates pro-survival genes
  (BCL2, MYC), promotes immune evasion (via STAT3, PD-L1, Treg expansion),
  and drives angiogenesis (VEGFA) and invasion (MMP9). This is a "disease-like"
  state in the sense that it's a maladaptive biological process that drugs can
  intervene on.

EDGE MAPPING RATIONALE:
  We cannot add new relation types without retraining from scratch. We map our
  biology to the 29 existing PrimeKG relation types:

  Biological relationship             → PrimeKG relation (internal key)   display_relation
  ─────────────────────────────────   ─────────────────────────────────   ────────────────
  pseudo-disease activates gene       → "disease_protein"                 "associated with"
  pseudo-disease causes phenotype     → "disease_phenotype_positive"      "phenotype present"
  drug inhibits β-AR (indication)     → "indication"                      "indication"
  drug worsens NE signaling (contra)  → "contraindication"                "contraindication"

  "disease_protein" is the broadest disease→gene relation in PrimeKG.
  No disease→pathway relation type exists; pathway edges are omitted.

AFTER RUNNING THIS SCRIPT:
  Inspect edges/sympathetic_tme_edges.tsv carefully.
  Add any additional edges you can justify from the literature.
  Remove any you can't support with a citation.
  The biological quality of this edge list IS the scientific contribution.
"""

import os
import pandas as pd

EDGES_DIR = './edges'
OUT_DIR   = './results'
os.makedirs(EDGES_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Pseudo-node definition ───────────────────────────────────────────────────
# Must have a unique ID not in the existing KG.
# node_type MUST be "disease" for the TxGNN decoder to score drugs against it.
PSEUDO_NODE = {
    'node_id':   'PSEUDO:sympathetic_adrenergic_tme',
    'node_type': 'disease',
    'node_name': 'sympathetic adrenergic tumor microenvironment signaling',
    'node_source': 'curated_synthetic',
}

# ─── Target node IDs (fill from 01_audit_kg.py output) ───────────────────────
# These must be real PrimeKG node IDs from your kg_audit.json.
# Placeholders shown — replace with actual IDs after running audit.
# Format is PrimeKG integer node ID.
#
# HOW TO FILL THESE IN:
#   1. Run 01_audit_kg.py
#   2. Open results/kg_audit.json
#   3. Find each gene in found_target_genes list
#   4. Copy the x_id value into the dict below
#
# Example from PrimeKG (IDs are NCBI Gene IDs for gene/protein nodes):
GENE_IDS = {
    'ADRB2':   154,      # Beta-2 adrenergic receptor — PRIMARY TARGET
    'ADRB1':   153,      # Beta-1 adrenergic receptor
    'ADRB3':   155,      # Beta-3 adrenergic receptor
    'ADRA1A':  148,      # Alpha-1A adrenergic receptor
    'ADRA2A':  150,      # Alpha-2A adrenergic receptor
    'TH':      7054,     # Tyrosine hydroxylase (NE synthesis)
    'DBH':     1621,     # Dopamine beta-hydroxylase (NE synthesis)
    'SLC6A2':  6530,     # Norepinephrine transporter (reuptake)
    'STAT3':   6774,     # Signal transducer (downstream of beta-AR)
    'PRKACA':  5566,     # PKA catalytic subunit alpha
    'CREB1':   1385,     # cAMP response element binding protein
    'MYC':     4609,     # Proto-oncogene c-Myc (NE upregulates)
    'BCL2':    596,      # Anti-apoptotic (NE promotes survival)
    'VEGFA':   7422,     # Angiogenesis (SNS promotes)
    'MMP9':    4318,     # Matrix metalloproteinase 9 (invasion)
    'CD274':   29126,    # PD-L1 (NE promotes immune evasion)
    'FOXP3':   50943,    # Treg transcription factor
    'TGFB1':   7040,     # TGF-beta 1 (immunosuppression)
    'PTPRC':   5788,     # CD45 (immune cell marker for TME context)
    'NFKB1':   4790,     # NF-kB (inflammatory signaling, NE activates)
}

# Phenotype/effect node IDs — PrimeKG uses integer IDs (not HP: prefix).
# These were confirmed present in PrimeKG via 01_audit_kg.py.
# Note: PrimeKG HPO coverage does not include angiogenesis or metastasis as
# standalone phenotype nodes; those concepts live in biological_process nodes
# which have no direct disease→bioprocess relation type in PrimeKG.
PHENOTYPE_IDS = {
    'Abnormal cell proliferation':              31377,  # tumor proliferation proxy
    'Abnormality of immune system physiology':  10978,  # immune evasion proxy
    'Abnormality of the autonomic nervous system': 2270, # sympathetic nervous system
}

# Pathway edges are omitted: PrimeKG has no disease→pathway relation type.
# The 29 fixed relation types only allow disease_protein and
# disease_phenotype_positive for disease outgoing edges.

# ─── Positive control drugs (known beta-blocker cancer biology) ──────────────
# These are the drugs we KNOW should score as indications.
# We include them as training edges so the fine-tuned model has ground truth.
# DO NOT include all beta-blockers — leave some out as a held-out test set.
#
# DrugBank IDs (DB prefix)
KNOWN_INDICATION_DRUGS = {
    'propranolol': 'DB00571',   # non-selective beta-blocker, most evidence
    'carvedilol':  'DB01136',   # non-selective beta-blocker + alpha-blocker
    'atenolol':    'DB00819',   # cardioselective beta-blocker
}

# Hold these out as test set — model should rediscover them
HELD_OUT_TEST_DRUGS = {
    'metoprolol':  'DB00264',   # cardioselective
    'nadolol':     'DB01203',   # non-selective
}

# ─── Build the edge list ──────────────────────────────────────────────────────
edges = []
PSEUDO_ID = PSEUDO_NODE['node_id']

def add_edge(x_id, x_type, x_name, relation, y_id, y_type, y_name, evidence):
    """Append a directed edge with biological justification."""
    edges.append({
        'x_id':     x_id,
        'x_type':   x_type,
        'x_name':   x_name,
        'relation': relation,
        'y_id':     y_id,
        'y_type':   y_type,
        'y_name':   y_name,
        'evidence': evidence,  # NOT used by TxGNN, but essential for our records
    })

# --- PSEUDO-DISEASE → GENE EDGES (relation: "associated with") ---------------
# Primary mechanistic target
add_edge(PSEUDO_ID, 'disease', 'sympathetic adrenergic TME',
         'disease_protein',
         GENE_IDS['ADRB2'], 'gene/protein', 'ADRB2',
         'NE from sympathetic nerves binds ADRB2 on tumor cells; '
         'Sood 2010 NatMed, Thaker 2006 NatMed')

add_edge(PSEUDO_ID, 'disease', 'sympathetic adrenergic TME',
         'disease_protein',
         GENE_IDS['ADRB1'], 'gene/protein', 'ADRB1',
         'ADRB1 expressed on stromal and immune cells in TME; '
         'Nagaraja 2019 Cancer Cell')

add_edge(PSEUDO_ID, 'disease', 'sympathetic adrenergic TME',
         'disease_protein',
         GENE_IDS['ADRA2A'], 'gene/protein', 'ADRA2A',
         'Alpha-2A AR expressed on sympathetic terminals; '
         'presynaptic inhibition relevant to SNS-TME crosstalk; '
         'Geng 2023 Br J Cancer')

# NE synthesis pathway genes
add_edge(PSEUDO_ID, 'disease', 'sympathetic adrenergic TME',
         'disease_protein',
         GENE_IDS['TH'], 'gene/protein', 'TH',
         'TH (rate-limiting NE synthesis) expressed in tumor-infiltrating '
         'sympathetic fibers; Sloan 2010 Brain Behav Immun')

add_edge(PSEUDO_ID, 'disease', 'sympathetic adrenergic TME',
         'disease_protein',
         GENE_IDS['DBH'], 'gene/protein', 'DBH',
         'DBH converts dopamine to NE in sympathetic terminals; '
         'SNS denervation (DBH KO) reduces tumor growth; Cole 2015 PLoS ONE')

add_edge(PSEUDO_ID, 'disease', 'sympathetic adrenergic TME',
         'disease_protein',
         GENE_IDS['SLC6A2'], 'gene/protein', 'SLC6A2',
         'NET reuptake transporter controls synaptic NE availability; '
         'NET inhibitors (reboxetine) modulate TME NE; Sloan 2016')

# Downstream signaling
add_edge(PSEUDO_ID, 'disease', 'sympathetic adrenergic TME',
         'disease_protein',
         GENE_IDS['PRKACA'], 'gene/protein', 'PRKACA',
         'Beta-AR → Gs → adenylyl cyclase → cAMP → PKA (PRKACA); '
         'PKA phosphorylates CREB and promotes survival; Thaker 2006')

add_edge(PSEUDO_ID, 'disease', 'sympathetic adrenergic TME',
         'disease_protein',
         GENE_IDS['CREB1'], 'gene/protein', 'CREB1',
         'PKA phosphorylates CREB1 at Ser133; CREB drives BCL2, VEGFA '
         'transcription in NE-stimulated tumor cells; Sood 2006 Cancer')

add_edge(PSEUDO_ID, 'disease', 'sympathetic adrenergic TME',
         'disease_protein',
         GENE_IDS['STAT3'], 'gene/protein', 'STAT3',
         'NE activates STAT3 in tumor cells via beta-AR; STAT3 promotes '
         'immune evasion and proliferation; PDAC beta-AR loop Nigri 2024')

# Pro-survival / oncogenic effectors
add_edge(PSEUDO_ID, 'disease', 'sympathetic adrenergic TME',
         'disease_protein',
         GENE_IDS['BCL2'], 'gene/protein', 'BCL2',
         'NE upregulates BCL2 via CREB, suppressing apoptosis in tumor cells; '
         'Sood 2010 NatMed')

add_edge(PSEUDO_ID, 'disease', 'sympathetic adrenergic TME',
         'disease_protein',
         GENE_IDS['MYC'], 'gene/protein', 'MYC',
         'c-Myc upregulated downstream of beta-AR signaling; '
         'Lamkin 2016 Breast Cancer Res')

add_edge(PSEUDO_ID, 'disease', 'sympathetic adrenergic TME',
         'disease_protein',
         GENE_IDS['VEGFA'], 'gene/protein', 'VEGFA',
         'NE promotes VEGFA secretion from tumor cells via PKA/CREB; '
         'promotes tumor angiogenesis; Lutgendorf 2003 JNCI')

add_edge(PSEUDO_ID, 'disease', 'sympathetic adrenergic TME',
         'disease_protein',
         GENE_IDS['MMP9'], 'gene/protein', 'MMP9',
         'Beta-AR activation upregulates MMP9 to promote invasion; '
         'Cole 2010 J Biol Chem')

# Immune evasion nodes
add_edge(PSEUDO_ID, 'disease', 'sympathetic adrenergic TME',
         'disease_protein',
         GENE_IDS['CD274'], 'gene/protein', 'CD274',
         'NE upregulates PD-L1 (CD274) on tumor cells, impairing CD8+ T cells; '
         'Geng 2023 Br J Cancer')

add_edge(PSEUDO_ID, 'disease', 'sympathetic adrenergic TME',
         'disease_protein',
         GENE_IDS['FOXP3'], 'gene/protein', 'FOXP3',
         'NE promotes Treg expansion via beta-AR on T cells; '
         'Muthuswamy 2017 Cancer Res')

add_edge(PSEUDO_ID, 'disease', 'sympathetic adrenergic TME',
         'disease_protein',
         GENE_IDS['TGFB1'], 'gene/protein', 'TGFB1',
         'Sympathetic activation promotes TGF-beta1 secretion; '
         'immunosuppressive TME remodeling; Powell 2018 Sci Rep')

add_edge(PSEUDO_ID, 'disease', 'sympathetic adrenergic TME',
         'disease_protein',
         GENE_IDS['NFKB1'], 'gene/protein', 'NFKB1',
         'NE activates NF-kB via beta-AR in macrophages and tumor cells; '
         'promotes inflammatory pro-tumor cytokines; Cole 2015')

# --- PSEUDO-DISEASE → PHENOTYPE EDGES (relation: "phenotype present") --------
for phen_name, phen_id in PHENOTYPE_IDS.items():
    add_edge(PSEUDO_ID, 'disease', 'sympathetic adrenergic TME',
             'disease_phenotype_positive',
             phen_id, 'effect/phenotype', phen_name,
             f'Sympathetic adrenergic TME signaling presents with {phen_name}')

# --- KNOWN INDICATION TRAINING EDGES -----------------------------------------
# These tell the model: propranolol, carvedilol, atenolol are "indications"
# for our pseudo-disease. This gives it a training signal.
# The model should then generalize to other beta-blockers and mechanistically
# related drugs it wasn't told about.
for drug_name, drug_id in KNOWN_INDICATION_DRUGS.items():
    add_edge(drug_id, 'drug', drug_name,
             'indication',
             PSEUDO_ID, 'disease', 'sympathetic adrenergic TME',
             f'{drug_name} is a beta-blocker with evidence of anti-tumor '
             f'activity via beta-AR blockade in breast/PDAC/prostate cancer')

# ─── Export ──────────────────────────────────────────────────────────────────
df = pd.DataFrame(edges)
edge_path = os.path.join(EDGES_DIR, 'sympathetic_tme_edges.tsv')
df.to_csv(edge_path, sep='\t', index=False)

print(f"Built {len(df)} edges for pseudo-node '{PSEUDO_ID}'")
print(f"\nEdge type breakdown:")
print(df['relation'].value_counts())
print(f"\nConnected node types:")
print(df['y_type'].value_counts())
print(f"\n✓ Saved to {edge_path}")

# ─── Sanity checks ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SANITY CHECKS:")

# 1. Minimum edge density
n_edges = len(df)
assert n_edges >= 20, f"Too few edges ({n_edges}). Add more — disease pooling needs ≥20."
print(f"  ✓ Edge count: {n_edges} (target: ≥20)")

# 2. At least one indication training edge
n_indications = (df['relation'] == 'indication').sum()
assert n_indications >= 1, "Must have at least one indication training edge."
print(f"  ✓ Indication training edges: {n_indications}")

# 3. No self-loops
self_loops = df[df['x_id'] == df['y_id']]
assert len(self_loops) == 0, "Self-loops detected — check edge definitions."
print(f"  ✓ No self-loops")

# 4. Pseudo-node is always x_id for disease edges (not y_id except for drug→disease)
disease_as_target = df[(df['y_id'] == PSEUDO_ID) & (df['relation'] != 'indication')]
if len(disease_as_target) > 0:
    print(f"  ⚠ {len(disease_as_target)} non-indication edges have pseudo-node as target — verify this is intentional")
else:
    print(f"  ✓ Edge directionality looks correct")

print("\nNEXT STEP: Run 03_inject_and_split.py")
print("  IMPORTANT: Verify that all node IDs in GENE_IDS, PHENOTYPE_IDS,")
print("  and PATHWAY_IDS actually exist in the KG (check kg_audit.json).")
print("  Replace placeholder IDs with real ones before proceeding.")
