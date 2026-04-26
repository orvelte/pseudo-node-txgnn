"""
06_query.py  —  Run inference on the pseudo-disease node
=========================================================
After fine-tuning, ask the model: "what drugs are indications for
sympathetic_adrenergic_tme?"

The model will score all ~7,957 drug candidates in the KG.
We filter and inspect the ranked list.

WHAT TO LOOK FOR IN RESULTS:
  1. Positive controls (propranolol, carvedilol, atenolol) should appear
     in the top 50 for their OWN indication. If they don't, the fine-tuning
     signal was too weak — check that the training label edges were injected.

  2. Held-out beta-blockers (metoprolol, nadolol) should appear in top 100.
     This is the "generalization test" — the model should generalize the
     beta-blocker class without being told about these specific drugs.

  3. Novel candidates: drugs NOT in the beta-blocker class that score highly.
     These are the actual repurposing predictions. Look for:
       - Alpha-2 agonists (clonidine, guanfacine, dexmedetomidine)
       - STAT3 inhibitors (stattic, niclosamide — an antiparasitic)
       - cAMP modulators (phosphodiesterase inhibitors: theophylline, cilostazol)
       - PKA inhibitors
       - Anti-anxiety drugs (benzodiazepines, if they hit adrenergic nodes)
       - Potentially: semaglutide/liraglutide (GLP-1s have adrenergic crosstalk)
"""

import os
import json
import pandas as pd
from txgnn import TxData, TxGNN

DATA_DIR   = './data'
CKPT_DIR   = './checkpoints'
OUT_DIR    = './results'
os.makedirs(OUT_DIR, exist_ok=True)

PSEUDO_ID  = 'PSEUDO:sympathetic_adrenergic_tme'

# Load fine-tuned model
TxData_obj = TxData(data_folder_path=DATA_DIR)
TxData_obj.prepare_split(split='complex_disease', seed=42)

model = TxGNN(data=TxData_obj, weight_bias_track=False,
              proj_name='SympathTME', exp_name='query')
model.load_pretrained(os.path.join(CKPT_DIR, 'finetuned_sympathetic_tme'))

print("Running indication inference for pseudo-disease node...")

# ── Core query ────────────────────────────────────────────────────────────────
results = model.predict_disease(
    disease=PSEUDO_ID,
    relation='indication',
    save_result=True,
)
# results: DataFrame with columns [drug_id, drug_name, score, rank]

# Also run contraindication query (drugs that would WORSEN sympathetic TME)
contras = model.predict_disease(
    disease=PSEUDO_ID,
    relation='contraindication',
    save_result=True,
)

# ── Annotate with drug class metadata ─────────────────────────────────────────
# Load DrugBank-derived drug class info (should be in ./data/)
drug_meta_path = os.path.join(DATA_DIR, 'drug_meta.csv')
if os.path.exists(drug_meta_path):
    drug_meta = pd.read_csv(drug_meta_path)
    results = results.merge(drug_meta, on='drug_id', how='left')

# ── Mark our positive controls ────────────────────────────────────────────────
KNOWN_BETAS = {
    'DB00571': 'propranolol',
    'DB01136': 'carvedilol',
    'DB00819': 'atenolol',
    'DB00264': 'metoprolol',  # held-out
    'DB01203': 'nadolol',     # held-out
}
results['is_known_beta_blocker'] = results['drug_id'].isin(KNOWN_BETAS)
results['known_drug_name'] = results['drug_id'].map(KNOWN_BETAS)

# ── Rank normalization ─────────────────────────────────────────────────────────
results['rank_pct'] = results['rank'] / len(results) * 100  # lower = better

# ── Save full results ─────────────────────────────────────────────────────────
results_path = os.path.join(OUT_DIR, 'drug_ranking_indication.csv')
results.to_csv(results_path, index=False)

contras_path = os.path.join(OUT_DIR, 'drug_ranking_contraindication.csv')
contras.to_csv(contras_path, index=False)

# ── Report ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"TOP 30 DRUG CANDIDATES (indication for sympathetic adrenergic TME):")
print("="*60)
cols = ['rank', 'drug_name', 'score', 'rank_pct', 'is_known_beta_blocker']
print(results[cols].head(30).to_string(index=False))

print("\n" + "="*60)
print("POSITIVE CONTROL RANKS (beta-blockers should be near top):")
for db_id, name in KNOWN_BETAS.items():
    row = results[results['drug_id'] == db_id]
    if len(row) > 0:
        rank = row['rank'].values[0]
        score = row['score'].values[0]
        is_holdout = db_id in ['DB00264', 'DB01203']
        status = ' [HELD OUT]' if is_holdout else ' [TRAINING]'
        print(f"  {name:15s}{status}  rank: {rank:5d}  score: {score:.4f}")
    else:
        print(f"  {name:15s}  NOT FOUND in results")

print("\n" + "="*60)
print("TOP 10 NOVEL CANDIDATES (not known beta-blockers):")
novel = results[~results['is_known_beta_blocker']].head(10)
print(novel[['rank', 'drug_name', 'score']].to_string(index=False))

print(f"\n✓ Full results saved to {results_path}")
print("NEXT STEP: Run 07_explain.py on top novel candidates")


# ============================================================
# 07_explain.py  —  Multi-hop path explanations
# ============================================================
"""
For the top N novel candidates, pull the multi-hop KG paths that
drove the score. This is where you validate the mechanism.

WHAT MAKES A GOOD EXPLANATION:
  The highest-weight paths should run through adrenergic biology:
  drug → inhibits → ADRB2 → associated_with → cAMP_signaling
       → implicated_in → sympathetic_adrenergic_tme

  If paths run through unrelated biology (e.g., the drug is a kinase
  inhibitor and the paths go through EGFR), the score is a false positive
  driven by off-target KG connectivity, not the relevant mechanism.

WHAT TO DO WITH BAD PATHS:
  If a high-scoring drug has paths through irrelevant biology:
  1. Check if the drug has known adrenergic activity that's just not in the KG
  2. If not, deprioritize it — the model is hallucinating via spurious edges
  3. Add a note in results/explanation_paths.json explaining the discard
"""

print("\nRunning explanations for top 10 novel candidates...")

novel_candidates = results[~results['is_known_beta_blocker']].head(10)
explanations = {}

for _, row in novel_candidates.iterrows():
    drug_id   = row['drug_id']
    drug_name = row['drug_name']

    try:
        exp = model.explain(
            drug=drug_id,
            disease=PSEUDO_ID,
            relation='indication',
        )
        explanations[drug_name] = {
            'drug_id':     drug_id,
            'rank':        int(row['rank']),
            'score':       float(row['score']),
            'paths':       exp,  # list of path dicts with weights
        }
        print(f"\n  {drug_name} (rank {int(row['rank'])}):")
        if isinstance(exp, list):
            for path in exp[:3]:  # top 3 paths
                print(f"    → {path}")
        else:
            print(f"    → {exp}")

    except Exception as e:
        print(f"  ⚠ Could not explain {drug_name}: {e}")
        explanations[drug_name] = {'error': str(e)}

exp_path = os.path.join(OUT_DIR, 'explanation_paths.json')
with open(exp_path, 'w') as f:
    json.dump(explanations, f, indent=2)

print(f"\n✓ Explanations saved to {exp_path}")


# ============================================================
# 08_validate_with_lincs.py  —  LINCS L1000 cross-validation
# ============================================================
"""
Independent validation using transcriptomic drug perturbation profiles.

The idea:
  1. Get a gene expression signature of "beta-adrenergic activation in tumor"
     (NE treatment in MCF7 or MDA-MB-231 breast cancer cells — available on GEO)
  2. Compute the INVERSE signature (genes to push in the opposite direction)
  3. Query LINCS L1000 for drugs whose perturbation profiles match the inverse
  4. Any drug in BOTH the TxGNN top-50 AND the LINCS top-50 is a strong hit

This cross-validation is independent of TxGNN's KG — it validates purely
from transcriptomic evidence. Concordance between the two = mechanistic
confidence; discordance = one source has noise or incomplete coverage.

GEO datasets to use for the NE signature:
  GSE71657  : NE treatment in breast cancer cell lines
  GSE84071  : adrenergic stimulation in PDAC
  GSE47756  : stress hormone effects on cancer transcriptome
"""

import scipy.stats as stats

def compute_ks_score(gene_signature: dict, drug_profile: dict) -> float:
    """
    Connectivity Map-style KS enrichment score.
    gene_signature: {gene_id: log2FC} — the INVERSE of the disease signature
    drug_profile:   {gene_id: log2FC} — LINCS drug perturbation
    Returns: enrichment score in [-1, 1]. Positive = drug inverts the signature.
    """
    # Rank drug profile genes by absolute effect
    ranked = sorted(drug_profile.keys(), key=lambda g: -abs(drug_profile.get(g, 0)))
    sig_genes = set(gene_signature.keys())
    n_total = len(ranked)
    n_sig   = len(sig_genes)

    # KS running sum
    running = 0
    max_dev = 0
    for i, gene in enumerate(ranked):
        if gene in sig_genes:
            running += 1 / n_sig
        else:
            running -= 1 / (n_total - n_sig)
        max_dev = max(max_dev, abs(running))

    # Sign by whether hits are concentrated at top
    top_half_hits = sum(1 for g in ranked[:n_total//2] if g in sig_genes)
    sign = 1 if top_half_hits > n_sig / 2 else -1
    return sign * max_dev


def run_lincs_crossval(indication_results_path, lincs_profiles_dir,
                       ne_signature_path, top_n=50):
    """
    Full LINCS cross-validation pipeline.

    Parameters
    ----------
    indication_results_path : str
        CSV from 06_query.py
    lincs_profiles_dir : str
        Directory with LINCS L1000 drug perturbation profiles (one file per drug)
    ne_signature_path : str
        CSV with {gene_id, log2FC} for NE-treated cancer cells (from GEO)
    top_n : int
        How many TxGNN candidates to cross-validate
    """
    results = pd.read_csv(indication_results_path)
    top_candidates = results.head(top_n)

    # Load NE signature and invert it (we want drugs that REVERSE NE effects)
    ne_sig = pd.read_csv(ne_signature_path)
    ne_inverse = {row['gene_id']: -row['log2FC'] for _, row in ne_sig.iterrows()}

    crossval_scores = []
    for _, candidate in top_candidates.iterrows():
        drug_name = candidate['drug_name']
        drug_id   = candidate['drug_id']

        # Look for LINCS profile (by DrugBank ID or drug name)
        profile_path = os.path.join(lincs_profiles_dir, f'{drug_id}.csv')
        if not os.path.exists(profile_path):
            profile_path = os.path.join(lincs_profiles_dir, f'{drug_name}.csv')
        if not os.path.exists(profile_path):
            continue  # drug not in LINCS

        drug_profile_df = pd.read_csv(profile_path)
        drug_profile = dict(zip(drug_profile_df['gene_id'], drug_profile_df['log2FC']))

        ks = compute_ks_score(ne_inverse, drug_profile)
        crossval_scores.append({
            'drug_name':    drug_name,
            'drug_id':      drug_id,
            'txgnn_rank':   candidate['rank'],
            'txgnn_score':  candidate['score'],
            'lincs_ks':     ks,
        })

    crossval_df = pd.DataFrame(crossval_scores)
    crossval_df = crossval_df.sort_values('lincs_ks', ascending=False)

    out_path = os.path.join(OUT_DIR, 'lincs_crossval.csv')
    crossval_df.to_csv(out_path, index=False)

    print("\n" + "="*60)
    print("LINCS CROSS-VALIDATION RESULTS:")
    print("Drugs scoring highly on BOTH TxGNN AND LINCS = strongest candidates")
    print("="*60)
    print(crossval_df.to_string(index=False))

    # Flag concordant hits
    concordant = crossval_df[
        (crossval_df['txgnn_rank'] <= top_n) &
        (crossval_df['lincs_ks'] > 0.3)  # strong positive enrichment
    ]
    print(f"\n{'='*60}")
    print(f"CONCORDANT HITS (TxGNN top-{top_n} AND LINCS KS > 0.3):")
    print(concordant[['drug_name', 'txgnn_rank', 'lincs_ks']].to_string(index=False))
    print(f"\n✓ Saved to {out_path}")

    return crossval_df


# ── Main cross-validation call ────────────────────────────────────────────────
# Uncomment when you have:
#   1. LINCS profiles downloaded (https://clue.io/data/CMap2020)
#   2. NE signature from GEO (GSE71657 or similar)

# crossval_df = run_lincs_crossval(
#     indication_results_path='./results/drug_ranking_indication.csv',
#     lincs_profiles_dir='./data/lincs_profiles/',
#     ne_signature_path='./data/ne_tumor_signature.csv',
#     top_n=50,
# )

print("\n" + "="*60)
print("PIPELINE COMPLETE")
print("="*60)
print("Outputs:")
print("  results/drug_ranking_indication.csv   — full ranked drug list")
print("  results/drug_ranking_contraindication.csv")
print("  results/explanation_paths.json         — multi-hop KG paths")
print("  results/lincs_crossval.csv             — LINCS concordance")
print("\nKey interpretation questions:")
print("  1. Do held-out beta-blockers (metoprolol, nadolol) rank in top 100?")
print("     YES → model generalized correctly. NO → fine-tune signal too weak.")
print("  2. Do novel top-10 candidates have paths through ADRB2/STAT3/PKA?")
print("     YES → mechanistically plausible. NO → spurious KG connectivity.")
print("  3. Do TxGNN top-50 and LINCS top-50 overlap significantly?")
print("     YES → high-confidence repurposing candidates for wet lab follow-up.")
