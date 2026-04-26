"""
04_pretrain.py  —  Pretrain R-GCN on augmented graph
=======================================================
Pretraining learns embeddings for ALL nodes including the pseudo-node,
by predicting all 29 relation types simultaneously.

The pseudo-node will get an embedding shaped by its neighborhood:
ADRB2, STAT3, CREB1, BCL2, VEGFA, etc. — this is the "biological context"
the GNN encodes before fine-tuning on drug-disease labels.

If you have GPU memory constraints, reduce n_hid and n_inp.
100-dim is the paper default; 64-dim works on 8GB VRAM.
"""

from txgnn import TxData, TxGNN
import os

DATA_DIR = './data'
CKPT_DIR = './checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)

# Load augmented data
TxData_obj = TxData(data_folder_path=DATA_DIR)
# Use our augmented split file
TxData_obj.prepare_split(
    split='complex_disease',
    seed=42,
    # If TxData accepts a custom split path, pass it here.
    # Otherwise, copy kg_augmented.csv to kg.csv before running.
)

# Point to augmented KG if TxData doesn't auto-detect:
# import shutil
# shutil.copy(f'{DATA_DIR}/kg_augmented.csv', f'{DATA_DIR}/kg.csv')

model = TxGNN(
    data=TxData_obj,
    weight_bias_track=False,  # set True to use wandb logging
    proj_name='SympathTME',
    exp_name='pretrain_augmented',
    device='cpu',
)

model.model_initialize(
    n_hid=100,             # hidden dimension (paper default)
    n_inp=100,             # input dimension
    n_out=100,             # output dimension
    proto=True,            # enable metric learning / disease pooling
    proto_num=3,           # number of similar diseases for pooling
    attention=False,       # False required for graph XAI later
    sim_measure='all_nodes_profile',  # use full neighborhood for similarity
    agg_measure='rarity',  # weight similar diseases by their rarity
    num_walks=200,
    walk_mode='bit',
    path_length=2,
)

print("Starting pretraining on augmented KG...")
print("This teaches the model the neighborhood of the pseudo-node.")

model.pretrain(
    n_epoch=2,             # 2 epochs is the paper default for pretraining
    learning_rate=1e-3,
    batch_size=1024,
    train_print_per_n=20,
)

model.save_model(os.path.join(CKPT_DIR, 'pretrained_augmented'))
print(f"\n✓ Pretrained model saved to {CKPT_DIR}/pretrained_augmented")
print("NEXT STEP: Run 05_finetune.py")


# ============================================================
# 05_finetune.py  —  Fine-tune on indication/contraindication
# ============================================================
"""
Fine-tuning trains the metric-learning decoder (the projection heads
for indication and contraindication) on drug-disease labels.

The pseudo-node's indication labels (propranolol, carvedilol, atenolol)
are in the training set. The model sees these positive examples and
learns that drugs modulating adrenergic signaling are "indications"
for our pseudo-disease.

It then generalizes this — via the learned metric space — to other
drugs that have similar embeddings to the known beta-blockers.
That generalization is the repurposing signal.

Expected fine-tuning time: ~15–30 minutes on GPU.
"""

# (Run after pretrain — in practice you'd run this as a separate script)

model.finetune(
    n_epoch=500,
    learning_rate=5e-4,
    train_print_per_n=25,
    valid_per_n=100,
    save_name=os.path.join(CKPT_DIR, 'finetuned_sympathetic_tme'),
)

print("\n✓ Fine-tuned model saved")
print("NEXT STEP: Run 06_query.py")
