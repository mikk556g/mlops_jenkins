import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# =========================================================
# IMPORTANT: DVC WORKFLOW (READ THIS FIRST)
# =========================================================
#
# 1. The dataset is tracked by DVC (NOT stored in Git).
#
# 2. BEFORE running training, you MUST download the data:
#
#       dvc pull
#
#    Run this command:
#       → In the project root folder
#       → ONCE per machine (NOT inside Python)
#
# 3. After "dvc pull", the dataset will appear locally
#    in the folder path defined below.
#
# =========================================================


# =========================================================
# INITIALIZE DISTRIBUTED TRAINING (DDP)
# =========================================================

# Set GPU for this process
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# Start distributed communication between processes
dist.init_process_group("nccl")

rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
device_id = local_rank


# =========================================================
# LOAD MODEL
# =========================================================

model = XXXXX().to(device_id)

# Wrap model so gradients sync across GPUs automatically
ddp_model = DDP(model, device_ids=[device_id])


# =========================================================
# LOAD DATASET FROM DVC LOCATION
# =========================================================

# This folder is tracked by DVC
DATA_DIR = "data/train_dataset"

# Safety check to remind you if you forgot dvc pull
if not os.path.exists(DATA_DIR):
    raise RuntimeError(
        "\nDataset not found!\n" "Run this BEFORE training:\n\n" "    dvc pull\n"
    )

# Create dataset normally (DVC is invisible here)
train_set = MyTrainDataset(DATA_DIR)

# Distributed sampler ensures each GPU gets different data
train_sampler = DistributedSampler(train_set)

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,  # Batch size PER GPU
    sampler=train_sampler,  # Important for DDP!
    pin_memory=True,
    shuffle=False,
)


# =========================================================
# TRAINING LOOP
# =========================================================


def train():
    for epoch in range(max_epochs):

        # VERY IMPORTANT:
        # Makes sure data is shuffled differently each epoch
        train_sampler.set_epoch(epoch)

        for batch in train_loader:
            inputs, targets = batch

            inputs = inputs.to(device_id)
            targets = targets.to(device_id)

            outputs = ddp_model(inputs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Only rank 0 should log/save (avoid duplicates)
        if rank == 0:
            print(f"Epoch {epoch} complete")
            torch.save(model.state_dict(), "model.pt")
