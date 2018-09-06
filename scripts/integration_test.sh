#! /bin/sh

scripts/toy.sh

TRAIN_SRC=data/toy_reverse/train/src.txt
TRAIN_TGT=data/toy_reverse/train/tgt.txt
DEV_SRC=data/toy_reverse/dev/src.txt
DEV_TGT=data/toy_reverse/dev/tgt.txt

# Start training
python scripts/integration_test.py --train_src $TRAIN_SRC --train_tgt $TRAIN_TGT --dev_src $DEV_SRC --dev_tgt $DEV_TGT
# Resume training
python scripts/integration_test.py --train_src $TRAIN_SRC --train_tgt $TRAIN_TGT --dev_src $DEV_SRC --dev_tgt $DEV_TGT --resume
# Load checkpoint
python scripts/integration_test.py --train_src $TRAIN_SRC --train_tgt $TRAIN_TGT --dev_src $DEV_SRC --dev_tgt $DEV_TGT \
	--load_checkpoint $(ls -t experiment/checkpoints/ | head -1)
