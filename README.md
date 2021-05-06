# KDD-Cup21-PCQM4M
codes for KDD Cup 21 PCQM4M competetion.


## Usage of Baseline models

```
LOG_DIR=./log
CHECKPOINT_DIR=./chkpoint
TEST_DIR=./test_results
```

### GIN

```
python main.py --gnn gin --log_dir $LOG_DIR --checkpoint_dir $CHECKPOINT_DIR --save_test_dir $TEST_DIR
```

### GIN-virtual

```
python main.py --gnn gin-virtual --log_dir $LOG_DIR --checkpoint_dir $CHECKPOINT_DIR --save_test_dir $TEST_DIR
```

### GCN

```
python main.py --gnn gcn --log_dir $LOG_DIR --checkpoint_dir $CHECKPOINT_DIR --save_test_dir $TEST_DIR
```

### GCN-virtual 

```
python main.py --gnn gcn-virtual --log_dir $LOG_DIR --checkpoint_dir $CHECKPOINT_DIR --save_test_dir $TEST_DIR
```
