#!/bin/bash
export PYTHONPATH="/home/cbtil/Documents/SRDIFF/DisC-Diff"
mpiexec -n 2 python scripts/super_res_train.py --config config/config_train.yaml