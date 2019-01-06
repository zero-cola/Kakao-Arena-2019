#!bin/bash
nohup python train.py > $1.log & disown
