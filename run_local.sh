#!/bin/bash

mpirun -np 1 ./a.out \
	--fromMaxStepCount=1 \
	--countMaxStepCount=10 \
	--mDeltaMaxStepCount=2 \
	--fromStateCount=10 \
	--countStateCount=10 \
	--deltaStateCount=10 \
	--iterationCount=100 \
	--stopActionUsed \
	--threadBlockSize=4 \
	--runCount=4 \
	--seed=1345789478

