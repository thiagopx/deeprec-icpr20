NRUNS=10
for run in $(seq 1 $NRUNS);
do
     LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64 python pre_train.py --run $run
done