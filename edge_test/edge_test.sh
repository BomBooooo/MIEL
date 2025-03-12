#!/bin/bash

docker_id="" 
results_dir="results"
for pl in 96 336 960 1680
do
  for i in 0 1 2
  do
    sudo perf stat -a -e "power/energy-pkg/,power/energy-ram/,mem-loads" sudo docker run -it --rm --cpus=2 --gpus all $docker_id python edge_test.py $pl 0 "ETTh1" 0 1 &> "${results_dir}/ETTh1_${pl}_warmup_result_cpu_${i}.txt"
    sudo perf stat -a -e "power/energy-pkg/,power/energy-ram/,mem-loads" sudo docker run -it --rm --cpus=2 --gpus all $docker_id python edge_test.py $pl 1 "ETTh1" 0 1 &> "${results_dir}/ETTh1_${pl}_result_cpu_${i}.txt"
  done
  echo "Finished ${data_name} ${pl}"
done
