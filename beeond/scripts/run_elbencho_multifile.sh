#!/bin/bash

env

echo "Got args: $*"

port=1161

datapath=$1
shift
numnodes=$1
shift

for i in $(seq 1 $numnodes); do
  newnode=$1
  shift
  nodelist="${nodelist}${newnode}:${port},"
done

#Remove trailing comma
nodelist=${nodelist%,}

elbencho --service --port ${port} --foreground &
service_pid=$!

ready="1"
count=0

while [ "$ready" -ne "0" ]; do
  echo "Waiting 30s for elbencho to come up"
  elbencho --hosts ${nodelist} $datapath
  ready=$?
  count=$(($count + 1))
  if [ "$count" -gt "10" ]; then
    echo "Failed to find elbencho workers"
    exit -1
  fi
  sleep 30
done

date > /data/$OMPI_COMM_WORLD_RANK.marker

if [ "$OMPI_COMM_WORLD_RANK" -eq "0" ]; then

threadrange="1 8 32 128"
totalfiles=12800
filesizerange="256k 1M 4M 64M"


for numthreads in ${threadrange}; do
  numfiles=$(($totalfiles / ${numthreads}))

  for filesize in ${filesizerange}; do
    blocksize=$filesize
    elbencho --hosts ${nodelist} -d -t $numthreads -N $numfiles -s $filesize -b $blocksize \
      --lat --lathisto \
      --csvfile ./outputs/multi_t${numthreads}_bs${blocksize}_fs${filesize}.csv $datapath \
      | tee -a ./outputs/multi_t${numthreads}_bs${blocksize}_fs${filesize}.log
    elbencho --hosts ${nodelist} --sync --dropcache -w -t $numthreads -N $numfiles -s $filesize -b $blocksize \
      --lat --lathisto \
      --csvfile ./outputs/multi_t${numthreads}_bs${blocksize}_fs${filesize}.csv $datapath \
      | tee -a ./outputs/multi_t${numthreads}_bs${blocksize}_fs${filesize}.log
    elbencho --hosts ${nodelist} --sync --dropcache -r -t $numthreads -N $numfiles -s $filesize -b $blocksize \
      --lat --lathisto \
      --csvfile ./outputs/multi_t${numthreads}_bs${blocksize}_fs${filesize}.csv $datapath \
      | tee -a ./outputs/multi_t${numthreads}_bs${blocksize}_fs${filesize}.log
    elbencho --hosts ${nodelist} -F -t $numthreads -N $numfiles -s $filesize -b $blocksize \
      --lat --lathisto \
      --csvfile ./outputs/multi_t${numthreads}_bs${blocksize}_fs${filesize}.csv $datapath \
      | tee -a ./outputs/multi_t${numthreads}_bs${blocksize}_fs${filesize}.log
    elbencho --hosts ${nodelist} -D -t $numthreads -N $numfiles -s $filesize -b $blocksize \
      --lat --lathisto \
      --csvfile ./outputs/multi_t${numthreads}_bs${blocksize}_fs${filesize}.csv $datapath \
      | tee -a ./outputs/multi_t${numthreads}_bs${blocksize}_fs${filesize}.log
    done
done

  elbencho --hosts ${nodelist} --quit
fi

wait $service_pid
