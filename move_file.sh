#!/bin/bash

set -e

mkdir data/train ; mkdir data/test ; mkdir data/temp
for i in ./data/speechocean762/WAVE/SPEAKER*/*.WAV
do
	mv $i data/temp
done

python3.8 split_train_test_file.py

mv data/train ./
mv data/test ./
rm -rf ./data
