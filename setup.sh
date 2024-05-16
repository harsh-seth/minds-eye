#!/bin/bash
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
git clone https://github.com/xingyuansun/pix3d.git
mkdir -p data/pix3d && cd data/pix3d && ../../pix3d/download_dataset.sh && rm -rf ../../pix3d && cd ../..
echo "You still need to copy over the grid images to data/pix3d/grid_images!"
