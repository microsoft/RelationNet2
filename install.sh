#!/usr/bin/env bash
MMDET=${1:-/tmp/mmdet}
CUR=$(pwd)
git clone https://github.com/open-mmlab/mmdetection.git ${MMDET}
cd ${MMDET}
git checkout c8511649550834ea168f610411a47a39cf194767
cd ${CUR}
cp -r ./code/* ${MMDET}/mmdet/
cp -r ./configs/* ${MMDET}/configs/
cd ${MMDET}
python -m pip install -e . 
