#!/bin/bash

#PBS -l nodes=1:gpu:ppn=2
#PBS -d .

source /glob/development-tools/versions/fpgasupportstack/a10/1.2.1/intelFPGA_pro/hld/init_opencl.sh
source /glob/development-tools/versions/fpgasupportstack/a10/1.2.1/inteldevstack/init_env.sh
export FPGA_BBB_CCI_SRC=/usr/local/intel-fpga-bbb
export PATH=/glob/intel-python/python2/bin:${PATH}

echo
echo start: $(date "+%y/%m/%d %H:%M:%S.%3N")
echo

make all

echo
echo stop: $(date "+%y/%m/%d %H:%M:%S.%3N")
echo
