#!/bin/bash
#
#SBATCH --job-name=toi_web
#SBATCH --output=/home/runner/work/tess-atlas/tess-atlas/test_jobgen/log_web/web_%j.log
#
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1







module load module load 1

source /opt/hostedtoolcache/Python/3.8.17/x64/bin/activate

make_webpages --webdir webpages --notebooks test_jobgen --add-api