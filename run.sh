#!/bin/bash

# 设置集群环境（根据你的集群配置调整）
#SBATCH --job-name=winograd_convolution  # 作业名称
#SBATCH --nodes=1                        # 使用的节点数
#SBATCH --ntasks-per-node=1              # 每个节点的任务数
#SBATCH --cpus-per-task=64               # 每个任务的 CPU 核心数
#SBATCH --gres=gpu:2                    # 请求 GPU 资源（如果需要）
#SBATCH --output=output.log              # 输出日志文件
#SBATCH --exclusive
#SBATCH --exclude hepnode0

# 加载必要的模块（根据集群环境调整）
module purge
module load cuda/11.8.0                  # 加载 CUDA Toolkit

# 编译代码
nvcc -O3 -Xcompiler "-O3 -march=native -fopenmp" -lcublas -lcudart -o winograd winograd.cc driver.cc

# 检查编译是否成功
if [ $? -ne 0 ]; then
    echo "Compilation failed. Check the output for errors."
    exit 1
fi

# 运行程序
export OMP_NUM_THREADS=64                 # 设置 OpenMP 线程数
./winograd conf/vgg16.conf                               # 运行程序

# 结束
echo "Job completed."


