# 并行归并排序之SYCL实现

### 问题描述

使用基于oneAPI的C++/SYCL实现⼀个高效的并行归并排序。需要考虑数据的分割和合并以及线程之间的协作。

归并排序是⼀种分治算法，其基本原理是将待排序的数组分成两部分，分别对这两部分进行排序，然后将已排序的子数组合并为⼀个有序数组。可考虑利用了异构并行计算的特点，将排序和合并操作分配给多个线程同时执行，以提高排序效率。

在实际实现中，归并排序可使用共享内存来加速排序过程。

### 项目介绍和分析

本项目利用基于SYCL的编程模型在devcloud平台上实现归并排序，具体实现过程如下：

1. 将待排序的数组分割成多个较小的子数组，并将这些⼦数组分配给不同的线程块进行处理。

2. 每个线程块内部的线程协作完成子数组的局部排序。

3. 通过多次迭代，不断合并相邻的有序⼦数组，直到整个数组有序。

项目中我利用共享内存来存储临时数据，减少对全局内存的访问次数，从而提高了排序的效率。另外，在合并操作中，考虑同步机制来保证多个线程之间的数据⼀致性。

我还针对数组大小、线程块大小、数据访问模式等因素，设尝试出合适的参数设置，充分利用目标计算硬件GPU的并行计算能力，提高排序的效率和性能。

sycl源代码在src目录下，其中`src/parallel_merge_sort.cpp`源文件为使用Intel devcloud平台GPU硬件加速并优化的代码，`src/merge_sort.cpp`为仅使用Intel devcloud平台CPU设备的代码。

### 构建项目时采用的技术栈及主要实现方案

![image-20231202225124127](/Users/kaiyu/Library/Application Support/typora-user-images/image-20231202225124127.png)

上图中列出了Intel OneAPI对异构编程提供的开发工具，我在开发中使用了SYCL语言在VScode和Intel Devcloud平台上进行开发，使用了OpenMP和gdb-oneapi debugger工具。

### 运行

在项目目录下执行`qsub build.sh`，然后执行`qsub run.sh`，然后查看`run.sh.o_xxxx`，即可看到类似如下输出：

![image-20231203200334399](/Users/kaiyu/Library/Application Support/typora-user-images/image-20231203200334399.png)

### 结果

在sycl编程环境下，使用GPU进行加速并优化后，比仅使用CPU快**226**倍。

### [代码分析](https://github.com/yangkaiyu-web/sycl-merge_sort/blob/main/src/%E4%BB%A3%E7%A0%81%E8%A7%A3%E6%9E%90.md)

见src目录下。

### 收获

我通过完成第二个作业，真正掌握了sycl语言的编程方法，并行编程的能力得到极大提升，特别是在如何在多单元同时计算和worker group时对全局index和本地index的处理上有了深入理解。对归并排序的并行算法有较完整的了解和编程实践，见识到了SYCL并行编程在提升运行速度上的威力。