#include <CL/sycl.hpp>
#include <vector>
#include <iostream>

using namespace cl::sycl;

class merge_sort_kernel;

void printArray(std::vector<int> &A, int size)
{
    for (auto i = 0; i < size; i++)
        std::cout << A[i] << " ";
    std::cout << std::endl;
}

// 并行归并排序的主函数
void parallel_merge_sort_local(queue &q, buffer<int, 1> &data, size_t n)
{
    const size_t local_size = 128;                                               // local memory的大小
    const size_t global_size = ((n + local_size - 1) / local_size) * local_size; // 将原数组划分成的子数组后索引下标相应扩展以被local_size整除分割
    q.submit([&](handler &h)
             {
        // sycl::stream out(4096, 4096, h);
        auto acc_data = data.get_access<access::mode::read_write>(h);
        local_accessor<int, 1> localA(range<1>(local_size), h), localB(range<1>(local_size), h); // localA负责存放原数组的local_size子数组，localB负责归并时的临时存储
        h.parallel_for<class merge_sort_kernel>(nd_range<1>(range<1>(global_size), range<1>(local_size)), [=](nd_item<1> item)
                                                {
            size_t i = item.get_global_id(0);
            size_t local_i = item.get_local_id(0);
            if (i < n)
                localA[local_i] = acc_data[i]; // 只对在原数组中的元素加载到local，多出的长度只为方便分割而扩展
            size_t base = (i / local_size) * local_size;
            item.barrier(access::fence_space::local_space); // 确保都加载到了local
            // 以下部分为使用local memory进行加速归并排序，最终得到多个内部递增有序的长度为128大小的子数组（最后一个
            // 子数组可能小于128），从size=1开始，对{[0, size-1], [size,2*{size}-1]}, {[2*size,3*size-1],
            // [3*size,4*size-1]}...进行按增序归并，以使local index在[0,1]、[2,3]、[4,5]、[6,7]...范围内的子
            // 数组分别有序，每次size增大一倍，直到达到local_size或者完整数组大小（取两者中小的）
            for (int size = 1; size < std::min(n - base, local_size); size *= 2)
            {
                if (local_i % (2 * size) == 0)
                {
                    size_t left = local_i;
                    size_t mid = std::min(base + left + size - 1, n - 1) - base;
                    size_t right = std::min(base + left + 2 * size - 1, n - 1) - base;

                    size_t k = left;
                    size_t l = left, r = mid + 1;
                    while (l <= mid && r <= right)
                    {
                        if (localA[l] < localA[r])
                        {
                            localB[k++] = localA[l++];
                        }
                        else
                        {
                            localB[k++] = localA[r++];
                        }
                    }

                    while (l <= mid)
                    {
                        localB[k++] = localA[l++];
                    }
                    while (r <= right)
                    {
                        localB[k++] = localA[r++];
                    }

                    for (k = left, l = left; l <= right; ++l, ++k)
                    {
                        localA[l] = localB[k];
                    }
                }
                item.barrier(access::fence_space::local_space);
            }
            if(i < n)
                acc_data[i]=localA[local_i]; }); })
        .wait();

    std::vector<int> v(n);
    buffer<int, 1> buf_tmp(v.data(), v.size());

    // 以下部分为对多个有序子数组排序得到完整的有序数组，不使用local memory因为数组过长，device local存放不下
    for (int size = local_size; size < n; size *= 2)
    {
        const size_t global_size = (n + size * 2 - 1) / (size * 2);
        // std::cout<<"size: " << size<<", global size: "<<global_size<<std::endl;
        q.submit([&](handler &h)
                 {
            // sycl::stream out(1024, 256, h);
            auto acc_tmp = buf_tmp.get_access<access::mode::read_write>(h);
            auto acc_data = data.get_access<access::mode::read_write>(h);
            h.parallel_for<class merge_kernel>(sycl::nd_range<1>(range<1>(global_size), range<1>(1)), [=](nd_item<1> item) {
                size_t i = item.get_global_id(0);
                size_t left = i*2*size;
                size_t mid = std::min(left + size - 1, n-1);
                size_t right = std::min(left + 2*size - 1, n-1);

                size_t k = left;
                size_t l = left, r = mid + 1;
                while (l <= mid && r <= right) {
                    if (acc_data[l] < acc_data[r]) {
                        acc_tmp[k++] = acc_data[l++];
                    } else {
                        acc_tmp[k++] = acc_data[r++];
                    }
                }

                while (l <= mid) {
                    acc_tmp[k++] = acc_data[l++];
                }
                while (r <= right) {
                    acc_tmp[k++] = acc_data[r++];
                }

                // out <<  "data[" << left << "~" << right << "]: ";
                for (k = left, l = left; l <= right; ++l, ++k) {
                    acc_data[l] = acc_tmp[k];
                    // out <<acc_data[l] << " ";
                }
                // out << sycl::endl;
                item.barrier(access::fence_space::local_space);
            }); })
            .wait();
    }
}

int main()
{
    std::vector<int> arr(200);
    for (int i = 0; i < 200; ++i)
    {
        arr[i] = 200 - i;
    }
    const size_t n = arr.size();
    std::cout << "Original array is " << std::endl;
    printArray(arr, n);
    queue q;
    // std::cout << q.get_device().get_info<sycl::info::device::max_work_group_size>() << std::endl;
    buffer<int, 1> data_buf(arr.data(), range<1>(n));

    parallel_merge_sort_local(q, data_buf, n);
    auto sorted_data = data_buf.get_host_access();
    // std::cout<< "------------------" <<std::endl;
    std::cout << "Sorted array is " << std::endl;
    for (int i = 0; i < n; ++i)
    {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
    // std::cout << "------------------" << std::endl;
    return 0;
}
