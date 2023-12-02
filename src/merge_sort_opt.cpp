#include <CL/sycl.hpp>
#include <vector>
#include <iostream>

using namespace cl::sycl;

class merge_sort_kernel;

// 并行归并排序的主函数
void parallel_merge_sort_local(queue &q, buffer<int, 1> &data, size_t n) {
    const size_t local_size = 128;
    const size_t global_size = ((n + local_size - 1) / local_size) * local_size;
    q.submit([&](handler &h) {
        sycl::stream out(1024, 1024, h);
        auto acc_data = data.get_access<access::mode::read_write>(h);
        local_accessor<int, 1> localA(range<1>(local_size), h), localB(range<1>(32), h);
        h.parallel_for<class merge_sort_kernel>(nd_range<1>(range<1>(global_size), range<1>(32)), [=](nd_item<1> item) {
            size_t i = item.get_global_id(0);
            if(i%32==0&&i<n)
            {
                out<<"------------------\n";
                for(int j=i;j<i+32;++j)
                {
                    out<<acc_data[j]<<" ";
                }
                out<<"\n------------------"<<sycl::endl;
            }
            size_t local_i = item.get_local_id(0);
            localA[local_i] = acc_data[i];
            item.barrier(access::fence_space::local_space);
            for (int size=1; size < std::min(n, local_size); size *= 2) {
                if (local_i < n && local_i % (2*size) == 0) {
                    size_t left = local_i;
                    size_t mid = std::min(left + size - 1, n-1);
                    size_t right = std::min(left + 2*size - 1, n-1);

                    size_t k = left;
                    size_t l = left, r = mid + 1;
                    while (l <= mid && r <= right) {
                        if (localA[l] < localA[r]) {
                            localB[k++] = localA[l++];
                        } else {
                            localB[k++] = localA[r++];
                        }
                    }

                    while (l <= mid) {
                        localB[k++] = localA[l++];
                    }
                    while (r <= right) {
                        localB[k++] = localA[r++];
                    }

                    for (k = left, l = left; l <= right; ++l, ++k) {
                        localA[l] = localB[k];
                    }
                }
                item.barrier(access::fence_space::local_space);
            }
            acc_data[i]=localA[local_i];
        });
    }).wait();

    /*
    std::vector<int> v(n);
    buffer<int, 1> buf_tmp(v.data(), v.size());

    for (int size = local_size; size < n; size *= 2)
    {
        const size_t global_size = (n + size*2 - 1) / (size*2);
        std::cout<<"size: " << size<<", global size: "<<global_size<<std::endl;
        q.submit([&](handler &h) {
            sycl::stream out(1024, 256, h);
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

                out <<  "data[" << left << "~" << right << "]: ";
                for (k = left, l = left; l <= right; ++l, ++k) {
                    acc_data[l] = acc_tmp[k];
                    out <<acc_data[l] << " ";
                }
                out << sycl::endl;
                item.barrier(access::fence_space::local_space);
            });
        }).wait();
    }
    */
}

int main() {
    std::vector<int> arr(200);
    for(int i=0;i<200;++i)
    {
        arr[i]=200-i;
    }
    const size_t n = arr.size();
    queue q;
    std::cout << q.get_device().get_info<sycl::info::device::max_work_group_size>() << std::endl;
    buffer<int, 1> data_buf(arr.data(), range<1>(n));

    parallel_merge_sort_local(q, data_buf, n);
    /*
    auto sorted_data = data_buf.get_host_access();
    std::cout<< "------------------" <<std::endl;
    for (size_t i = 0; i < n; i++) {
        std::cout << sorted_data[i] << " ";
    }
    std::cout << std::endl;
    std::cout<< "------------------" <<std::endl;
    */
    return 0;
}
