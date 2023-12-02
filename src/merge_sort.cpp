#include <CL/sycl.hpp>
#include <vector>
#include <iostream>

using namespace cl::sycl;

class merge_sort_kernel;

// 并行归并排序的主函数
void parallel_merge_sort(queue &q, buffer<int, 1> &data, int n) {
    // 根据硬件和数据大小确定合适的线程块大小
    const size_t local_size = 64; // 示例值
    const size_t global_size = ((n + local_size - 1) / local_size) * local_size;

    for (int size = 1; size < n; size = 2*size) {
        q.submit([&](handler &h) {
            auto acc = data.get_access<access::mode::read_write>(h);
            h.parallel_for<class merge_sort_kernel>(nd_range<1>(range<1>(global_size), range<1>(local_size)), [=](nd_item<1> item) {
            int i = item.get_global_id(0);

            if (i < n / (2*size) * (2*size)) {
                int left = i / (2*size) * (2*size);
                int mid = std::min(left + size - 1, n-1);
                int right = std::min(left + 2*size - 1, n-1);

                // 分配临时数组
                int *temp = new int[right - left + 1];

                // 合并左右子数组
                int k = 0;
                int l = left, r = mid + 1;
                while (l <= mid && r <= right) {
                    if (acc[l] < acc[r]) {
                        temp[k++] = acc[l++];
                    } else {
                        temp[k++] = acc[r++];
                    }
                }

                // 复制剩余的元素
                while (l <= mid) {
                    temp[k++] = acc[l++];
                }
                while (r <= right) {
                    temp[k++] = acc[r++];
                }

                // 将排序后的元素复制回原数组
                for (k = 0, l = left; l <= right; ++l, ++k) {
                    acc[l] = temp[k];
                }

                delete[] temp;
            }
        });

        }).wait();
    }
}

int main() {
    std::vector<int> arr = {1,4,2,9,5,6,0,7,5};
    const size_t n = arr.size();

    queue q;
    buffer<int, 1> data_buf(arr.data(), range<1>(n));

    parallel_merge_sort(q, data_buf, n);

    auto sorted_data = data_buf.get_access<access::mode::read>();
    for (size_t i = 0; i < n; i++) {
        std::cout << sorted_data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
