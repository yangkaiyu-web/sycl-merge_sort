#include <iostream>
#include <vector>
using namespace std;

void merge(vector<int> &array, int const left,
           int const mid, int const right)
{
    auto const subArrayOne = mid - left + 1;
    auto const subArrayTwo = right - mid;

    auto *leftArray = new int[subArrayOne],
         *rightArray = new int[subArrayTwo];

    for (auto i = 0; i < subArrayOne; i++)
        leftArray[i] = array[left + i];
    for (auto j = 0; j < subArrayTwo; j++)
        rightArray[j] = array[mid + 1 + j];

    auto indexOfSubArrayOne = 0,
         indexOfSubArrayTwo = 0;

    int indexOfMergedArray = left;
    while (indexOfSubArrayOne < subArrayOne &&
           indexOfSubArrayTwo < subArrayTwo)
    {
        if (leftArray[indexOfSubArrayOne] <=
            rightArray[indexOfSubArrayTwo])
        {
            array[indexOfMergedArray] =
                leftArray[indexOfSubArrayOne];
            indexOfSubArrayOne++;
        }
        else
        {
            array[indexOfMergedArray] =
                rightArray[indexOfSubArrayTwo];
            indexOfSubArrayTwo++;
        }
        indexOfMergedArray++;
    }

    while (indexOfSubArrayOne < subArrayOne)
    {
        array[indexOfMergedArray] =
            leftArray[indexOfSubArrayOne];
        indexOfSubArrayOne++;
        indexOfMergedArray++;
    }

    while (indexOfSubArrayTwo < subArrayTwo)
    {
        array[indexOfMergedArray] =
            rightArray[indexOfSubArrayTwo];
        indexOfSubArrayTwo++;
        indexOfMergedArray++;
    }
}

void mergeSort(vector<int> &array,
               int const begin,
               int const end)
{
    if (begin >= end)
        return;

    auto mid = begin + (end - begin) / 2;
    mergeSort(array, begin, mid);
    mergeSort(array, mid + 1, end);
    merge(array, begin, mid, end);
}

void printArray(vector<int> &A, int size)
{
    for (auto i = 0; i < size; i++)
        cout << A[i] << " ";
    cout << endl;
}

int main()
{
    std::vector<int> arr(200);
    for (int i = 0; i < 200; ++i)
    {
        arr[i] = 200 - i;
    }
    auto n = arr.size();

    cout << "Original array is " << endl;
    printArray(arr, n);

    mergeSort(arr, 0, n - 1);

    cout << "Sorted array is " << endl;
    printArray(arr, n);
    return 0;
}
