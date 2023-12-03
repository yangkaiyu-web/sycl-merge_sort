
CXX = icpx

SYCL_CXXFLAGS = -std=c++17 -fsycl -O3 -o
SYCL_LDFLAGS = 
SYCL_EXE_NAME = merge_sort_cpu
SYCL_EXE_NAME_OPT = parallel_merge_sort
SYCL_SOURCES = src/merge_sort_cpu.cpp
SYCL_OPT_SOURCES = src/parallel_merge_sort.cpp

all:
	$(CXX) $(SYCL_CXXFLAGS) $(SYCL_EXE_NAME) $(SYCL_SOURCES) $(SYCL_LDFLAGS)
	$(CXX) $(SYCL_CXXFLAGS) $(SYCL_EXE_NAME_OPT) $(SYCL_OPT_SOURCES) $(SYCL_LDFLAGS)

run_opt:
	./$(SYCL_EXE_NAME_OPT) > /dev/null

run_cpu:
	./$(SYCL_EXE_NAME) > /dev/null

clean: 
	rm -rf $(SYCL_EXE_NAME)
	rm -rf $(SYCL_EXE_NAME_OPT)
