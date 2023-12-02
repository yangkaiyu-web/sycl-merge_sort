
CXX = icpx

SYCL_CXXFLAGS = -std=c++17 -fsycl -O3 -o
SYCL_LDFLAGS = 
SYCL_EXE_NAME = merge_sort
SYCL_EXE_NAME_OPT = merge_sort_opt
SYCL_SOURCES = src/merge_sort.cpp
SYCL_OPT_SOURCES = src/merge_sort_opt.cpp

all:
	$(CXX) $(SYCL_CXXFLAGS) $(SYCL_EXE_NAME) $(SYCL_SOURCES) $(SYCL_LDFLAGS)
	$(CXX) $(SYCL_CXXFLAGS) $(SYCL_EXE_NAME_OPT) $(SYCL_OPT_SOURCES) $(SYCL_LDFLAGS)

run:
	./$(SYCL_EXE_NAME) > /dev/null

run_cpu:
	./$(SYCL_EXE_NAME_OPT) > /dev/null

clean: 
	rm -rf $(SYCL_EXE_NAME)
	rm -rf $(SYCL_EXE_NAME_OPT)
