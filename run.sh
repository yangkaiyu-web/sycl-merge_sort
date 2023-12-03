#!/usr/bin/awk -f
#PBS -l nodes=1:gpu:ppn=2
#PBS -d .

source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1

convert_to_seconds() {
    local time_str=$1
    local minutes=$(echo $time_str | cut -d'm' -f1)
    local seconds=$(echo $time_str | cut -d'm' -f2 | sed 's/s//')
    echo "$minutes $seconds" | awk '{print $2}'
}

# Run the command using GPU and capture its execution time
time1=$( { time make run_opt; } 2>&1 | grep real | awk '{print $2}' )

# Run the command using CPU only and capture its execution time
time2=$( { time make run_cpu; } 2>&1 | grep real | awk '{print $2}' )

time1_sec=$(convert_to_seconds $time1)
time2_sec=$(convert_to_seconds $time2)

ratio1=$(echo "$time1_sec $time2_sec" | awk '{print $1 / $2}')

echo "Speed-up ratio (parallel_merge_sort / non_parallel_sort): $ratio1"
