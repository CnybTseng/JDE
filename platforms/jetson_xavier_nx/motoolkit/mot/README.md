# Debug segmentation fault
$ ulimit -c unlimited <br>
$ gdb jde core <br>

# Profile computation latency
https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/clocks.html <br>
Configuring GPU Clocks <br>
With GPU DVFS enabled using the devfreq framework, GPU frequency changes based on load. You can instead run the GPU at a fixed frequency if necessary. <br>
To run the GPU at a fixed frequency <br>
## 1.Enter the following command.
$ cd /sys/devices/gpu.0/devfreq/17000000.gv11b/ <br>

## 2.List the available frequencies:
$ cat available_frequencies <br>
114750000 204000000 306000000 408000000 510000000 599250000 701250000 752250000 803250000 854250000 905250000 956250000 1007250000 1058250000 1109250000 <br>

$ cat min_freq <br>
114750000 <br>

$ cat max_freq <br>
1109250000 <br>

## 3.Fix the frequency.
* To fix the frequency at maximum supported frequency: <br>
echo <max> min_freq <br>
Where <max> in the maximum supported frequency. For example, to fix the frequency at the maximum supported frequency from the list in step 2: <br>
echo 1109250000 > min_freq <br>
echo 114750000 > min_freq <br>
* To fix any other supported frequency: <br>
echo <freq> > min_freq <br>
echo <freq> > max_freq <br>
Where <freq> in the desired supported frequency. For example, to fix the frequency at 828750000: <br>
echo 828750000 > min_freq <br>
echo 828750000 > max_freq <br>

# Enable all CPUs
$ echo 1 > /sys/devices/system/cpu/cpu2/online <br>
$ echo 1 > /sys/devices/system/cpu/cpu3/online <br>
$ echo 1 > /sys/devices/system/cpu/cpu4/online <br>
$ echo 1 > /sys/devices/system/cpu/cpu5/online <br>

# Memmory leak detection
$ sudo apt install valgrind <br>
valgrind --leak-check=full \ <br>
         --show-leak-kinds=all \ <br>
         --track-origins=yes \ <br>
         --verbose \ <br>
         --log-file=valgrind-out.txt \ <br>
         ./mot-test ./mot.yaml ../../data/mini/ <br>



# Optimization

* before optimization <br>
========== mot profile ==========
                                                   TensorRT layer name    Runtime, %  Invocations  Runtime, ms
                                                   image preprocessing          9.8%         1500      4345.58
                                                         JDE inference         46.5%         1500     20686.45
                                                            JDE decode          7.7%         1500      3421.50
                                                    online association         36.1%         1500     16043.61
========== mot total runtime = 44497.1 ms ==========

* optimization 1 <br>
========== mot profile ==========
                                                   TensorRT layer name    Runtime, %  Invocations  Runtime, ms
                                                   image preprocessing          9.8%         1500      3966.79
                                                         JDE inference         49.8%         1500     20202.26
                                                            JDE decode          8.6%         1500      3474.85
                                                    online association         31.8%         1500     12907.69
========== mot total runtime = 40551.6 ms ==========

* optimization 2 <br>
========== mot profile ==========
                                                   TensorRT layer name    Runtime, %  Invocations  Runtime, ms
                                                   image preprocessing         10.4%         1500      4127.62
                                                         JDE inference         50.8%         1500     20127.52
                                                            JDE decode          8.5%         1500      3377.99
                                                    online association         30.2%         1500     11975.58
========== mot total runtime = 39608.7 ms ==========

* optimization 3 <br>
========== mot profile ==========
                                                   TensorRT layer name    Runtime, %  Invocations  Runtime, ms
                                                   image preprocessing         10.7%         1500      3754.32
                                                         JDE inference         56.3%         1500     19711.72
                                                            JDE decode          8.8%         1500      3079.26
                                                    online association         24.1%         1500      8442.25
========== mot total runtime = 34987.5 ms ==========

* optimization 4 <br>
========== mot profile ==========
                                                   TensorRT layer name    Runtime, %  Invocations  Runtime, ms
                                                   image preprocessing         12.1%         1500      3416.70
                                                         JDE inference         51.1%         1500     14436.14
                                                            JDE decode         10.4%         1500      2937.04
                                                    online association         26.4%         1500      7473.78
========== mot total runtime = 28263.7 ms ==========

* optimization 5 <br>
========== mot profile ==========
                                                   TensorRT layer name    Runtime, %  Invocations  Runtime, ms
                                                   image preprocessing         14.6%         1500      3709.73
                                                         JDE inference         55.4%         1500     14040.30
                                                        postprocessing          2.2%         1500       556.01
                                                    online association         27.8%         1500      7032.09
========== mot total runtime = 25338.1 ms ==========

* optimization 6 <br>
========== mot profile ==========
                                                   TensorRT layer name    Runtime, %  Invocations  Runtime, ms
                                                   image preprocessing         13.2%         1500      3560.65
                                                         JDE inference         56.1%         1500     15145.83
                                                            JDE decode          2.2%         1500       601.52
                                                    online association         28.5%         1500      7707.38
========== mot total runtime = 27015.4 ms ==========

cmake -DCMAKE_INSTALL_PREFIX="$(pwd)/install" \
    -DBUILD_SHARED_LIBS=OFF \
    ../opencv-4.5.0/

cmake -DCMAKE_INSTALL_PREFIX="$(pwd)/install" \
    -DCMAKE_CXX_FLAGS="-fPIC" \
    -DBUILD_SHARED_LIBS=OFF \
    ../jsoncpp-1.9.4/