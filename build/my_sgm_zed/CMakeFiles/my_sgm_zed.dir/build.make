# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wang/code/c++Code/my_sgm_zed_multithread_v2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build

# Include any dependencies generated for this target.
include my_sgm_zed/CMakeFiles/my_sgm_zed.dir/depend.make

# Include the progress variables for this target.
include my_sgm_zed/CMakeFiles/my_sgm_zed.dir/progress.make

# Include the compile flags for this target's objects.
include my_sgm_zed/CMakeFiles/my_sgm_zed.dir/flags.make

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_disparity.o: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/flags.make
my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_disparity.o: ../my_sgm_zed/src/get_disparity.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_disparity.o"
	cd /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/my_sgm_zed && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/my_sgm_zed.dir/src/get_disparity.o -c /home/wang/code/c++Code/my_sgm_zed_multithread_v2/my_sgm_zed/src/get_disparity.cpp

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_disparity.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_sgm_zed.dir/src/get_disparity.i"
	cd /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/my_sgm_zed && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wang/code/c++Code/my_sgm_zed_multithread_v2/my_sgm_zed/src/get_disparity.cpp > CMakeFiles/my_sgm_zed.dir/src/get_disparity.i

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_disparity.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_sgm_zed.dir/src/get_disparity.s"
	cd /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/my_sgm_zed && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wang/code/c++Code/my_sgm_zed_multithread_v2/my_sgm_zed/src/get_disparity.cpp -o CMakeFiles/my_sgm_zed.dir/src/get_disparity.s

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_disparity.o.requires:

.PHONY : my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_disparity.o.requires

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_disparity.o.provides: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_disparity.o.requires
	$(MAKE) -f my_sgm_zed/CMakeFiles/my_sgm_zed.dir/build.make my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_disparity.o.provides.build
.PHONY : my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_disparity.o.provides

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_disparity.o.provides.build: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_disparity.o


my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/flags.make
my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o: ../my_sgm_zed/src/get_point_cloud.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o"
	cd /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/my_sgm_zed && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o -c /home/wang/code/c++Code/my_sgm_zed_multithread_v2/my_sgm_zed/src/get_point_cloud.cpp

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.i"
	cd /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/my_sgm_zed && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wang/code/c++Code/my_sgm_zed_multithread_v2/my_sgm_zed/src/get_point_cloud.cpp > CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.i

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.s"
	cd /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/my_sgm_zed && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wang/code/c++Code/my_sgm_zed_multithread_v2/my_sgm_zed/src/get_point_cloud.cpp -o CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.s

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o.requires:

.PHONY : my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o.requires

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o.provides: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o.requires
	$(MAKE) -f my_sgm_zed/CMakeFiles/my_sgm_zed.dir/build.make my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o.provides.build
.PHONY : my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o.provides

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o.provides.build: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o


my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_roi.o: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/flags.make
my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_roi.o: ../my_sgm_zed/src/get_roi.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_roi.o"
	cd /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/my_sgm_zed && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/my_sgm_zed.dir/src/get_roi.o -c /home/wang/code/c++Code/my_sgm_zed_multithread_v2/my_sgm_zed/src/get_roi.cpp

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_roi.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_sgm_zed.dir/src/get_roi.i"
	cd /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/my_sgm_zed && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wang/code/c++Code/my_sgm_zed_multithread_v2/my_sgm_zed/src/get_roi.cpp > CMakeFiles/my_sgm_zed.dir/src/get_roi.i

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_roi.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_sgm_zed.dir/src/get_roi.s"
	cd /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/my_sgm_zed && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wang/code/c++Code/my_sgm_zed_multithread_v2/my_sgm_zed/src/get_roi.cpp -o CMakeFiles/my_sgm_zed.dir/src/get_roi.s

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_roi.o.requires:

.PHONY : my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_roi.o.requires

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_roi.o.provides: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_roi.o.requires
	$(MAKE) -f my_sgm_zed/CMakeFiles/my_sgm_zed.dir/build.make my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_roi.o.provides.build
.PHONY : my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_roi.o.provides

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_roi.o.provides.build: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_roi.o


my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/little_tips.o: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/flags.make
my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/little_tips.o: ../my_sgm_zed/src/little_tips.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/little_tips.o"
	cd /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/my_sgm_zed && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/my_sgm_zed.dir/src/little_tips.o -c /home/wang/code/c++Code/my_sgm_zed_multithread_v2/my_sgm_zed/src/little_tips.cpp

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/little_tips.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_sgm_zed.dir/src/little_tips.i"
	cd /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/my_sgm_zed && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wang/code/c++Code/my_sgm_zed_multithread_v2/my_sgm_zed/src/little_tips.cpp > CMakeFiles/my_sgm_zed.dir/src/little_tips.i

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/little_tips.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_sgm_zed.dir/src/little_tips.s"
	cd /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/my_sgm_zed && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wang/code/c++Code/my_sgm_zed_multithread_v2/my_sgm_zed/src/little_tips.cpp -o CMakeFiles/my_sgm_zed.dir/src/little_tips.s

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/little_tips.o.requires:

.PHONY : my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/little_tips.o.requires

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/little_tips.o.provides: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/little_tips.o.requires
	$(MAKE) -f my_sgm_zed/CMakeFiles/my_sgm_zed.dir/build.make my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/little_tips.o.provides.build
.PHONY : my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/little_tips.o.provides

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/little_tips.o.provides.build: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/little_tips.o


my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/main.o: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/flags.make
my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/main.o: ../my_sgm_zed/src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/main.o"
	cd /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/my_sgm_zed && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/my_sgm_zed.dir/src/main.o -c /home/wang/code/c++Code/my_sgm_zed_multithread_v2/my_sgm_zed/src/main.cpp

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/main.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_sgm_zed.dir/src/main.i"
	cd /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/my_sgm_zed && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wang/code/c++Code/my_sgm_zed_multithread_v2/my_sgm_zed/src/main.cpp > CMakeFiles/my_sgm_zed.dir/src/main.i

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/main.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_sgm_zed.dir/src/main.s"
	cd /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/my_sgm_zed && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wang/code/c++Code/my_sgm_zed_multithread_v2/my_sgm_zed/src/main.cpp -o CMakeFiles/my_sgm_zed.dir/src/main.s

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/main.o.requires:

.PHONY : my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/main.o.requires

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/main.o.provides: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/main.o.requires
	$(MAKE) -f my_sgm_zed/CMakeFiles/my_sgm_zed.dir/build.make my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/main.o.provides.build
.PHONY : my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/main.o.provides

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/main.o.provides.build: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/main.o


my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/my_camera.o: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/flags.make
my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/my_camera.o: ../my_sgm_zed/src/my_camera.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/my_camera.o"
	cd /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/my_sgm_zed && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/my_sgm_zed.dir/src/my_camera.o -c /home/wang/code/c++Code/my_sgm_zed_multithread_v2/my_sgm_zed/src/my_camera.cpp

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/my_camera.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_sgm_zed.dir/src/my_camera.i"
	cd /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/my_sgm_zed && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wang/code/c++Code/my_sgm_zed_multithread_v2/my_sgm_zed/src/my_camera.cpp > CMakeFiles/my_sgm_zed.dir/src/my_camera.i

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/my_camera.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_sgm_zed.dir/src/my_camera.s"
	cd /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/my_sgm_zed && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wang/code/c++Code/my_sgm_zed_multithread_v2/my_sgm_zed/src/my_camera.cpp -o CMakeFiles/my_sgm_zed.dir/src/my_camera.s

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/my_camera.o.requires:

.PHONY : my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/my_camera.o.requires

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/my_camera.o.provides: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/my_camera.o.requires
	$(MAKE) -f my_sgm_zed/CMakeFiles/my_sgm_zed.dir/build.make my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/my_camera.o.provides.build
.PHONY : my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/my_camera.o.provides

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/my_camera.o.provides.build: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/my_camera.o


# Object files for target my_sgm_zed
my_sgm_zed_OBJECTS = \
"CMakeFiles/my_sgm_zed.dir/src/get_disparity.o" \
"CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o" \
"CMakeFiles/my_sgm_zed.dir/src/get_roi.o" \
"CMakeFiles/my_sgm_zed.dir/src/little_tips.o" \
"CMakeFiles/my_sgm_zed.dir/src/main.o" \
"CMakeFiles/my_sgm_zed.dir/src/my_camera.o"

# External object files for target my_sgm_zed
my_sgm_zed_EXTERNAL_OBJECTS =

my_sgm_zed/my_sgm_zed: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_disparity.o
my_sgm_zed/my_sgm_zed: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o
my_sgm_zed/my_sgm_zed: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_roi.o
my_sgm_zed/my_sgm_zed: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/little_tips.o
my_sgm_zed/my_sgm_zed: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/main.o
my_sgm_zed/my_sgm_zed: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/my_camera.o
my_sgm_zed/my_sgm_zed: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/build.make
my_sgm_zed/my_sgm_zed: /usr/local/cuda-11.1/lib64/libcudart_static.a
my_sgm_zed/my_sgm_zed: /usr/lib/x86_64-linux-gnu/librt.so
my_sgm_zed/my_sgm_zed: src/libsgm.a
my_sgm_zed/my_sgm_zed: /usr/local/zed/lib/libsl_zed.so
my_sgm_zed/my_sgm_zed: /usr/lib/x86_64-linux-gnu/libopenblas.so
my_sgm_zed/my_sgm_zed: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
my_sgm_zed/my_sgm_zed: /usr/lib/x86_64-linux-gnu/libcuda.so
my_sgm_zed/my_sgm_zed: /usr/local/cuda-11.1/lib64/libcudart.so
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_gapi.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_stitching.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_aruco.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_bgsegm.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_bioinspired.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_ccalib.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_dnn_objdetect.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_dnn_superres.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_dpm.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_face.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_freetype.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_fuzzy.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_hfs.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_img_hash.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_line_descriptor.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_quality.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_reg.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_rgbd.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_saliency.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_stereo.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_structured_light.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_superres.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_surface_matching.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_tracking.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_videostab.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_xfeatures2d.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_xobjdetect.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_xphoto.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/cuda-11.1/lib64/libcudart_static.a
my_sgm_zed/my_sgm_zed: /usr/lib/x86_64-linux-gnu/librt.so
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_shape.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_highgui.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_datasets.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_plot.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_text.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_dnn.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_ml.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_phase_unwrapping.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_optflow.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_ximgproc.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_video.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_videoio.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_imgcodecs.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_objdetect.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_calib3d.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_features2d.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_flann.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_photo.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_imgproc.so.4.2.0
my_sgm_zed/my_sgm_zed: /usr/local/lib/libopencv_core.so.4.2.0
my_sgm_zed/my_sgm_zed: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable my_sgm_zed"
	cd /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/my_sgm_zed && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/my_sgm_zed.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
my_sgm_zed/CMakeFiles/my_sgm_zed.dir/build: my_sgm_zed/my_sgm_zed

.PHONY : my_sgm_zed/CMakeFiles/my_sgm_zed.dir/build

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/requires: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_disparity.o.requires
my_sgm_zed/CMakeFiles/my_sgm_zed.dir/requires: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_point_cloud.o.requires
my_sgm_zed/CMakeFiles/my_sgm_zed.dir/requires: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/get_roi.o.requires
my_sgm_zed/CMakeFiles/my_sgm_zed.dir/requires: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/little_tips.o.requires
my_sgm_zed/CMakeFiles/my_sgm_zed.dir/requires: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/main.o.requires
my_sgm_zed/CMakeFiles/my_sgm_zed.dir/requires: my_sgm_zed/CMakeFiles/my_sgm_zed.dir/src/my_camera.o.requires

.PHONY : my_sgm_zed/CMakeFiles/my_sgm_zed.dir/requires

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/clean:
	cd /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/my_sgm_zed && $(CMAKE_COMMAND) -P CMakeFiles/my_sgm_zed.dir/cmake_clean.cmake
.PHONY : my_sgm_zed/CMakeFiles/my_sgm_zed.dir/clean

my_sgm_zed/CMakeFiles/my_sgm_zed.dir/depend:
	cd /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wang/code/c++Code/my_sgm_zed_multithread_v2 /home/wang/code/c++Code/my_sgm_zed_multithread_v2/my_sgm_zed /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/my_sgm_zed /home/wang/code/c++Code/my_sgm_zed_multithread_v2/build/my_sgm_zed/CMakeFiles/my_sgm_zed.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : my_sgm_zed/CMakeFiles/my_sgm_zed.dir/depend

