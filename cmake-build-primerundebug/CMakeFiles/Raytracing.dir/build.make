# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/steinraf/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/221.5080.224/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/steinraf/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/221.5080.224/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/steinraf/ETH/CG/CustomRenderer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/steinraf/ETH/CG/CustomRenderer/cmake-build-primerundebug

# Include any dependencies generated for this target.
include CMakeFiles/Raytracing.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Raytracing.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Raytracing.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Raytracing.dir/flags.make

CMakeFiles/Raytracing.dir/main.cu.o: CMakeFiles/Raytracing.dir/flags.make
CMakeFiles/Raytracing.dir/main.cu.o: ../main.cu
CMakeFiles/Raytracing.dir/main.cu.o: CMakeFiles/Raytracing.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/steinraf/ETH/CG/CustomRenderer/cmake-build-primerundebug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/Raytracing.dir/main.cu.o"
	/opt/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/Raytracing.dir/main.cu.o -MF CMakeFiles/Raytracing.dir/main.cu.o.d -x cu -dc /home/steinraf/ETH/CG/CustomRenderer/main.cu -o CMakeFiles/Raytracing.dir/main.cu.o

CMakeFiles/Raytracing.dir/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/Raytracing.dir/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/Raytracing.dir/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/Raytracing.dir/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/Raytracing.dir/src/cudaHelpers.cu.o: CMakeFiles/Raytracing.dir/flags.make
CMakeFiles/Raytracing.dir/src/cudaHelpers.cu.o: ../src/cudaHelpers.cu
CMakeFiles/Raytracing.dir/src/cudaHelpers.cu.o: CMakeFiles/Raytracing.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/steinraf/ETH/CG/CustomRenderer/cmake-build-primerundebug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/Raytracing.dir/src/cudaHelpers.cu.o"
	/opt/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/Raytracing.dir/src/cudaHelpers.cu.o -MF CMakeFiles/Raytracing.dir/src/cudaHelpers.cu.o.d -x cu -dc /home/steinraf/ETH/CG/CustomRenderer/src/cudaHelpers.cu -o CMakeFiles/Raytracing.dir/src/cudaHelpers.cu.o

CMakeFiles/Raytracing.dir/src/cudaHelpers.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/Raytracing.dir/src/cudaHelpers.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/Raytracing.dir/src/cudaHelpers.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/Raytracing.dir/src/cudaHelpers.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/Raytracing.dir/src/scene/scene.cu.o: CMakeFiles/Raytracing.dir/flags.make
CMakeFiles/Raytracing.dir/src/scene/scene.cu.o: ../src/scene/scene.cu
CMakeFiles/Raytracing.dir/src/scene/scene.cu.o: CMakeFiles/Raytracing.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/steinraf/ETH/CG/CustomRenderer/cmake-build-primerundebug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object CMakeFiles/Raytracing.dir/src/scene/scene.cu.o"
	/opt/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/Raytracing.dir/src/scene/scene.cu.o -MF CMakeFiles/Raytracing.dir/src/scene/scene.cu.o.d -x cu -dc /home/steinraf/ETH/CG/CustomRenderer/src/scene/scene.cu -o CMakeFiles/Raytracing.dir/src/scene/scene.cu.o

CMakeFiles/Raytracing.dir/src/scene/scene.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/Raytracing.dir/src/scene/scene.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/Raytracing.dir/src/scene/scene.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/Raytracing.dir/src/scene/scene.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/Raytracing.dir/src/hittable.cu.o: CMakeFiles/Raytracing.dir/flags.make
CMakeFiles/Raytracing.dir/src/hittable.cu.o: ../src/hittable.cu
CMakeFiles/Raytracing.dir/src/hittable.cu.o: CMakeFiles/Raytracing.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/steinraf/ETH/CG/CustomRenderer/cmake-build-primerundebug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CUDA object CMakeFiles/Raytracing.dir/src/hittable.cu.o"
	/opt/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/Raytracing.dir/src/hittable.cu.o -MF CMakeFiles/Raytracing.dir/src/hittable.cu.o.d -x cu -dc /home/steinraf/ETH/CG/CustomRenderer/src/hittable.cu -o CMakeFiles/Raytracing.dir/src/hittable.cu.o

CMakeFiles/Raytracing.dir/src/hittable.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/Raytracing.dir/src/hittable.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/Raytracing.dir/src/hittable.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/Raytracing.dir/src/hittable.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/Raytracing.dir/src/camera.cu.o: CMakeFiles/Raytracing.dir/flags.make
CMakeFiles/Raytracing.dir/src/camera.cu.o: ../src/camera.cu
CMakeFiles/Raytracing.dir/src/camera.cu.o: CMakeFiles/Raytracing.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/steinraf/ETH/CG/CustomRenderer/cmake-build-primerundebug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CUDA object CMakeFiles/Raytracing.dir/src/camera.cu.o"
	/opt/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/Raytracing.dir/src/camera.cu.o -MF CMakeFiles/Raytracing.dir/src/camera.cu.o.d -x cu -dc /home/steinraf/ETH/CG/CustomRenderer/src/camera.cu -o CMakeFiles/Raytracing.dir/src/camera.cu.o

CMakeFiles/Raytracing.dir/src/camera.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/Raytracing.dir/src/camera.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/Raytracing.dir/src/camera.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/Raytracing.dir/src/camera.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/Raytracing.dir/src/utility/meshLoader.cu.o: CMakeFiles/Raytracing.dir/flags.make
CMakeFiles/Raytracing.dir/src/utility/meshLoader.cu.o: ../src/utility/meshLoader.cu
CMakeFiles/Raytracing.dir/src/utility/meshLoader.cu.o: CMakeFiles/Raytracing.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/steinraf/ETH/CG/CustomRenderer/cmake-build-primerundebug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CUDA object CMakeFiles/Raytracing.dir/src/utility/meshLoader.cu.o"
	/opt/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/Raytracing.dir/src/utility/meshLoader.cu.o -MF CMakeFiles/Raytracing.dir/src/utility/meshLoader.cu.o.d -x cu -dc /home/steinraf/ETH/CG/CustomRenderer/src/utility/meshLoader.cu -o CMakeFiles/Raytracing.dir/src/utility/meshLoader.cu.o

CMakeFiles/Raytracing.dir/src/utility/meshLoader.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/Raytracing.dir/src/utility/meshLoader.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/Raytracing.dir/src/utility/meshLoader.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/Raytracing.dir/src/utility/meshLoader.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/Raytracing.dir/src/bsdf.cu.o: CMakeFiles/Raytracing.dir/flags.make
CMakeFiles/Raytracing.dir/src/bsdf.cu.o: ../src/bsdf.cu
CMakeFiles/Raytracing.dir/src/bsdf.cu.o: CMakeFiles/Raytracing.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/steinraf/ETH/CG/CustomRenderer/cmake-build-primerundebug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CUDA object CMakeFiles/Raytracing.dir/src/bsdf.cu.o"
	/opt/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/Raytracing.dir/src/bsdf.cu.o -MF CMakeFiles/Raytracing.dir/src/bsdf.cu.o.d -x cu -dc /home/steinraf/ETH/CG/CustomRenderer/src/bsdf.cu -o CMakeFiles/Raytracing.dir/src/bsdf.cu.o

CMakeFiles/Raytracing.dir/src/bsdf.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/Raytracing.dir/src/bsdf.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/Raytracing.dir/src/bsdf.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/Raytracing.dir/src/bsdf.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/Raytracing.dir/src/utility/vector.cu.o: CMakeFiles/Raytracing.dir/flags.make
CMakeFiles/Raytracing.dir/src/utility/vector.cu.o: ../src/utility/vector.cu
CMakeFiles/Raytracing.dir/src/utility/vector.cu.o: CMakeFiles/Raytracing.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/steinraf/ETH/CG/CustomRenderer/cmake-build-primerundebug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CUDA object CMakeFiles/Raytracing.dir/src/utility/vector.cu.o"
	/opt/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/Raytracing.dir/src/utility/vector.cu.o -MF CMakeFiles/Raytracing.dir/src/utility/vector.cu.o.d -x cu -dc /home/steinraf/ETH/CG/CustomRenderer/src/utility/vector.cu -o CMakeFiles/Raytracing.dir/src/utility/vector.cu.o

CMakeFiles/Raytracing.dir/src/utility/vector.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/Raytracing.dir/src/utility/vector.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/Raytracing.dir/src/utility/vector.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/Raytracing.dir/src/utility/vector.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/Raytracing.dir/src/scene/sceneLoader.cu.o: CMakeFiles/Raytracing.dir/flags.make
CMakeFiles/Raytracing.dir/src/scene/sceneLoader.cu.o: ../src/scene/sceneLoader.cu
CMakeFiles/Raytracing.dir/src/scene/sceneLoader.cu.o: CMakeFiles/Raytracing.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/steinraf/ETH/CG/CustomRenderer/cmake-build-primerundebug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CUDA object CMakeFiles/Raytracing.dir/src/scene/sceneLoader.cu.o"
	/opt/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/Raytracing.dir/src/scene/sceneLoader.cu.o -MF CMakeFiles/Raytracing.dir/src/scene/sceneLoader.cu.o.d -x cu -dc /home/steinraf/ETH/CG/CustomRenderer/src/scene/sceneLoader.cu -o CMakeFiles/Raytracing.dir/src/scene/sceneLoader.cu.o

CMakeFiles/Raytracing.dir/src/scene/sceneLoader.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/Raytracing.dir/src/scene/sceneLoader.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/Raytracing.dir/src/scene/sceneLoader.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/Raytracing.dir/src/scene/sceneLoader.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target Raytracing
Raytracing_OBJECTS = \
"CMakeFiles/Raytracing.dir/main.cu.o" \
"CMakeFiles/Raytracing.dir/src/cudaHelpers.cu.o" \
"CMakeFiles/Raytracing.dir/src/scene/scene.cu.o" \
"CMakeFiles/Raytracing.dir/src/hittable.cu.o" \
"CMakeFiles/Raytracing.dir/src/camera.cu.o" \
"CMakeFiles/Raytracing.dir/src/utility/meshLoader.cu.o" \
"CMakeFiles/Raytracing.dir/src/bsdf.cu.o" \
"CMakeFiles/Raytracing.dir/src/utility/vector.cu.o" \
"CMakeFiles/Raytracing.dir/src/scene/sceneLoader.cu.o"

# External object files for target Raytracing
Raytracing_EXTERNAL_OBJECTS =

CMakeFiles/Raytracing.dir/cmake_device_link.o: CMakeFiles/Raytracing.dir/main.cu.o
CMakeFiles/Raytracing.dir/cmake_device_link.o: CMakeFiles/Raytracing.dir/src/cudaHelpers.cu.o
CMakeFiles/Raytracing.dir/cmake_device_link.o: CMakeFiles/Raytracing.dir/src/scene/scene.cu.o
CMakeFiles/Raytracing.dir/cmake_device_link.o: CMakeFiles/Raytracing.dir/src/hittable.cu.o
CMakeFiles/Raytracing.dir/cmake_device_link.o: CMakeFiles/Raytracing.dir/src/camera.cu.o
CMakeFiles/Raytracing.dir/cmake_device_link.o: CMakeFiles/Raytracing.dir/src/utility/meshLoader.cu.o
CMakeFiles/Raytracing.dir/cmake_device_link.o: CMakeFiles/Raytracing.dir/src/bsdf.cu.o
CMakeFiles/Raytracing.dir/cmake_device_link.o: CMakeFiles/Raytracing.dir/src/utility/vector.cu.o
CMakeFiles/Raytracing.dir/cmake_device_link.o: CMakeFiles/Raytracing.dir/src/scene/sceneLoader.cu.o
CMakeFiles/Raytracing.dir/cmake_device_link.o: CMakeFiles/Raytracing.dir/build.make
CMakeFiles/Raytracing.dir/cmake_device_link.o: /usr/local/lib/libPNGwriter.a
CMakeFiles/Raytracing.dir/cmake_device_link.o: libGLAD.a
CMakeFiles/Raytracing.dir/cmake_device_link.o: /usr/lib/libpng.so
CMakeFiles/Raytracing.dir/cmake_device_link.o: /usr/lib/libz.so
CMakeFiles/Raytracing.dir/cmake_device_link.o: /usr/lib/libfreetype.so
CMakeFiles/Raytracing.dir/cmake_device_link.o: CMakeFiles/Raytracing.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/steinraf/ETH/CG/CustomRenderer/cmake-build-primerundebug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CUDA device code CMakeFiles/Raytracing.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Raytracing.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Raytracing.dir/build: CMakeFiles/Raytracing.dir/cmake_device_link.o
.PHONY : CMakeFiles/Raytracing.dir/build

# Object files for target Raytracing
Raytracing_OBJECTS = \
"CMakeFiles/Raytracing.dir/main.cu.o" \
"CMakeFiles/Raytracing.dir/src/cudaHelpers.cu.o" \
"CMakeFiles/Raytracing.dir/src/scene/scene.cu.o" \
"CMakeFiles/Raytracing.dir/src/hittable.cu.o" \
"CMakeFiles/Raytracing.dir/src/camera.cu.o" \
"CMakeFiles/Raytracing.dir/src/utility/meshLoader.cu.o" \
"CMakeFiles/Raytracing.dir/src/bsdf.cu.o" \
"CMakeFiles/Raytracing.dir/src/utility/vector.cu.o" \
"CMakeFiles/Raytracing.dir/src/scene/sceneLoader.cu.o"

# External object files for target Raytracing
Raytracing_EXTERNAL_OBJECTS =

Raytracing: CMakeFiles/Raytracing.dir/main.cu.o
Raytracing: CMakeFiles/Raytracing.dir/src/cudaHelpers.cu.o
Raytracing: CMakeFiles/Raytracing.dir/src/scene/scene.cu.o
Raytracing: CMakeFiles/Raytracing.dir/src/hittable.cu.o
Raytracing: CMakeFiles/Raytracing.dir/src/camera.cu.o
Raytracing: CMakeFiles/Raytracing.dir/src/utility/meshLoader.cu.o
Raytracing: CMakeFiles/Raytracing.dir/src/bsdf.cu.o
Raytracing: CMakeFiles/Raytracing.dir/src/utility/vector.cu.o
Raytracing: CMakeFiles/Raytracing.dir/src/scene/sceneLoader.cu.o
Raytracing: CMakeFiles/Raytracing.dir/build.make
Raytracing: /usr/local/lib/libPNGwriter.a
Raytracing: libGLAD.a
Raytracing: /usr/lib/libpng.so
Raytracing: /usr/lib/libz.so
Raytracing: /usr/lib/libfreetype.so
Raytracing: CMakeFiles/Raytracing.dir/cmake_device_link.o
Raytracing: CMakeFiles/Raytracing.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/steinraf/ETH/CG/CustomRenderer/cmake-build-primerundebug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Linking CXX executable Raytracing"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Raytracing.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Raytracing.dir/build: Raytracing
.PHONY : CMakeFiles/Raytracing.dir/build

CMakeFiles/Raytracing.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Raytracing.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Raytracing.dir/clean

CMakeFiles/Raytracing.dir/depend:
	cd /home/steinraf/ETH/CG/CustomRenderer/cmake-build-primerundebug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/steinraf/ETH/CG/CustomRenderer /home/steinraf/ETH/CG/CustomRenderer /home/steinraf/ETH/CG/CustomRenderer/cmake-build-primerundebug /home/steinraf/ETH/CG/CustomRenderer/cmake-build-primerundebug /home/steinraf/ETH/CG/CustomRenderer/cmake-build-primerundebug/CMakeFiles/Raytracing.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Raytracing.dir/depend

