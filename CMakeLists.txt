project(tlz-nd CXX)
cmake_minimum_required(VERSION 3.1)

if(UNIX)
	add_compile_options(-Wall -Wextra -Wno-sign-compare -std=c++14)
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lpthread")
elseif(WIN32)
	add_compile_options(/wd4018)
endif()

enable_testing()

include_directories(SYSTEM external/include)

# Test
file(GLOB_RECURSE TEST_SRC "test/*.cc" "src/*")
add_executable(nd_test ${TEST_SRC})
target_compile_definitions(nd_test PRIVATE TLZ_ND_STANDALONE)
add_test(COMMAND nd_test)

# Examples
find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})

file(GLOB EXAMPLE_SRC "examples/*.cc")
file(GLOB_RECURSE EXAMPLE_SUPPORT_SRC "examples/support/*")

foreach(ex_cpp ${EXAMPLE_SRC})
	get_filename_component(ex_nm ${ex_cpp} NAME_WE)
	message(${ex_nm})
	add_executable(${ex_nm} ${ex_cpp} ${EXAMPLE_SUPPORT_SRC})
	target_compile_definitions(${ex_nm} PRIVATE TLZ_ND_STANDALONE)
	target_link_libraries(${ex_nm} ${PNG_LIBRARY})
endforeach()


