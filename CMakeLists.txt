project(mf CXX)
cmake_minimum_required(VERSION 3.1)

include(GenerateExportHeader)

if(UNIX)
	add_compile_options(-std=c++14)
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lpthread")
	
	find_program(CCACHE_FOUND ccache)
	if(CCACHE_FOUND)
		#set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
		#set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
	endif()
elseif(WIN32)
	add_compile_options(/wd4018)
endif()

enable_testing()

include_directories(SYSTEM external/include)

file(GLOB_RECURSE TEST_SRC "test/*.cc" "src/*")
add_executable(tff_nd_test ${TEST_SRC})
target_compile_definitions(tff_nd_test PRIVATE TFF_ND_STANDALONE)

file(GLOB EXAMPLE_SRC "examples/*.cc")
file(GLOB_RECURSE EXAMPLE_SUPPORT_SRC "examples/support/*")

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})

foreach(ex_cpp ${EXAMPLE_SRC})
	get_filename_component(ex_nm ${ex_cpp} NAME_WE)
	message(${ex_nm})
	add_executable(${ex_nm} ${ex_cpp} ${EXAMPLE_SUPPORT_SRC})
	target_compile_definitions(${ex_nm} PRIVATE TFF_ND_STANDALONE)
	target_link_libraries(${ex_nm} ${PNG_LIBRARY})
endforeach()


add_compile_options(-Wno-unused-comparison -Wall -Wextra -Wpedantic)
