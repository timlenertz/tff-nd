project(mf CXX)
cmake_minimum_required(VERSION 3.1)

include(GenerateExportHeader)

if(UNIX)
	add_compile_options(-std=c++14)
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lpthread")
	
	find_program(CCACHE_FOUND ccache)
	if(CCACHE_FOUND)
		set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
		set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
	endif()
elseif(WIN32)
	add_compile_options(/wd4018)
endif()

enable_testing()

include_directories(SYSTEM external/include)

file(GLOB_RECURSE TEST_SRC "test/*.cc" "src/*")
add_executable(tff_nd_test ${TEST_SRC})
target_compile_definitions(tff_nd_test PRIVATE TFF_ND_STANDALONE)
target_compile_options(tff_nd_test PRIVATE -Wno-unused-comparison)

add_compile_options(-Wall -Wextra -Wpedantic)
