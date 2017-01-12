#ifndef TFF_ND_RAW_ALLOCATOR_H_
#define TFF_ND_RAW_ALLOCATOR_H_

#include "../config.h"
#include <cstdlib>

namespace tff {
	
#if TFF_ND_WITH_ALLOCATION

class raw_allocator {
public:
	static std::size_t size_granularity() { return 1; }

	void* raw_allocate(std::size_t size, std::size_t alignment = 1) {
		#if __cplusplus > 201500
		return std::aligned_alloc(alignment, size);
		#else
		std::ptrdiff_t offset = alignment - 1 + sizeof(void*);
		void* p1 = std::malloc(size + offset);
		std::uintptr_t p1_int = reinterpret_cast<std::uintptr_t>(p1);
		std::uintptr_t p2_int = (p1_int + offset) & ~(alignment - 1);
		void** p2 = reinterpret_cast<void**>(p2_int);
		p2[-1] = p1;
		return static_cast<void*>(p2);
		#endif
	}
	
	void raw_deallocate(void* ptr, std::size_t) {
		#if __cplusplus > 201500
		std::free(ptr);
		#else
		void** p2 = reinterpret_cast<void**>(ptr);
		void* p1 = p2[-1];
		std::free(p1);
		#endif
	}
};


template<typename T>
constexpr bool is_raw_allocator = false;

template<>
constexpr bool is_raw_allocator<raw_allocator> = true;

#endif

}

#endif
