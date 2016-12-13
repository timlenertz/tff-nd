#ifndef TFF_NDARRAY_POD_ARRAY_FORMAT_H_
#define TFF_NDARRAY_POD_ARRAY_FORMAT_H_

#include <cstdlib>
#include <type_traits>
#include "common.h"

namespace tff {

class pod_array_format {
private:		
	std::size_t elem_size_;
	std::size_t elem_alignment_;
	std::size_t length_;
	std::size_t stride_;
	
public:
	pod_array_format(std::size_t elem_size, std::size_t elem_align, std::size_t length, std::size_t stride) :
		elem_size_(elem_size), elem_alignment_(elem_align), length_(length), stride_(stride)
	{		
		Assert(is_nonzero_multiple_of(stride, elem_align));
		Assert(stride >= elem_size);
	}
	
	pod_array_format(const pod_array_format&) = default;
	pod_array_format& operator=(const pod_array_format&) = default;

	std::size_t size() const { return length_ * stride_; }

	std::size_t length() const { return length_; }
	std::size_t stride() const { return stride_; } // TODO rename elem_stride like opaque_ndarray_format ?
	
	std::size_t elem_size() const { return elem_size_; }
	std::size_t elem_alignment() const { return elem_alignment_; }

	std::size_t elem_padding() const { return stride() - elem_size(); }
	
	bool is_contiguous() const { return (elem_padding() == 0); }
};


/// Compare two data stored in \a a and \a b, both having format \a frame_format.
bool pod_array_compare(const void* a, const void* b, const pod_array_format&);

/// Copy data at \a origin having format \a frame_format to \a destination.
void pod_array_copy(void* destination, const void* origin, const pod_array_format&);


bool operator==(const pod_array_format&, const pod_array_format&);
bool operator!=(const pod_array_format&, const pod_array_format&);


bool same_coverage(const pod_array_format&, const pod_array_format&);


template<typename Elem>
pod_array_format make_pod_array_format(std::size_t length, std::size_t stride = sizeof(Elem), std::size_t align = alignof(Elem)) {
	Assert(std::is_pod<Elem>::value, "Elem must be POD type for pod_format"); // TODO static_assert (need constexpr if)
	return pod_array_format(sizeof(Elem), align, length, stride);
}

inline pod_array_format make_pod_array_format(std::size_t length, std::size_t alignment = 1) {
	std::size_t stride = round_up(length, alignment);
	return pod_array_format(length, alignment, 1, stride);
}

}

#include "pod_array_format.icc"

#endif

