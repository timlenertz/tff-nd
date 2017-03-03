#ifndef TFF_NDARRAY_OPAQUE_FORMAT_NDARRAY_H_
#define TFF_NDARRAY_OPAQUE_FORMAT_NDARRAY_H_

#include "../config.h"
#if TFF_ND_WITH_OPAQUE

#include "../pod_array_format.h"
#include "../ndcoord_dyn.h"
#include <type_traits>

namespace tff {

template<bool Mutable>
class opaque_ndarray_frame_handle;


class opaque_ndarray_format {
private:
	std::size_t elem_size_ = 0;
	std::size_t elem_alignment_requirement_ = 0;
	std::size_t elem_stride_ = 0;
	bool pod_ = false;
	ndsize_dyn shape_;

public:
	using frame_handle_type = opaque_ndarray_frame_handle<true>;
	using const_frame_handle_type = opaque_ndarray_frame_handle<false>;
	
	using frame_pointer_type = void*;
	using const_frame_pointer_type = const void*;
	
	opaque_ndarray_format() = default;
	explicit opaque_ndarray_format(const pod_array_format&);
	opaque_ndarray_format(const pod_array_format&, const ndsize_dyn&);
	opaque_ndarray_format(std::size_t elem_sz, std::size_t align, std::size_t stride, bool pod, const ndsize_dyn& shp);
	
	std::size_t size() const { return shape_.product() * elem_stride_; }
	std::size_t alignment_requirement() const { return elem_alignment_requirement_; }

	bool is_pod() const { return pod_; }
	pod_array_format pod_format() const;
	
	bool is_ndarray() const { return true; }
	std::size_t dimension() const { return shape_.size(); }
	const ndsize_dyn& shape() const { return shape_; }
	std::size_t elem_size() const { return elem_size_; }
	std::size_t elem_alignment_requirement() const { return elem_alignment_requirement_; }
	std::size_t elem_stride() const { return elem_stride_; }

	friend bool operator==(const opaque_ndarray_format& a, const opaque_ndarray_format& b);
	friend bool operator!=(const opaque_ndarray_format& a, const opaque_ndarray_format& b);
};


template<typename Elem>
opaque_ndarray_format default_opaque_ndarray_format(const ndsize_dyn& shp) {
	return opaque_ndarray_format(
		sizeof(Elem),
		alignof(Elem),
		sizeof(Elem),
		std::is_pod<Elem>::value,
		shp
	);
}


template<bool Mutable>
class opaque_ndarray_frame_handle {
public:
	using frame_format_type = opaque_ndarray_format;
	using frame_pointer_type = std::conditional_t<Mutable, void*, const void*>;

private:
	frame_pointer_type ptr_;
	opaque_ndarray_format frame_format_;
	
public:
	opaque_ndarray_frame_handle(frame_pointer_type ptr, const opaque_ndarray_format& frm) :
		ptr_(ptr),
		frame_format_(frm) { }
		
	opaque_ndarray_frame_handle(const opaque_ndarray_frame_handle<true>& hd) :
		ptr_(hd.ptr()),
		frame_format_(hd.frame_format()) { }

	frame_pointer_type ptr() const { return ptr_; }
	const opaque_ndarray_format& frame_format() const { return frame_format_; }
	
	void assign(const opaque_ndarray_frame_handle<false>& vw) {
		Assert(frame_format_.is_pod());
		Assert(frame_format() == vw.frame_format());
		if(ptr() != vw.ptr()) pod_array_copy(ptr(), vw.ptr(), frame_format_.pod_format());
	}
	
	bool compare(const opaque_ndarray_frame_handle<false>& vw) const {
		Assert(frame_format_.is_pod());
		Assert(frame_format() == vw.frame_format());
		if(ptr() != vw.ptr()) return pod_array_compare(ptr(), vw.ptr(), frame_format_.pod_format());
		else return true;
	}
		
	void construct() const { }
	void destruct() const { }
	void initialize() const { }
};


}

#include "ndarray.icc"

#endif
#endif

