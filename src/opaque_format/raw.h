#ifndef TLZ_NDARRAY_OPAQUE_FORMAT_RAW_H_
#define TLZ_NDARRAY_OPAQUE_FORMAT_RAW_H_

#include "../config.h"
#if TLZ_ND_WITH_OPAQUE

#include <cstring>
#include "../ndcoord_dyn.h"

namespace tlz {
	
template<bool Mutable> class opaque_raw_frame_handle;

class opaque_raw_format {
private:
	std::size_t frame_size_;
	std::size_t frame_alignment_requirement_;

public:
	using frame_handle_type = opaque_raw_frame_handle<true>;
	using const_frame_handle_type = opaque_raw_frame_handle<false>;
	
	using frame_pointer_type = void*;
	using const_frame_pointer_type = const void*;
		
	opaque_raw_format() :
		frame_size_(0),  frame_alignment_requirement_(1) { }
	explicit opaque_raw_format(std::size_t size, std::size_t align = 1) :
		frame_size_(size), frame_alignment_requirement_(align) { }

	std::size_t size() const { return round_up(frame_size_, frame_alignment_requirement_); }
	std::size_t content_size() const { return frame_size_; }
	std::size_t alignment_requirement() const { return frame_alignment_requirement_; }
	
	bool is_pod() const { return true; }
	pod_array_format pod_format() const { return make_pod_array_format(frame_size_, frame_alignment_requirement_); }

	bool is_ndarray() const { return false; }
	std::size_t dimension() const { return 0; }
	ndsize_dyn shape() const { return make_ndsize_dyn(); }
	std::size_t elem_size() const { return 1; }
	std::size_t elem_alignment_requirement() const { return frame_alignment_requirement_; }
	std::size_t elem_stride() const { return 1; }

	friend bool operator==(const opaque_raw_format& a, const opaque_raw_format& b)
		{ return (a.frame_size_ == b.frame_size_) && (a.frame_alignment_requirement_ == b.frame_alignment_requirement_); }
	friend bool operator!=(const opaque_raw_format& a, const opaque_raw_format& b)
		{ return (a.frame_size_ != b.frame_size_) || (a.frame_alignment_requirement_ != b.frame_alignment_requirement_); }
};


template<bool Mutable>
class opaque_raw_frame_handle {	
public:
	using frame_format_type = opaque_raw_format;
	using frame_pointer_type = std::conditional_t<Mutable, void*, const void*>;

private:
	frame_pointer_type ptr_;
	std::size_t content_size_;

public:
	opaque_raw_frame_handle(frame_pointer_type ptr, const frame_format_type& frm) :
		ptr_(ptr),
		content_size_(frm.content_size()) { }
	opaque_raw_frame_handle(const opaque_raw_frame_handle<true>& hd) :
		ptr_(hd.ptr()),
		content_size_(hd.content_size()) { }

	frame_pointer_type ptr() const { return ptr_; }
	std::size_t content_size() const { return content_size_; }

	void assign(const opaque_raw_frame_handle<false>& hd) {
		Assert(content_size() == hd.content_size());
		if(ptr() != hd.ptr()) std::memcpy(ptr(), hd.ptr(), content_size());
	}
	
	bool compare(const opaque_raw_frame_handle<false>& hd) const {		
		Assert(content_size() == hd.content_size());
		if(ptr() != hd.ptr()) return (std::memcmp(ptr(), hd.ptr(), content_size()) == 0);
		else return true;
	}
	
	void construct() const { }
	void destruct() const { }
	void initialize() const { }
};

}

#endif
#endif
