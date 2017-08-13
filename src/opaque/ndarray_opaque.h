#ifndef TLZ_NDARRAY_OPAQUE_H_
#define TLZ_NDARRAY_OPAQUE_H_

#include "../config.h"
#if TLZ_ND_WITH_ALLOCATION && TLZ_ND_WITH_OPAQUE

#include "../common.h"
#include "ndarray_opaque_view.h"
#include "../detail/ndarray_wrapper.h"

namespace tlz {
	
namespace detail {
	template<std::size_t Dim, typename Frame_format, typename Allocator>
	using ndarray_opaque_base_ = detail::ndarray_wrapper<
		ndarray_opaque_view<Dim, true, Frame_format>,
		ndarray_opaque_view<Dim, false, Frame_format>,
		Allocator
	>;
}

template<std::size_t Dim, typename Frame_format, typename Allocator = raw_allocator>
class ndarray_opaque : public detail::ndarray_opaque_base_<Dim, Frame_format, Allocator> {
	using base = detail::ndarray_opaque_base_<Dim, Frame_format, Allocator>;

private:
	void construct_frames_();
	void destruct_frames_();

public:
	using typename base::view_type;
	using typename base::const_view_type;
	using typename base::shape_type;
	using typename base::strides_type;
	using frame_format_type = Frame_format;
	using frame_handle_type = typename view_type::frame_handle_type;
	using const_frame_handle_type = typename const_view_type::frame_handle_type;
	using frame_pointer_type = typename view_type::frame_pointer_type;
	using const_frame_pointer_type = typename const_view_type::frame_pointer_type;
	
	ndarray_opaque(const shape_type&, const frame_format_type&, std::size_t frame_padding = 0, const Allocator& = Allocator());
	explicit ndarray_opaque(const const_view_type& vw, std::size_t frame_padding = 0, const Allocator& = Allocator());
	ndarray_opaque(const ndarray_opaque&);
	ndarray_opaque(ndarray_opaque&&);

	~ndarray_opaque();
	
	void assign(const const_view_type& vw, std::size_t frame_padding = 0);
	
	ndarray_opaque& operator=(const const_view_type& vw) { assign(vw); return *this; }
		
	ndarray_opaque& operator=(const ndarray_opaque& arr);
	ndarray_opaque& operator=(ndarray_opaque&& arr);
	
	const frame_format_type& frame_format() const { return base::get_view_().frame_format(); }
	
	const_frame_handle_type frame_handle() const { return base::get_view_().frame_handle(); }
	operator const_frame_handle_type () const { return base::get_view_().frame_handle(); }

	frame_handle_type frame_handle() { return base::get_view_().frame_handle(); }
	operator frame_handle_type () { return base::get_view_().frame_handle(); }
};


template<typename Frame_format, typename Allocator = raw_allocator>
using ndarray_opaque_frame = ndarray_opaque<0, Frame_format, Allocator>;


template<typename Frame_format, typename Allocator = raw_allocator>
auto make_ndarray_opaque_frame(const Frame_format& frm, const Allocator& allocator = Allocator()) {
	return ndarray_opaque_frame<Frame_format, Allocator>(make_ndsize(), frm, 0, allocator);
}


template<std::size_t Dim, typename Frame_format, typename Allocator>
struct is_ndarray_opaque_view<ndarray_opaque<Dim, Frame_format, Allocator>> : std::true_type {};


}

#include "ndarray_opaque.tcc"

#endif
#endif
