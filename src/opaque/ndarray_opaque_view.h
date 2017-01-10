#ifndef TFF_NDARRAY_OPAQUE_VIEW_H_
#define TFF_NDARRAY_OPAQUE_VIEW_H_

#include "../config.h"
#if TFF_ND_WITH_OPAQUE

#include "../detail/ndarray_opaque_view_wrapper.h"
#include "../ndarray_view.h"

namespace tff {

template<std::size_t Dim, bool Mutable, typename Frame_format>
using ndarray_opaque_view = detail::ndarray_opaque_view_wrapper<Dim, Mutable, Frame_format, ndarray_view>;



template<std::size_t Tail_dim, std::size_t Dim, bool Mutable, typename Frame_format>
bool tail_has_pod_format(const ndarray_opaque_view<Dim, Mutable, Frame_format>& vw) {
	if(! vw.frame_format().is_pod()) return false;
	pod_array_format frame_pod_format = vw.frame_format().pod_format();
	if(frame_pod_format.is_contiguous()) return vw.has_default_strides(Dim - Tail_dim);
	else return vw.has_default_strides_without_padding(Dim - Tail_dim);
}


template<std::size_t Dim, bool Mutable, typename Frame_format>
bool has_pod_format(const ndarray_opaque_view<Dim, Mutable, Frame_format>& vw) {
	return tail_has_pod_format<Dim>(vw);
}


template<std::size_t Tail_dim, std::size_t Dim, bool Mutable, typename Frame_format>
pod_array_format tail_pod_format(const ndarray_opaque_view<Dim, Mutable, Frame_format>& vw) {
	pod_array_format frame_pod_format = vw.frame_format().pod_format();
	
	if(Tail_dim == 0) {
		return frame_pod_format;
		
	} else if(frame_pod_format.is_contiguous()) {
		std::size_t frame_padding = vw.default_strides_padding(Dim - Tail_dim);
		std::size_t elem_size = frame_pod_format.size();
		std::size_t elem_align = frame_pod_format.elem_alignment();
		std::size_t length = tail<Tail_dim>(vw.shape()).product();
		std::size_t stride = vw.strides().back();
		return pod_array_format(elem_size, elem_align, length, stride);
		
	} else {
		Assert(vw.has_default_strides_without_padding(Dim - Tail_dim));
		std::size_t elem_size = frame_pod_format.elem_size();
		std::size_t elem_align = frame_pod_format.elem_alignment();
		std::size_t length = tail<Tail_dim>(vw.shape()).product() * frame_pod_format.length();
		std::size_t stride = frame_pod_format.stride();
		return pod_array_format(elem_size, elem_align, length, stride);
	}
}

template<std::size_t Dim, bool Mutable, typename Frame_format>
pod_array_format pod_format(const ndarray_opaque_view<Dim, Mutable, Frame_format>& vw) {
	return tail_pod_format<Dim>(vw);
}


}

#endif

#endif