#ifndef TFF_NDARRAY_WRAPAROUND_OPAQUE_VIEW_H_
#define TFF_NDARRAY_WRAPAROUND_OPAQUE_VIEW_H_

#include "../config.h"
#if TFF_ND_WITH_OPAQUE && TFF_ND_WITH_WRAPAROUND

#include "../detail/ndarray_opaque_view_wrapper.h"
#include "../ndarray_wraparound_view.h"
#include "ndarray_opaque_view.h"

namespace tff {

template<std::size_t Dim, bool Mutable, typename Frame_format>
using ndarray_wraparound_opaque_view = detail::ndarray_opaque_view_wrapper<Dim, Mutable, Frame_format, ndarray_wraparound_view>;


template<std::size_t Tail_dim, std::size_t Dim, bool Mutable, typename Frame_format>
bool tail_has_pod_format(const ndarray_wraparound_opaque_view<Dim, Mutable, Frame_format>& vw) {
	return (Tail_dim == 0) && (vw.frame_format().is_pod());
}


template<std::size_t Dim, bool Mutable, typename Frame_format>
bool has_pod_format(const ndarray_wraparound_opaque_view<Dim, Mutable, Frame_format>& vw) {
	return tail_has_pod_format<Dim>(vw);
}


template<std::size_t Tail_dim, std::size_t Dim, bool Mutable, typename Frame_format>
pod_array_format tail_pod_format(const ndarray_wraparound_opaque_view<Dim, Mutable, Frame_format>& vw) {
	pod_array_format frame_pod_format = vw.frame_format().pod_format();
	Assert(Tail_dim == 0);
	return vw.frame_format().pod_format();
}

template<std::size_t Dim, bool Mutable, typename Frame_format>
pod_array_format pod_format(const ndarray_wraparound_opaque_view<Dim, Mutable, Frame_format>& vw) {
	return tail_pod_format<Dim>(vw);
}


template<std::size_t Dim, bool Mutable, typename Frame_format>
ndarray_wraparound_opaque_view<Dim, Mutable, Frame_format> wraparound(
	const ndarray_opaque_view<Dim, Mutable, Frame_format>& vw,
	const ndptrdiff<Dim>& start,
	const ndptrdiff<Dim>& end,
	const ndptrdiff<Dim>& steps = ndptrdiff<Dim>(1)
) {
	auto base_view = wraparound(vw.base_view(), ndcoord_cat(start, 0), ndcoord_cat(end, 1), ndcoord_cat(steps, 1));
	return ndarray_wraparound_opaque_view<Dim, Mutable, Frame_format>(base_view, vw.frame_format());
}


}

#endif

#endif
