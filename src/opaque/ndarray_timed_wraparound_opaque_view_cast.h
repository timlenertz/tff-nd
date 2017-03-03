#ifndef TFF_NDARRAY_TIMED_WRAPAROUND_OPAQUE_VIEW_CAST_H_
#define TFF_NDARRAT_TIMED_WRAPAROUND_OPAQUE_VIEW_CAST_H_

#include "../config.h"
#if TFF_ND_WITH_OPAQUE && TFF_ND_WITH_TIMED && TFF_ND_WITH_WRAPAROUND

#include "ndarray_timed_wraparound_opaque_view.h"
#include "ndarray_wraparound_opaque_view_cast.h"
#include "../ndarray_timed_wraparound_view.h"

namespace tff {

template<std::size_t Opaque_dim, typename Opaque_frame_format, std::size_t Concrete_dim, typename Concrete_elem>
auto to_opaque(const ndarray_timed_wraparound_view<Concrete_dim, Concrete_elem>& concrete_view, const Opaque_frame_format& frm) {
	return timed(to_opaque<Opaque_dim>(concrete_view.non_timed(), frm), concrete_view.start_time());
}

template<std::size_t Opaque_dim, std::size_t Concrete_dim, typename Concrete_elem>
auto to_opaque(const ndarray_timed_wraparound_view<Concrete_dim, Concrete_elem>& concrete_view) {
	return timed(to_opaque<Opaque_dim>(concrete_view.non_timed()), concrete_view.start_time());
}

template<std::size_t Concrete_dim, typename Concrete_elem, std::size_t Opaque_dim, bool Opaque_mutable, typename Opaque_frame_format>
auto from_opaque(const ndarray_timed_wraparound_opaque_view<Opaque_dim, Opaque_mutable, Opaque_frame_format>& opaque_view) {
	return timed(from_opaque<Concrete_dim, Concrete_elem>(opaque_view.non_timed()), opaque_view.start_time());
}

}

#endif
#endif