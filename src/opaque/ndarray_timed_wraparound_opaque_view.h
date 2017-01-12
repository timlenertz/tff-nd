#ifndef TFF_NDARRAY_TIMED_WRAPAROUND_OPAQUE_VIEW_H_
#define TFF_NDARRAY_TIMED_WRAPAROUND_OPAQUE_VIEW_H_

#include "../config.h"
#if TFF_ND_WITH_OPAQUE && TFF_ND_WITH_TIMED && TFF_ND_WITH_WRAPAROUND

#include "ndarray_wraparound_opaque_view.h"
#include "../detail/ndarray_timed_view_derived.h"

namespace tff {

template<std::size_t Dim, bool Mutable, typename Frame_format>
using ndarray_timed_wraparound_opaque_view = detail::ndarray_timed_view_derived<ndarray_wraparound_opaque_view<Dim, Mutable, Frame_format>>;

template<std::size_t Dim, bool Mutable, typename Frame_format>
auto timed(const ndarray_wraparound_opaque_view<Dim, Mutable, Frame_format>& vw, time_unit start_time = 0) {
	return ndarray_timed_wraparound_opaque_view<Dim, Mutable, Frame_format>(vw, start_time);
}

}

#endif
#endif
