#ifndef TFF_NDARRAY_TIMED_VIEW_H_
#define TFF_NDARRAY_TIMED_VIEW_H_

#include "config.h"
#if TFF_ND_WITH_TIMED

#include "common.h"
#include "ndarray_view.h"
#include "detail/ndarray_timed_view_derived.h"
#include "ndarray_traits.h"

namespace tff {

/// \ref ndarray_view with absolute time indices associated to first dimension.
/** Each frame `vw[i]` is associated with time index `t = start_time + i`. */
template<std::size_t Dim, typename T>
using ndarray_timed_view = detail::ndarray_timed_view_derived<ndarray_view<Dim, T>>;

template<std::size_t Dim, typename T>
struct is_ndarray_view<detail::ndarray_timed_view_derived<ndarray_view<Dim, T>>> : std::true_type {};

template<std::size_t Dim, typename T>
auto timed(const ndarray_view<Dim, T>& vw, time_unit start_time = 0) {
	return ndarray_timed_view<Dim, T>(vw, start_time);
}

};

#endif
#endif
