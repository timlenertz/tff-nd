#ifndef TFF_NDARRAY_TIMED_WRAPAROUND_VIEW_H_
#define TFF_NDARRAY_TIMED_WRAPAROUND_VIEW_H_

#include "config.h"
#if TFF_ND_WITH_TIMED && TFF_ND_WITH_WRAPAROUND

#include "common.h"
#include "ndarray_wraparound_view.h"
#include "detail/ndarray_timed_view_derived.h"

namespace tff {

template<std::size_t Dim, typename T>
using ndarray_timed_wraparound_view = detail::ndarray_timed_view_derived<ndarray_wraparound_view<Dim, T>>;

}

#endif

#endif