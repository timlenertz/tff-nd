#ifndef TFF_NDARRAY_TIMED_OPAQUE_VIEW_H_
#define TFF_NDARRAY_TIMED_OPAQUE_VIEW_H_

#include "../config.h"
#if TFF_ND_WITH_OPAQUE && TFF_ND_WITH_TIMED

#include "ndarray_opaque_view.h"
#include "../detail/ndarray_timed_view_derived.h"

namespace tff {

template<std::size_t Dim, bool Mutable, typename Frame_format>
using ndarray_timed_opaque_view = detail::ndarray_timed_view_derived<ndarray_opaque_view<Dim, Mutable, Frame_format>>;

}

#endif
#endif
