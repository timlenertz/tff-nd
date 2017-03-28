#ifndef TFF_NDARRAY_OPAQUE_VIEW_H_
#define TFF_NDARRAY_OPAQUE_VIEW_H_

#include "../config.h"
#if TFF_ND_WITH_OPAQUE

#include "../detail/ndarray_opaque_view_wrapper.h"
#include "../ndarray_view.h"

namespace tff {

template<std::size_t Dim, bool Mutable, typename Frame_format>
using ndarray_opaque_view = detail::ndarray_opaque_view_wrapper<Dim, Mutable, Frame_format, ndarray_view>;


template<std::size_t Dim, bool Mutable, typename Frame_format>
struct is_ndarray_opaque_view<ndarray_opaque_view<Dim, Mutable, Frame_format>> : std::true_type {};

}

#endif

#endif