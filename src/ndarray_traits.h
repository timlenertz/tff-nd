#ifndef TFF_NDARRAY_TRAITS_H_
#define TFF_NDARRAY_TRAITS_H_

#include "common.h"

namespace tff {

namespace detail {

template<typename From_view, typename To_view>
struct is_convertible_ndarray_view : conjunction<
	std::is_convertible<typename From_view::value_type, typename To_view::value_type>,
	std::integral_constant<bool, From_view::dimension() == To_view::dimension()>
> { };

};

template<typename View>
struct is_ndarray_view : std::false_type {};


template<typename From_view, typename To_view>
struct is_convertible_ndarray_view : conjunction<
	is_ndarray_view<From_view>,
	is_ndarray_view<To_view>,
	detail::is_convertible_ndarray_view<From_view, To_view>
> { };

}

#endif
