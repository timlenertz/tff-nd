#ifndef TFF_NDARRAY_TRAITS_H_
#define TFF_NDARRAY_TRAITS_H_

namespace tff {

template<typename View>
struct is_ndarray_view : std::false_type {};


template<typename From_view, typename To_view>
struct is_convertible_ndarray_view : conjunction<
	is_ndarray_view<From_view>,
	is_ndarray_view<To_view>,
	std::is_convertible<typename From_view::value_type, typename To_view::value_type>,
	std::integral_constant<bool, From_view::dimension() == To_view::dimension()>
> { };

}

#endif