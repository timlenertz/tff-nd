#ifndef TFF_NDARRAY_OPAQUE_TRAITS_H_
#define TFF_NDARRAY_OPAQUE_TRAITS_H_

namespace tff {

namespace detail {

template<typename From_view, typename To_view>
struct is_convertible_ndarray_opaque_view : conjunction<
	std::is_same<typename From_view::frame_format_type, typename To_view::frame_format_type>,
	std::integral_constant<bool, From_view::dimension() == To_view::dimension()>
> {};

}

template<typename View>
struct is_ndarray_opaque_view : std::false_type {};

template<typename From_view, typename To_view>
struct is_convertible_ndarray_opaque_view : conjunction<
	is_ndarray_opaque_view<From_view>,
	is_ndarray_opaque_view<To_view>,
	detail::is_convertible_ndarray_opaque_view<From_view, To_view>
> { };

};

#endif
