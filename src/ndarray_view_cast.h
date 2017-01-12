#ifndef TFF_NDARRAY_VIEW_CAST_H_
#define TFF_NDARRAY_VIEW_CAST_H_

#include <cstddef>
#include <type_traits>
#include "common.h"
#include "ndarray_view.h"
#include "ndarray_timed_view.h"
#include "elem.h"
#include "elem_tuple.h"

namespace tff {
	
template<typename Output_view, typename Input_view> Output_view ndarray_view_cast(const Input_view& vw);

namespace detail {
	template<typename Output_view, typename Input_view>
	struct ndarray_view_caster;
	
	// no-op cast
	template<std::size_t Dim, typename Elem>
	struct ndarray_view_caster<
		ndarray_view<Dim, Elem>, // out
		ndarray_view<Dim, Elem> // in
	>{
		using view_type = ndarray_view<Dim, Elem>;
		const view_type& operator()(const view_type& arr) const {
			return arr;
		}
	};

	
	#if TFF_ND_WITH_ELEM
	
	// single element from elem_tuple
	template<typename Output_elem, std::size_t Dim, typename... Input_elems>
	struct ndarray_view_caster<
		ndarray_view<Dim, Output_elem>, // out
		ndarray_view<Dim, elem_tuple<Input_elems...>> // in
	>{
		using input_tuple_type = elem_tuple<Input_elems...>;
		
		using output_view_type = ndarray_view<Dim, Output_elem>;
		using input_view_type = ndarray_view<Dim, input_tuple_type>;
						
		output_view_type operator()(const input_view_type& arr) const {
			constexpr std::ptrdiff_t index = elem_tuple_index<Output_elem, input_tuple_type>();
			constexpr std::ptrdiff_t offset = elem_tuple_offset<index, input_tuple_type>();

			auto* start = reinterpret_cast<Output_elem*>(
				advance_raw_ptr(arr.start(), offset)
			);
			return output_view_type(
				start,
				arr.shape(),
				arr.strides()
			);
		}
	};
	
	// scalars from elem
	template<std::size_t Dim, typename Input_elem>
	struct ndarray_view_caster<
		ndarray_view<Dim + 1, typename elem_traits<std::remove_cv_t<Input_elem>>::scalar_type>, // out
		ndarray_view<Dim, Input_elem> // in
	>{
		using elem_traits_type = elem_traits<std::remove_cv_t<Input_elem>>;
		using elem_scalar_type = typename elem_traits_type::scalar_type;
		
		using output_view_type = ndarray_view<Dim + 1, elem_scalar_type>;
		using input_view_type = ndarray_view<Dim, Input_elem>;
		
		output_view_type operator()(const input_view_type& arr) const {
			auto* start = reinterpret_cast<elem_scalar_type*>(arr.start());
			return output_view_type(
				start,
				ndcoord_cat(arr.shape(), make_ndsize(elem_traits_type::components)),
				ndcoord_cat(arr.strides(), make_ndptrdiff(elem_traits_type::stride))
			);
		}
	};
	
	// scalars from elem, const
	template<std::size_t Dim, typename Input_elem>
	struct ndarray_view_caster<
		ndarray_view<Dim + 1, const typename elem_traits<std::remove_cv_t<Input_elem>>::scalar_type>, // out
		ndarray_view<Dim, Input_elem> // in
	>{
		using elem_traits_type = elem_traits<std::remove_cv_t<Input_elem>>;
		using elem_scalar_type = const typename elem_traits_type::scalar_type;
		
		using output_view_type = ndarray_view<Dim + 1, const elem_scalar_type>;
		using input_view_type = ndarray_view<Dim, Input_elem>;
		
		output_view_type operator()(const input_view_type& arr) const {
			auto* start = reinterpret_cast<const elem_scalar_type*>(arr.start());
			return output_view_type(
				start,
				ndcoord_cat(arr.shape(), make_ndsize(elem_traits_type::components)),
				ndcoord_cat(arr.strides(), make_ndptrdiff(elem_traits_type::stride))
			);
		}
	};
	
	#endif

	
	#if TFF_ND_WITH_TIMED
	
	// ndarray_timed_view to ndarray_timed_view
	template<std::size_t Output_dim, typename Output_elem, std::size_t Input_dim, typename Input_elem>
	struct ndarray_view_caster<
		ndarray_timed_view<Output_dim, Output_elem>, // out
		ndarray_timed_view<Input_dim, Input_elem> // in
	>{
		using output_view_type = ndarray_timed_view<Output_dim, Output_elem>;
		using input_view_type = ndarray_timed_view<Input_dim, Input_elem>;

		using untimed_output_view_type = ndarray_view<Output_dim, Output_elem>;
		using untimed_input_view_type = ndarray_view<Input_dim, Input_elem>;
		
		output_view_type operator()(const input_view_type& vw) const {
			auto untimed_output_view = ndarray_view_cast<untimed_output_view_type>(
				untimed_input_view_type(vw)
			);
			return ndarray_timed_view<Output_dim, Output_elem>(untimed_output_view, vw.start_time());
		}
	};
	
	#endif
	
	template<typename Output_view, typename Input_view, typename Void = void_t<>>
	struct ndarray_view_cast_detector : std::false_type { };
	
	template<typename Output_view, typename Input_view>
	struct ndarray_view_cast_detector<
		Output_view,
		Input_view,
		void_t<ndarray_view_caster<Output_view, Input_view>>
	> : std::true_type { };
}


/// Cast \ref ndarray_view to one with different dimension and/or element type.
/** The input and output views may be of type \ref ndarray_view or \ref ndarray_timed_view. No data is copied or
 ** modified, instead the dimension, start, shape and stride of casted view is adjusted so as to point to a subset
 ** of the original view's raw data.
 ** - _No-op_: Output and input views have same dimension and element type.
 ** - _Tuple element_: Input element type is an `elem_tuple`. Output element type is the type of one of the elements
 **                    in this tuple. Input and output dimension is same. Returns view of same shape, but with changed
 **                    start and strides, which covers only that element in each tuple.
 **                    Example: `ndarray_view<1, point_xyzrgb>` --> `ndarray_view<1, rgb_color>`
 ** - _Scalars from vector_: Input element type is vector type, such as `rgb_color`. Output element type is the scalar
 **                          type of this vector. Output dimension is one more than input dimension. Returns view where
 **                          added, last dimension is index of scalar element from the original vector elements.
 **                          Example: `ndarray_view<2, rgb_color>` --> `ndarray_view<3, byte>`. */
template<typename Output_view, typename Input_view>
Output_view ndarray_view_cast(const Input_view& vw) {
	detail::ndarray_view_caster<Output_view, Input_view> caster;
	return caster(vw);
}


/// Verify at compile time whether cast from `Input_view` to `Output_view` is possible.
template<typename Output_view, typename Input_view>
constexpr bool ndarray_view_can_cast =
	detail::ndarray_view_cast_detector<Output_view, Input_view>::value;


/// Cast \ref ndarray_view to one with different, unrelated element type.
/** Size and alignment requirement of `Output_view::value_type` must be $leq$ to that of `Input_view::value_type`.
 ** No data is copied or modified, the returned \ref ndarray_view points to the same data.
 ** Modifying the element `out_vw.at(coord)` only affects `in_vw.at(coord)`, the value at the same coordinates of the
 ** input view. */
template<typename Output_view, typename Input_view>
Output_view ndarray_view_reinterpret_cast(const Input_view& in_view) {
	using out_elem_type = typename Output_view::value_type;
	static_assert(Output_view::dimension() == Input_view::dimension(), "output and input view must have same dimension");
	std::ptrdiff_t in_stride = in_view.strides().back();
	if(in_stride < sizeof(out_elem_type))
		throw std::invalid_argument("output ndarray_view elem type is too large");
	if(in_stride % alignof(out_elem_type) != 0)
		throw std::invalid_argument("output ndarray_view elem type has incompatible alignment");
	
	auto new_start = reinterpret_cast<out_elem_type*>(in_view.start());
	return Output_view(new_start, in_view.shape(), in_view.strides());
}


}

#endif

