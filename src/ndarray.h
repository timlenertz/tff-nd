#ifndef TFF_NDARRAY_H_
#define TFF_NDARRAY_H_

#include "config.h"
#if TFF_ND_WITH_ALLOCATION

#include <memory>
#include "ndarray_view.h"
#include "detail/ndarray_wrapper.h"

namespace tff {


/// Container for \ref ndarray_view.
template<std::size_t Dim, typename Elem, typename Allocator>
class ndarray;

template<std::size_t Dim, typename T, typename Allocator>
struct is_ndarray_view<ndarray<Dim, T, Allocator>> : std::true_type {};

/// Container for \ref ndarray_view.
template<std::size_t Dim, typename Elem, typename Allocator = std::allocator<Elem>>
class ndarray : public detail::ndarray_wrapper<ndarray_view<Dim, Elem>, ndarray_view<Dim, const Elem>, Allocator> {
	using base = detail::ndarray_wrapper<ndarray_view<Dim, Elem>, ndarray_view<Dim, const Elem>, Allocator>;

private:
	template<typename Other_view, typename U = void>
	using enable_if_convertible_ = std::enable_if_t<is_convertible_ndarray_view<Other_view, ndarray>::value, U>;
	
	void construct_elems_();
	void destruct_elems_();

public:
	using typename base::view_type;
	using typename base::const_view_type;
	using typename base::shape_type;
	using typename base::strides_type;
	
	using value_type = Elem;
	using pointer = Elem*;
	using reference = Elem&;
	
	/// \name Constructor
	///@{	
	/// Construct empty \ref ndarray with given shape.
	/** Has default strides, optionally with specified element padding. */
	explicit ndarray(const shape_type& shape, std::size_t elem_padding = 0, const Allocator& = Allocator());
	
	/// Construct \ref ndarray with shape and copy of elements from a \ref ndarray_view.
	/** Has default strides, optionally with specified element padding. Does not take strides from \a vw. */
	template<typename Other_view, typename = enable_if_convertible_<Other_view>>
	explicit ndarray(const Other_view& vw, std::size_t elem_padding = 0, const Allocator& = Allocator());
	
	
	/// Copy-construct from another \ref ndarray of same type.
	/** Takes strides from \a arr. */
	ndarray(const ndarray& arr);
	
	/// Move-construct from another \ref ndarray of same type.
	/** Takes strides from \a arr and sets \a arr to null. */
	ndarray(ndarray&& arr);
	
	~ndarray();
	///@}
	

	/// \name Deep assignment
	///@{
	/// Assign shape and elements from \ref vw.
	/** Resets to default strides, optionally with specified element padding. Reallocates memory if necessary. */
	template<typename Other_view>
	enable_if_convertible_<Other_view> assign(const Other_view& vw, std::size_t elem_padding = 0);
	
	/// Assign shape and elements from \ref vw.
	/** Equivalent to `assign(vw)`. */
	template<typename Other_view>
	enable_if_convertible_<Other_view, ndarray&> operator=(const Other_view& vw)
		{ assign(vw); return *this; }

	/// Copy-assign from another \ref ndarray.
	/** Takes strides from \a arr */
	ndarray& operator=(const ndarray& arr);

	/// Move-assign from another \ref ndarray.
	/** Takes strides from \a arr and sets \a arr to null. */
	ndarray& operator=(ndarray&& arr);
	///@}
};


template<std::size_t Dim, typename Elem>
auto make_ndarray(const ndarray_view<Dim, Elem>& vw) {
	using array_elem_type = std::remove_const_t<Elem>;
	using array_type = ndarray<Dim, array_elem_type>;
	return array_type(vw);
}


}

#include "ndarray.tcc"

#endif
#endif
