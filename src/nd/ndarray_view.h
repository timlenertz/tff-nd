#ifndef TFF_NDARRAY_VIEW_H_
#define TFF_NDARRAY_VIEW_H_

#include <type_traits>
#include <utility>
#include "common.h"
#include "ndcoord.h"
#include "ndspan.h"
#include "detail/ndarray_view_fcall.h"
#include "pod_array_format.h"
#include "ndarray_iterator.h"


namespace tff {

template<std::size_t Dim, typename T> class ndarray_view;

namespace detail {
	template<std::size_t Dim, typename T>
	ndarray_view<Dim - 1, T> get_subscript(const ndarray_view<Dim, T>& array, std::ptrdiff_t c) {
		return array.slice(c, 0);
	}
	
	template<typename T>
	T& get_subscript(const ndarray_view<1, T>& array, std::ptrdiff_t c) {
		return array.at({c});
	}
}


/// Mapping between coordinates, indices, and addresses of multi-dimensional data.
/** Templated for dimension and element type. `T` must be `elem` type, i.e. `elem_traits<T>` must be defined.
 ** The `ndarray_view` is defined by the three values `(start, shape, strides)`.
 ** - `start` is the pointer to the array element at indices `(0, 0, ..., 0)`.
 ** - `shape` is the vector of size `Dim` which defined length of array in each axis.
 ** - `strides` defined memory layout of elements: `strides[i]` defined distance _in bytes_ between `arr[][k][]`
 **    and `arr[][k+1][]` (the `i`-th coordinate changes).
 ** `ndarray_view` is non-owning view to data. `T` can be a `const` type. Always gives full access to elements, no
 ** matter if `this` const. Subscript operator `operator[]` and functions `section()`, etc., return another
 ** `ndarray_view` to given region.
 ** 
 ** Stride values may be negative to form reverse-order view on that axis (then `start` is no longer the element with
 ** lowest absolute address.) If strides are in non-descending order, coordinates to address mapping is no longer
 ** row-major. Strides need to be set to at least `sizeof(T)` and multiple of `alignof(T)`. Default strides produce
 ** row-major mapping, optionally with padding between elements. The term _default strides_ is still used here if
 ** there is inter-element padding. Coordinates map to address, but not the other way.
 ** 
 ** Coordinates map one-to-one to index, and index to coordinates. Index always advances in row-major order with
 ** coordinates, independently of strides. Index of coordinates `(0, 0, ..., 0)` is `0`.
 ** Random-access iterator `ndarray_iterator` always traverses ndarray in index order. As an optimization, iterator
 ** incrementation and decrementation is more efficient when all strides, or tail of strides, is default.
 **
 ** **Important**: Assignment and comparison operators perform deep assignment/comparison in the elements that the view
 ** points to, and not of the `ndarray_view` itself. Shallow assignment and comparison is done with `same()` and
 ** `reset()`. This simplifies interface: assigning a single element `arr[0][2] = 3` works the same as assigning an
 ** entire region `arr[0] = make_frame(...)`. (Because `operator[]` returns an `ndarray_view`.)
 ** Copy-constructing a view does not copy data. Semantics are similar to C++ references.
 **
 ** Default constructor, or `null()` returns _null view_. All null views compare equal (with `same()`), and `is_null()`
 ** or explicit `bool` conversion operator test for null view.
 ** Zero-length views (where `shape().product() == 0`) are possible, and are not equal to null views. */
template<std::size_t Dim, typename T>
class ndarray_view {
	static_assert(Dim >= 1, "ndarray_view dimension must be >= 1");

public:
	using value_type = T;
	using pointer = T*;
	using reference = T&;
	using index_type = std::ptrdiff_t;
	using coordinates_type = ndptrdiff<Dim>;
	using shape_type = ndsize<Dim>;
	using strides_type = ndptrdiff<Dim>;
	using span_type = ndspan<Dim>;
	
	using iterator = ndarray_iterator<Dim, T>;

	constexpr static std::size_t dimension = Dim;

protected:
	pointer start_;
	shape_type shape_;
	strides_type strides_;
	
	std::ptrdiff_t contiguous_length_;
	
	ndarray_view section_(std::ptrdiff_t dim, std::ptrdiff_t start, std::ptrdiff_t end, std::ptrdiff_t step) const;
	std::ptrdiff_t fix_coordinate_(std::ptrdiff_t c, std::ptrdiff_t dim) const;

private:
	using fcall_type = detail::ndarray_view_fcall<ndarray_view<Dim, T>, 1>;
	
public:
	/// \name Construction
	///@{
	/// Create null view.
	ndarray_view() : ndarray_view(nullptr, shape_type(0)) { }
	
	/// Create view with explicitly specified start, shape and strides.
	ndarray_view(pointer start, const shape_type&, const strides_type&);
	
	/// Create view with explicitly specified start and shape, with default strides (without padding).
	ndarray_view(pointer start, const shape_type& shape);
	
	/// Copy-construct view.
	/** Does not copy data. Can create `ndarray_view<const T>` from `ndarray_view<T>`. (But not the other way.) */
	ndarray_view(const ndarray_view<Dim, std::remove_const_t<T>>& arr) :
		ndarray_view(arr.start(), arr.shape(), arr.strides()) { }
	
	static ndarray_view null() { return ndarray_view(); }
	bool is_null() const { return (start_ == nullptr); }
	explicit operator bool () const { return ! is_null(); }

	template<typename... Args> void reset(const Args&... args) {
		reset(ndarray_view(args...));
	}
	void reset(const ndarray_view& other) ;
	///@}
	


	/// \name Attributes 
	///@{
	/// Number of elements, i.e. product of shape components.	
	pointer start() const { return start_; }
	const shape_type& shape() const { return shape_; }
	const strides_type& strides() const { return strides_; }
	
	std::size_t size() const { return shape().product(); }
	span_type full_span() const { return span_type(0, shape_); }
		
	/// Default strides which correspond to row-major order for specified shape.
	/** Optionally with \a padding between elements. */
	static strides_type default_strides(const shape_type&, std::size_t elem_padding = 0);
	
	/// Check if view has default strides.
	/** If \a minimal_dimension is specified, checks if view has default strides in dimensions from `Dim - 1` down to
	 ** \a minimal_dimension. Strides from `minimal_dimension - 1` down to `0` may be non-default. */
	bool has_default_strides(std::ptrdiff_t minimal_dimension = 0) const ;
	
	/// Returns padding of the view which has default strides.
	/** If view does not have default strides, throws exception.
	 ** \param minimal_dimension Like in has_default_strides(). */
	std::size_t default_strides_padding(std::ptrdiff_t minimal_dimension = 0) const;
	
	/// Check if view has default strides without padding.
	/** \param minimal_dimension Like in has_default_strides(). */
	bool has_default_strides_without_padding(std::ptrdiff_t minimal_dimension = 0) const ;
	///@}	



	/// \name Deep assignment
	///@{
	template<typename T2> void assign_static_cast(const ndarray_view<Dim, T2>&) const;
	template<typename T2> void assign(const ndarray_view<Dim, T2>&) const;
	void assign(const ndarray_view<Dim, const T>& other) const;

	template<typename Arg> const ndarray_view& operator=(Arg&& arg) const
		{ assign(std::forward<Arg>(arg)); return *this; }
	const ndarray_view& operator=(const ndarray_view& other) const
		{ assign(other); return *this; }
	///@}



	/// \name Deep comparison
	///@{
	template<typename T2> bool compare(const ndarray_view<Dim, T2>&) const;
	bool compare(const ndarray_view<Dim, const T>& other) const;
		
	template<typename Arg> bool operator==(Arg&& arg) const { return compare(std::forward<Arg>(arg)); }
	template<typename Arg> bool operator!=(Arg&& arg) const { return ! compare(std::forward<Arg>(arg)); }
	///@}
	
	
	
	/// \name Iteration
	///@{
	std::ptrdiff_t contiguous_length() const { return contiguous_length_; }

	coordinates_type index_to_coordinates(index_type) const;
	index_type coordinates_to_index(const coordinates_type&) const;
	pointer coordinates_to_pointer(const coordinates_type&) const;	
			
	iterator begin() const;
	iterator end() const;
	///@}



	/// \name Indexing
	///@{
	/// Access element at coordinates \a coord.
	reference at(const coordinates_type& coord) const;

	
	/// Cuboid section of view, with interval in each axis.
	/** Can also specify step for each axis: Stride of the new view are stride of this view multiplied by step.
	 ** Step `1` does not change stride, step `2` skips every second element on that axis, negative step also reverses
	 ** direction. It is not necessary that coordinate of last element on an axis coincides with `end - 1`. */
	ndarray_view section
		(const coordinates_type& start, const coordinates_type& end, const strides_type& steps = strides_type(1)) const;
	
	/// Cuboid section of view, defined using `ndspan` object.
	ndarray_view section(const span_type& span, const strides_type& steps = strides_type(1)) const {
		return section(span.start_pos(), span.end_pos(), steps);
	}
	
	/// Create \ref ndarray_view with one less dimension, by fixing coordinate of axis \a dimension to \a c.
	ndarray_view<Dim - 1, T> slice(std::ptrdiff_t c, std::ptrdiff_t dimension) const;
	
	/// Subscript operator, creates slice on first dimension.
	/** If `Dim > 1`, equivalent to `slide(c, 0)`. If `Dim == 1`, returns reference to `c`-th element in view.
	 ** Can access elements in multi-dimensional array like `arr[i][j][k] = value`. */
	decltype(auto) operator[](std::ptrdiff_t c) const {
		return detail::get_subscript(*this, c);
	}
	
	fcall_type operator()(std::ptrdiff_t start, std::ptrdiff_t end, std::ptrdiff_t step = 1) const {
		return section_(0, start, end, step);
	}
	fcall_type operator()(std::ptrdiff_t c) const {
		return section_(0, c, c + 1, 1);
	}
	fcall_type operator()() const {
		return *this;
	}
	///@}
		
	
	

	ndarray_view<1 + Dim, T> add_front_axis() const;
	
	ndarray_view swapaxis(std::size_t axis1, std::size_t axis2) const;
};


template<std::size_t Dim, typename T1, typename T2>
bool same(const ndarray_view<Dim, T1>&, const ndarray_view<Dim, T2>&);



template<std::size_t Dim_to, typename Elem_to, std::size_t Dim_from, typename Elem_from>
constexpr bool ndarray_view_is_assignable = 
	(Dim_to == Dim_from) &&
	! std::is_const<Elem_to>::value &&
	std::is_convertible<Elem_from, Elem_to>::value;



template<typename T>
ndarray_view<2, T> flip(const ndarray_view<2, T>& vw) {
	return vw.swapaxis(0, 1);
}


template<std::size_t Dim, typename T, std::size_t New_dim>
ndarray_view<New_dim, T> reshape(const ndarray_view<Dim, T>&, const ndsize<New_dim>&);


template<std::size_t Dim, typename T>
ndarray_view<1, T> flatten(const ndarray_view<Dim, T>&);


///////////////


template<std::size_t Tail_dim, std::size_t Dim, typename Elem>
bool tail_has_pod_format(const ndarray_view<Dim, Elem>& vw) {
	return std::is_pod<Elem>::value && vw.has_default_strides(Dim - Tail_dim);
}

template<std::size_t Dim, typename Elem>
bool has_pod_format(const ndarray_view<Dim, Elem>& vw) {
	return tail_has_pod_format<Dim>(vw);
}


template<std::size_t Tail_dim, std::size_t Dim, typename Elem>
pod_array_format tail_pod_format(const ndarray_view<Dim, Elem>& vw) {
	Assert(tail_has_pod_format<Tail_dim>(vw));
	std::size_t count = tail<Tail_dim>(vw.shape()).product();
	std::size_t stride = vw.strides().back();
	return make_pod_array_format<Elem>(count, stride);
}


template<std::size_t Dim, typename Elem>
pod_array_format pod_format(const ndarray_view<Dim, Elem>& vw) {
	return tail_pod_format<Dim>(vw);
}

}

#include "ndarray_view.tcc"

#endif
