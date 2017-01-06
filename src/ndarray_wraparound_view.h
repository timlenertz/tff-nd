#ifndef TFF_NDARRAY_WRAPAROUND_VIEW_H_
#define TFF_NDARRAY_WRAPAROUND_VIEW_H_

#include "config.h"
#if TFF_ND_WITH_WRAPAROUND

#include "ndarray_view.h"
#include "detail/ndarray_view_fcall.h"
#include "ndarray_iterator.h"
#include <utility>

namespace tff {

template<std::size_t Dim, typename T>
class ndarray_wraparound_view;

template<std::size_t Dim, typename T>
struct is_ndarray_view<ndarray_wraparound_view<Dim, T>> : std::true_type {};


namespace detail {
	template<std::size_t Dim, typename T>
	ndarray_wraparound_view<Dim - 1, T> get_subscript(const ndarray_wraparound_view<Dim, T> &array, std::ptrdiff_t c) {
		return array.slice(c, 0);
	}

	template<typename T>
	T &get_subscript(const ndarray_wraparound_view<1, T> &array, std::ptrdiff_t c) {
		return array.at({c});
	}
}


template<std::size_t Dim, typename T>
class ndarray_wraparound_view : private ndarray_view<Dim, T> {
	using base = ndarray_view<Dim, T>;
	
public:
	using typename base::value_type;
	using typename base::pointer;
	using typename base::reference;
	using typename base::index_type;
	using typename base::coordinates_type;
	using typename base::shape_type;
	using typename base::strides_type;
	using typename base::span_type;
	
	using iterator = ndarray_iterator<ndarray_wraparound_view<Dim, T>>;
	
protected:
	strides_type wrap_offsets_;
	strides_type wrap_circumferences_;
	
	using base::fix_coordinate_;

private:
	template<typename Other_view, typename U = void>
	using enable_if_convertible_ = std::enable_if_t<is_convertible_ndarray_view<Other_view, ndarray_wraparound_view>::value, U>;
	
	using fcall_type = detail::ndarray_view_fcall<ndarray_wraparound_view<Dim, T>, 1>;

public:
	/// \name Construction
	///@{
	ndarray_wraparound_view() { }
	
	ndarray_wraparound_view(
		pointer start,
		const shape_type& shape,
		const strides_type& strides,
		const strides_type& offsets,
		const strides_type& circumferences
	);
	
	ndarray_wraparound_view(const ndarray_view<Dim, std::remove_const_t<T>>&);
	
	ndarray_wraparound_view(const ndarray_wraparound_view<Dim, std::remove_const_t<T>>&);
	
	static ndarray_wraparound_view null() { return ndarray_wraparound_view(); }
	
	bool is_null() const { return (start() == nullptr); }
	explicit operator bool() const { return !is_null(); }
	
	template<typename... Args>
	void reset(const Args &... args) { reset(ndarray_wraparound_view(args...)); }
	
	void reset(const ndarray_wraparound_view &);
	///@}
	
	/// \name Attributes
	///@{
	using base::dimension;
	using base::start;
	using base::shape;
	using base::strides;
	using base::size;
	using base::full_span;
	
	const strides_type& wrap_offsets() const { return wrap_offsets_; }
	const strides_type& wrap_circumferences() const { return wrap_circumferences_; }
	///@}
	
	
	/// \name Deep assignment
	///@{
	template<typename Other_view>
	enable_if_convertible_<Other_view> assign(const Other_view &) const;
	
	template<typename Arg>
	const ndarray_wraparound_view &operator=(Arg &&arg) const {
		assign(std::forward<Arg>(arg));
		return *this;
	}
	
	const ndarray_wraparound_view &operator=(const ndarray_wraparound_view &other) const {
		assign(other);
		return *this;
	}
	///@}
	
	
	
	/// \name Deep comparison
	///@{
	template<typename Other_view>
	enable_if_convertible_<Other_view, bool> compare(const Other_view &) const;
	
	template<typename Arg>
	bool operator==(Arg &&arg) const { return compare(std::forward<Arg>(arg)); }
	
	template<typename Arg>
	bool operator!=(Arg &&arg) const { return !compare(std::forward<Arg>(arg)); }
	///@}
	
	
	/// \name Iteration
	///@{
	std::ptrdiff_t contiguous_length() const { return 1; }
	
	iterator begin() const;
	iterator end() const;
	///@}
	
	
	/// \name Indexing
	///@{
	ndarray_wraparound_view
	axis_section(std::ptrdiff_t axis, std::ptrdiff_t start, std::ptrdiff_t end, std::ptrdiff_t step) const;
	
	using base::index_to_coordinates;
	using base::coordinates_to_index;
	
	pointer coordinates_to_pointer(const coordinates_type &) const;
	
	reference at(const coordinates_type &coord) const;
	
	ndarray_wraparound_view section
		(const coordinates_type &start, const coordinates_type &end, const strides_type &steps = strides_type(1)) const;
	
	ndarray_wraparound_view section(const span_type &span, const strides_type &steps = strides_type(1)) const {
		return section(span.start_pos(), span.end_pos(), steps);
	}
	
	ndarray_wraparound_view<Dim - 1, T> slice(std::ptrdiff_t c, std::ptrdiff_t dim) const;
	
	decltype(auto) operator[](std::ptrdiff_t c) const {
		return detail::get_subscript(*this, c);
	}
	
	fcall_type operator()(std::ptrdiff_t start, std::ptrdiff_t end, std::ptrdiff_t step = 1) const {
		return axis_section(0, start, end, step);
	}
	
	fcall_type operator()(std::ptrdiff_t c) const {
		return axis_section(0, c, c + 1, 1);
	}
	
	fcall_type operator()() const {
		return *this;
	}
	///@}
};


template<std::size_t Dim, typename T>
ndarray_wraparound_view<Dim, T> wraparound(
	const ndarray_view<Dim, T>& vw,
	const ndptrdiff<Dim>& start,
	const ndptrdiff<Dim>& end,
	const ndptrdiff<Dim>& steps = ndptrdiff<Dim>(1)
);


template<std::size_t Dim, typename T1, typename T2>
bool same(const ndarray_wraparound_view<Dim, T1> &, const ndarray_wraparound_view<Dim, T2> &);


template<typename T>
ndarray_wraparound_view<2, T> flip(const ndarray_wraparound_view<2, T>& vw) { return swapaxis(vw, 0, 1); }

template<std::size_t Dim, typename T>
ndarray_wraparound_view<Dim, T> swapaxis(const ndarray_wraparound_view<Dim, T>&, std::ptrdiff_t axis1, std::ptrdiff_t axis2);

template<std::size_t Dim, typename T>
ndarray_wraparound_view<Dim, T> step(const ndarray_wraparound_view<Dim, T>& vw, std::ptrdiff_t axis, std::ptrdiff_t step) {
	return vw.axis_section(axis, 0, vw.shape()[axis], step);
};

template<typename T>
ndarray_wraparound_view<1, T> step(const ndarray_wraparound_view<1, T>& vw, std::ptrdiff_t step) {
	return vw.axis_section(0, 0, vw.shape()[0], step);
};

template<std::size_t Dim, typename T>
ndarray_wraparound_view<Dim, T> reverse(const ndarray_wraparound_view<Dim, T>& vw, std::ptrdiff_t axis = 0) {
	return step(vw, axis, -1);
};



}

#include "ndarray_wraparound_view.tcc"

#endif

#endif