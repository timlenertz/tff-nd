#ifndef TFF_NDARRAY_WRAPAROUND_VIEW_H_
#define TFF_NDARRAY_WRAPAROUND_VIEW_H_

#include "ndarray_view.h"
#include "detail/ndarray_view_fcall.h"
#include <utility>

namespace tff {

template<std::size_t Dim, typename T>
class ndarray_wraparound_view;

namespace detail {
	template<std::size_t Dim, typename T>
	ndarray_wraparound_view<Dim - 1, T> get_subscript(const ndarray_wraparound_view<Dim, T>& array, std::ptrdiff_t c) {
		return array.slice(c, 0);
	}

	template<typename T>
	T& get_subscript(const ndarray_wraparound_view<1, T>& array, std::ptrdiff_t c) {
		return array.at({c});
	}
};


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
	
	constexpr static std::size_t dimension = Dim;
	
private:
	strides_type wrap_offsets_;
	strides_type wrap_circumferences_;

private:
	using fcall_type = detail::ndarray_view_fcall<ndarray_wraparound_view<Dim, T> ,1>;
	
public:
	/// \name Construction
	///@{
	ndarray_wraparound_view();
	
	ndarray_wraparound_view(
		pointer start,
		const shape_type& shape,
		const strides_type& strides,
		const strides_type& offsets,
		const strides_type& circumferences
	);
	
	ndarray_wraparound_view(const ndarray_wraparound_view<Dim, std::remove_const_t<T>>& vw);
	
	static ndarray_wraparound_view null() const { return ndarray_wraparound_view(); }
	bool is_null() const { return (start() == nullptr); }
	explicit operator bool () const { return ! is_null(); }
	
	template<typename... Args>
	void reset(const Args&... args) { reset(ndarray_wraparound_view(args...)); }
	
	void reset(const ndarray_wraparound_view&);
	///@}
	
	/// \name Attributes
	///@{
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
	std::ptrdiff_t contiguous_length() const { return 0; }
	
	iterator begin() const;
	iterator end() const;
	///@}
	
	
	/// \name Indexing
	///@{
	using base::index_to_coordinates;
	using base::coordinates_to_index;
	pointer coordinates_to_pointer(const coordinates_type&) const;
	
	reference at(const coordinates_type& coord) const;
	
	ndarray_wraparound_view section
		(const coordinates_type& start, const coordinates_type& end, const strides_type& steps = strides_type(1)) const;
	
	ndarray_wraparound_view section(const span_type& span, const strides_type& steps = strides_type(1)) const {
		return section(span.start_pos(), span.end_pos(), steps);
	}
	
	ndarray_wraparound_view<Dim - 1, T> slice(std::ptrdiff_t c, std::ptrdiff_t dimension) const;
	
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
};

}

#include "ndarray_wraparound_view.tcc"

#endif