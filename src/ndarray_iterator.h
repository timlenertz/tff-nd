#ifndef TFF_NDARRAY_ITERATOR_H_
#define TFF_NDARRAY_ITERATOR_H_

#include <iterator>

namespace tff {

template<std::size_t Dim, typename Elem>
class ndarray_view;

/// Random access iterator which traverses an \ref ndarray_view.
/** Always traverses the elements in order of increasing index, regardless of strides. Random-access iterator operations
 ** (addition, comparation, etc.) act on index values. Index and coordinates of current item can be accessed using
 ** index() and coordinates().
 ** Incrementation and decrementation are optimized when the strides of the `ndarray_view` are (partially) default, i.e.
 ** items with sequential indices have sequential memory addresses, possibly with padding inbetween. This can be for the
 ** entire `ndarray_view`, or just for smaller segments at a time.
 ** If `coordinates()` is called at each iteration, using `ndspan_iterator` may be more efficient because it does not
 ** recompute the coordinates from an index each time. */
template<typename View>
class ndarray_iterator {
	//static_assert(is_ndarray_view<View>, "ndarray_iterator template argument must be ndarray view type");
	
public:
	using view_type = View;
	using value_type = typename view_type::value_type;
	
	using iterator_category = std::random_access_iterator_tag;
	using difference_type = std::ptrdiff_t;
	using pointer = value_type*;
	using reference = value_type&;
	
	using index_type = typename view_type::index_type;
	using coordinates_type = typename view_type::coordinates_type;

private:
	const view_type view_;
	const std::ptrdiff_t contiguous_length_;
	pointer pointer_ = nullptr;
	index_type index_ = 0;
	
	
	void forward_(std::ptrdiff_t);
	void backward_(std::ptrdiff_t);

public:
	ndarray_iterator() = default;
	ndarray_iterator(const view_type& vw, index_type index, pointer ptr);
	ndarray_iterator(const ndarray_iterator&) = default;
	
	ndarray_iterator& operator=(const ndarray_iterator&);

	const view_type& view() const { return view_; }
	
	index_type index() const { return index_; }
	coordinates_type coordinates() const { return view_.index_to_coordinates(index_); }

	pointer ptr() const { return pointer_; }
	reference operator*() const { return *pointer_; }
	pointer operator->() const { return pointer_; }
	reference operator[](std::ptrdiff_t n) const { return *(*this + n); }

	ndarray_iterator& operator++();
	ndarray_iterator operator++(int);
	ndarray_iterator& operator--();
	ndarray_iterator operator--(int);
	
	ndarray_iterator& operator+=(std::ptrdiff_t);
	ndarray_iterator& operator-=(std::ptrdiff_t);
	
	friend bool operator==(const ndarray_iterator& a, const ndarray_iterator& b) noexcept
		{ return a.index() == b.index(); }
	friend bool operator!=(const ndarray_iterator& a, const ndarray_iterator& b) noexcept
		{ return a.index() != b.index(); }
	friend bool operator<(const ndarray_iterator& a, const ndarray_iterator& b) noexcept
		{ return a.index() < b.index(); }
	friend bool operator<=(const ndarray_iterator& a, const ndarray_iterator& b) noexcept
		{ return a.index() <= b.index(); }
	friend bool operator>(const ndarray_iterator& a, const ndarray_iterator& b) noexcept
		{ return a.index() > b.index(); }
	friend bool operator>=(const ndarray_iterator& a, const ndarray_iterator& b) noexcept
		{ return a.index() >= b.index(); }
	
	friend ndarray_iterator operator+(const ndarray_iterator& it, std::ptrdiff_t n)
		{ auto copy = it; copy += n; return copy; }
	friend ndarray_iterator operator+(std::ptrdiff_t n, const ndarray_iterator& it)
		{ auto copy = it; copy += n; return copy; }
	friend ndarray_iterator operator-(const ndarray_iterator& it, std::ptrdiff_t n)
		{ auto copy = it; copy -= n; return copy; }
	friend std::ptrdiff_t operator-(const ndarray_iterator& a, const ndarray_iterator& b)
		{ return a.index() - b.index(); }
};


}

#include "ndarray_iterator.tcc"

#endif
