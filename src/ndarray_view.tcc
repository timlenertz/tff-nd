#include <cstdlib>
#include <algorithm>
#include <type_traits>
#include "common.h"
#include "detail/ndarray_initializer_helper.h"
#include <iostream>

namespace tlz {


template<std::size_t Dim, typename T>
auto ndarray_view<Dim, T>::default_strides(const shape_type& shape, std::size_t padding) -> strides_type {
	if(Dim == 0) return ndptrdiff<Dim>();
	Assert(is_multiple_of(padding, alignof(T)));
	strides_type strides;
	strides[Dim - 1] = sizeof(T) + padding;
	for(std::ptrdiff_t i = Dim - 1; i > 0; --i)
		strides[i - 1] = strides[i] * shape[i];
	return strides;
}



template<std::size_t Dim, typename T>
bool ndarray_view<Dim, T>::has_default_strides(std::ptrdiff_t minimal_dimension) const {
	if(Dim == 0) return true;
	if(strides_.back() < sizeof(T)) return false;
	for(std::ptrdiff_t i = Dim - 2; i >= minimal_dimension; --i) {
		std::ptrdiff_t expected_stride = shape_[i + 1] * strides_[i + 1];
		if(strides_[i] != expected_stride) return false;
	}
	return true;
}


template<std::size_t Dim, typename T>
bool ndarray_view<Dim, T>::has_default_strides_without_padding(std::ptrdiff_t minimal_dimension) const {
	if(Dim == 0) return true;
	else if(has_default_strides(minimal_dimension)) return (default_strides_padding(minimal_dimension) == 0);
	else return false;
}




template<std::size_t Dim, typename T>
std::size_t ndarray_view<Dim, T>::default_strides_padding(std::ptrdiff_t minimal_dimension) const {
	if(Dim == 0) return 0;
	Assert(has_default_strides(minimal_dimension));
	return (strides_.back() - sizeof(T));
}




template<std::size_t Dim, typename T>
ndarray_view<Dim, T>::ndarray_view(pointer start, const shape_type& shape) :
	ndarray_view(start, shape, default_strides(shape)) { }



template<std::size_t Dim, typename T>
ndarray_view<Dim, T>::ndarray_view(pointer start, const shape_type& shape, const strides_type& strides) :
	start_(start), shape_(shape), strides_(strides) { }


template<std::size_t Dim, typename T>
std::ptrdiff_t ndarray_view<Dim, T>::fix_coordinate_(std::ptrdiff_t c, std::ptrdiff_t dimension) const {
	std::ptrdiff_t n = shape_[dimension];
	if(c < 0) c = n + c;
	Assert_crit(c < n, "coordinate is out of range");
	return c;
}


template<std::size_t Dim, typename T>
auto ndarray_view<Dim, T>::index_to_coordinates(index_type index) const -> coordinates_type {	
	if(index == 0) return coordinates_type(0);
	coordinates_type coord;
	index_type remainder = index;
	std::size_t factor = shape_.tail().product();
	for(std::ptrdiff_t i = 0; i < Dim - 1; ++i) {
		auto div = std::div(remainder, factor);
		coord[i] = div.quot;
		remainder = div.rem;
		factor /= shape_[i + 1];
	}
	coord.back() = remainder;
	return coord;
}


template<std::size_t Dim, typename T>
auto ndarray_view<Dim, T>::coordinates_to_index(const coordinates_type& coord) const -> index_type {
	std::ptrdiff_t index = coord.back();
	std::ptrdiff_t factor = shape_[Dim - 1];
	for(std::ptrdiff_t i = Dim - 2; i >= 0; --i) {
		index += factor * coord[i];
		factor *= shape_[i];
	}
	return index;
}


template<std::size_t Dim, typename T>
void ndarray_view<Dim, T>::reset(const ndarray_view& other) {
	start_ = other.start_;
	shape_ = other.shape_;
	strides_ = other.strides_;
}


template<std::size_t Dim, typename T> template<typename Other_view>
auto ndarray_view<Dim, T>::assign(const Other_view& other) const -> enable_if_convertible_<Other_view> {
	static_assert(! std::is_const<value_type>::value, "cannot assign to const ndarray_view");
	
	using elem_type = std::remove_cv_t<value_type>;
	using other_elem_type = std::remove_cv_t<typename Other_view::value_type>;
	if(std::is_same<elem_type, other_elem_type>::value && has_pod_format() && other.has_pod_format() && pod_format() == other.pod_format()) {
		// optimize when possible
		pod_array_copy(static_cast<void*>(start()), static_cast<const void*>(other.start()), pod_format());
	} else {
		Assert_crit(shape() == other.shape(), "ndarray_view must have same shape for assignment");
		if(shape().product() == 0) return;
		std::copy(other.begin(), other.end(), begin());
	}
}


template<std::size_t Dim, typename T>
void ndarray_view<Dim, T>::assign(initializer_list_type init) const {
	Assert(initializer_helper_type::is_valid(init), "initializer_list must be valid");
	Assert(initializer_helper_type::shape(init) == shape(), "initializer_list to assign from has different shape");
	initializer_helper_type::copy_into(init, *this);
}


template<std::size_t Dim, typename T>
void ndarray_view<Dim, T>::fill(const value_type& val) const {
	static_assert(! std::is_const<value_type>::value, "cannot assign to const ndarray_view");
	std::fill(begin(), end(), val);
}


template<std::size_t Dim, typename T> template<typename Other_view>
auto ndarray_view<Dim, T>::compare(const Other_view& other) const -> enable_if_convertible_<Other_view, bool> {
	if(shape() != other.shape()) return false;
	//else if(same(*this, other)) return true;

	using elem_type = std::remove_cv_t<value_type>;
	using other_elem_type = std::remove_cv_t<typename Other_view::value_type>;
	if(std::is_same<elem_type, other_elem_type>::value && has_pod_format() && other.has_pod_format() && pod_format() == other.pod_format()) {
		return pod_array_compare(static_cast<const void*>(start()), static_cast<const void*>(other.start()), pod_format());
	} else {
		return std::equal(other.begin(), other.end(), begin());
	}
}


template<std::size_t Dim, typename T>
auto ndarray_view<Dim, T>::coordinates_to_pointer(const coordinates_type& coord) const -> pointer {
	pointer ptr = start_;
	for(std::ptrdiff_t i = 0; i < Dim; ++i)
		ptr = advance_raw_ptr(ptr, strides_[i] * coord[i]);
	return ptr;
}


template<std::size_t Dim, typename T>
auto ndarray_view<Dim, T>::at(const coordinates_type& coord) const -> reference {
	coordinates_type real_coord;
	for(std::ptrdiff_t i = 0; i < Dim; ++i) real_coord[i] = fix_coordinate_(coord[i], i);
	return *coordinates_to_pointer(real_coord);
}


template<std::size_t Dim, typename T>
auto ndarray_view<Dim, T>::axis_section(std::ptrdiff_t i, std::ptrdiff_t start, std::ptrdiff_t end, std::ptrdiff_t step) const -> ndarray_view {
	if(start < 0) start = shape_[i] + start;
	if(end < 0) end = shape_[i] + end;
	
	Assert_crit(start >= 0 && start <= shape_[i], "section start range");
	Assert_crit(end >= 0 && end <= shape_[i], "section end range");
	Assert_crit(start < end, "section start must be lower than end");
	Assert_crit(step != 0, "section step must not be zero");
	
	shape_type new_shape = shape_;
	strides_type new_strides = strides_;
	
	std::ptrdiff_t n = end - start;
	std::ptrdiff_t rel_start;
	new_strides[i] = strides_[i] * step;
	
	if(step > 0) {
		new_shape[i] = 1 + ((n - 1) / step);
		rel_start = start;
		
	} else {
		new_shape[i] = 1 + ((n - 1) / -step);
		rel_start = start - (step * (new_shape[i]-1));
	}
	
	pointer new_start = advance_raw_ptr(start_, strides_[i] * rel_start);

	return ndarray_view(new_start, new_shape, new_strides);
}


template<std::size_t Dim, typename T>
auto ndarray_view<Dim, T>::section(const coordinates_type& start, const coordinates_type& end, const strides_type& steps) const -> ndarray_view {
	ndarray_view new_view = *this;
	for(std::ptrdiff_t i = 0; i < Dim; ++i) {
		auto sec = new_view.axis_section(i, start[i], end[i], steps[i]);
		new_view.reset(sec);
	}
	return new_view;
}


template<std::size_t Dim, typename T>
auto ndarray_view<Dim, T>::slice(std::ptrdiff_t c, std::ptrdiff_t dimension) const -> ndarray_view<Dim - 1, T> {
	return ndarray_view<Dim - 1, T>(
		advance_raw_ptr(start_, strides_[dimension] * fix_coordinate_(c, dimension)),
		shape_.erase(dimension),
		strides_.erase(dimension)
	);
}


template<std::size_t Dim, typename T>
std::ptrdiff_t ndarray_view<Dim, T>::contiguous_length() const {
	std::ptrdiff_t i;
	std::ptrdiff_t contiguous_len = shape_.back();
	for(i = Dim - 1; i > 0; i--) {
		if(strides_[i - 1] == shape_[i] * strides_[i]) contiguous_len *= shape_[i - 1];
		else break;
	}
	return contiguous_len;
}


template<std::size_t Dim, typename T>
inline auto ndarray_view<Dim, T>::begin() const -> iterator {
	return iterator(*this, 0, start_);
}


template<std::size_t Dim, typename T>
auto ndarray_view<Dim, T>::end() const -> iterator {
	index_type end_index = shape().product();
	coordinates_type end_coord = index_to_coordinates(end_index);
	return iterator(*this, end_index, coordinates_to_pointer(end_coord));
}


template<std::size_t Dim, typename T1, typename T2>
bool same(const ndarray_view<Dim, T1>& a, const ndarray_view<Dim, T2>& b) {
	if(a.is_null() && b.is_null()) return true;
	else return (a.start() == b.start()) && (a.shape() == b.shape()) && (a.strides() == b.strides());
}

}
