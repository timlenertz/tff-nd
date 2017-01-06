#include "common.h"

namespace tff {

template<std::size_t Dim, typename T>
ndarray_wraparound_view<Dim, T>::ndarray_wraparound_view(
	pointer start,
	const shape_type& shape,
	const strides_type& strides,
	const strides_type& offsets,
	const strides_type& circumferences
) :
	base(start, shape, strides),
	wrap_offsets_(offsets),
	wrap_circumferences_(circumferences)
{
	base::contiguous_length_ = 0;
}


template<std::size_t Dim, typename T>
ndarray_wraparound_view<Dim, T>::ndarray_wraparound_view
(const ndarray_wraparound_view<Dim, std::remove_const_t<T>>& vw) :
	base(vw),
	wrap_offsets_(vw.wrap_offsets()),
	wrap_circumferences_(vw.wrap_circumferences()) { }


template<std::size_t Dim, typename T>
void ndarray_wraparound_view<Dim, T>::reset(const ndarray_wraparound_view& vw) {
	base::reset(vw);
	wrap_offsets_ = vw.wrap_offsets_;
	wrap_circumferences_ = vw.wrap_circumferences_;
}


template<std::size_t Dim, typename T> template<typename Other_view>
std::enable_if_t<is_ndarray_view<Other_view>> ndarray_wraparound_view<Dim, T>::assign(const Other_view& other) const {
	Assert_crit(shape() == other.shape(), "ndarray_view must have same shape for assignment");
	if(shape().product() == 0) return;
	std::copy(other.begin(), other.end(), begin());
}


template<std::size_t Dim, typename T> template<typename Other_view>
std::enable_if_t<is_ndarray_view<Other_view>, bool> ndarray_wraparound_view<Dim, T>::compare(const Other_view& other) const {
	if(shape() != other.shape()) return false;
	else return std::equal(other.begin(), other.end(), begin());
}


template<std::size_t Dim, typename T>
auto ndarray_wraparound_view<Dim, T>::coordinates_to_pointer(const coordinates_type& coord) const -> pointer {
	pointer ptr = base::start_;
	for(std::ptrdiff_t i = 0; i < Dim; ++i) {
		std::ptrdiff_t dist = base::strides_[i] * coord[i];
		dist = positive_modulo(dist + wrap_offsets_[i], wrap_circumferences_[i]) - wrap_offsets_[i];
		ptr = advance_raw_ptr(ptr, dist);
	}
	return ptr;
}


template<std::size_t Dim, typename T>
auto ndarray_wraparound_view<Dim, T>::at(const coordinates_type& coord) const -> reference {
	coordinates_type real_coord;
	for(std::ptrdiff_t i = 0; i < Dim; ++i) real_coord[i] = fix_coordinate_(coord[i], i);
	return *coordinates_to_pointer(real_coord);
}


template<std::size_t Dim, typename T>
auto ndarray_wraparound_view<Dim, T>::section_
(std::ptrdiff_t i, std::ptrdiff_t start, std::ptrdiff_t end, std::ptrdiff_t step) const -> ndarray_wraparound_view {
	if(start < 0) start = base::shape_[i] + start;
	if(end < 0) end = base::shape_[i] + end;
		
	Assert_crit(start >= 0 && start <= base::shape_[i], "section start range");
	Assert_crit(end >= 0 && end <= base::shape_[i], "section end range");
	Assert_crit(start < end, "section start must be lower than end");
	Assert_crit(step != 0, "section step must not be zero");
	
	shape_type new_shape = base::shape_;
	strides_type new_strides = base::strides_;
	
	std::ptrdiff_t n = end - start;
	std::ptrdiff_t rel_start;
	new_strides[i] = base::strides_[i] * step;
	
	if(step > 0) {
		new_shape[i] = 1 + ((n - 1) / step);
		rel_start = start;
		
	} else {
		new_shape[i] = 1 + ((n - 1) / -step);
		rel_start = start - (step * (new_shape[i]-1));
	}
	
	std::ptrdiff_t start_diff = base::strides_[i] * rel_start;
	start_diff = positive_modulo(start_diff + wrap_offsets_[i], wrap_circumferences_[i]) - wrap_offsets_[i];
	
	pointer new_start = advance_raw_ptr(base::start_, start_diff);
	
	strides_type section_wrap_offsets = wrap_offsets_;
	section_wrap_offsets[i] = positive_modulo(section_wrap_offsets[i] + start_diff, wrap_circumferences_[i]);
	
	return ndarray_wraparound_view(new_start, new_shape, new_strides, section_wrap_offsets, wrap_circumferences_);
}


template<std::size_t Dim, typename T>
ndarray_wraparound_view<Dim, T> ndarray_wraparound_view<Dim, T>::section
(const coordinates_type& start, const coordinates_type& end, const strides_type& steps) const {
	ndarray_wraparound_view new_view = *this;
	for(std::ptrdiff_t i = 0; i < Dim; ++i) {
		auto sec = new_view.section_(i, start[i], end[i], steps[i]);
		new_view.reset(sec);
	}
	return new_view;
}



template<std::size_t Dim, typename T>
ndarray_wraparound_view<Dim - 1, T> ndarray_wraparound_view<Dim, T>::slice
(std::ptrdiff_t c, std::ptrdiff_t dimension) const {
	return ndarray_wraparound_view<Dim - 1, T>(
		advance_raw_ptr(base::start_, base::strides_[dimension] * fix_coordinate_(c, dimension)),
		base::shape_.erase(dimension),
		base::strides_.erase(dimension),
		wrap_offsets_.erase(dimension),
		wrap_circumferences_.erase(dimension)
	);
}


template<std::size_t Dim, typename T>
inline auto ndarray_wraparound_view<Dim, T>::begin() const -> iterator {
	return iterator(*this, 0, base::start_);
}


template<std::size_t Dim, typename T>
auto ndarray_wraparound_view<Dim, T>::end() const -> iterator {
	index_type end_index = base::shape().product();
	coordinates_type end_coord = index_to_coordinates(end_index);
	return iterator(*this, end_index, coordinates_to_pointer(end_coord));
}



};