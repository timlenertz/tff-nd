#include "common.h"

namespace tlz {

template<std::size_t Dim, typename T>
ndarray_wraparound_view<Dim, T> wraparound(
	const ndarray_view<Dim, T>& vw,
	const ndptrdiff<Dim>& start_pos,
	const ndptrdiff<Dim>& end_pos,
	const ndptrdiff<Dim>& steps
) {
	using view_type = ndarray_wraparound_view<Dim, T>;
	using pointer = typename view_type::pointer;
	using shape_type = typename view_type::shape_type;
	using strides_type = typename view_type::strides_type;
	
	pointer new_start = vw.start();
	shape_type new_shape;
	strides_type new_strides;
	strides_type wrap_offsets;
	strides_type wrap_circumferences;
	
	for(std::ptrdiff_t axis = 0; axis < Dim; ++axis) {
		std::ptrdiff_t start = start_pos[axis];
		std::ptrdiff_t end = end_pos[axis];
		std::ptrdiff_t step = steps[axis];
			
		Assert_crit(start < end, "section start must be lower than end");
		Assert_crit(step != 0, "section step must not be zero");
		
		// start of wraparound section must be inside span of this view
		std::ptrdiff_t n = end - start;
		std::ptrdiff_t rel_start;
		
		wrap_circumferences[axis] = vw.shape()[axis] * vw.strides()[axis];
		new_strides[axis] = vw.strides()[axis] * step;
		
		if(step > 0) {
			new_shape[axis] = 1 + ((n - 1) / step);
			rel_start = positive_modulo(start, vw.shape()[axis]);
		} else {
			new_shape[axis] = 1 + ((n - 1) / -step);
			rel_start = positive_modulo(start - (step * (new_shape[axis]-1)), vw.shape()[axis]);
		}
		
		new_start = advance_raw_ptr(new_start, vw.strides()[axis] * rel_start);
		wrap_offsets[axis] = rel_start * vw.strides()[axis];
		
		// TODO negative wrap_circumferences (if this->strides < 0)
		// seems to work (-> tests)
		// but same() no longer correct
	}
	
	return view_type(
		new_start,
		new_shape,
		new_strides,
		wrap_offsets,
		wrap_circumferences
	);
}


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
	wrap_circumferences_(circumferences) { }


template<std::size_t Dim, typename T>
ndarray_wraparound_view<Dim, T>::ndarray_wraparound_view
(const ndarray_wraparound_view<Dim, std::remove_const_t<T>>& vw) :
	base(vw.non_wraparound()),
	wrap_offsets_(vw.wrap_offsets()),
	wrap_circumferences_(vw.wrap_circumferences()) { }


template<std::size_t Dim, typename T>
ndarray_wraparound_view<Dim, T>::ndarray_wraparound_view
(const ndarray_view<Dim, std::remove_const_t<T>>& vw) :
	base(vw),
	wrap_offsets_(0),
	wrap_circumferences_(vw.shape() * vw.strides()) { }


template<std::size_t Dim, typename T>
void ndarray_wraparound_view<Dim, T>::reset(const ndarray_wraparound_view& vw) {
	base::reset(vw.non_wraparound());
	wrap_offsets_ = vw.wrap_offsets_;
	wrap_circumferences_ = vw.wrap_circumferences_;
}


template<std::size_t Dim, typename T> template<typename Other_view>
auto ndarray_wraparound_view<Dim, T>::assign(const Other_view& other) const -> enable_if_convertible_<Other_view> {
	Assert_crit(shape() == other.shape(), "ndarray_view must have same shape for assignment");
	if(shape().product() == 0) return;
	std::copy(other.begin(), other.end(), begin());
}


template<std::size_t Dim, typename T>
void ndarray_wraparound_view<Dim, T>::assign(initializer_list_type init) const {
	Assert(initializer_helper_type::is_valid(init), "initializer_list must be valid");
	Assert(initializer_helper_type::shape(init) == shape(), "initializer_list to assign from has different shape");
	initializer_helper_type::copy_into(init, *this);
}


template<std::size_t Dim, typename T> template<typename Other_view>
auto ndarray_wraparound_view<Dim, T>::compare(const Other_view& other) const -> enable_if_convertible_<Other_view, bool> {
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
auto ndarray_wraparound_view<Dim, T>::axis_section
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
		auto sec = new_view.axis_section(i, start[i], end[i], steps[i]);
		new_view.reset(sec);
	}
	return new_view;
}



template<std::size_t Dim, typename T>
ndarray_wraparound_view<Dim - 1, T> ndarray_wraparound_view<Dim, T>::slice
(std::ptrdiff_t pos, std::ptrdiff_t axis) const {
	if(pos < 0) pos = base::shape_[axis] + pos;
		
	Assert_crit(pos >= 0 && pos <= base::shape_[axis], "position range");
	
	std::ptrdiff_t start_diff = base::strides_[axis] * pos;
	start_diff = positive_modulo(start_diff + wrap_offsets_[axis], wrap_circumferences_[axis]) - wrap_offsets_[axis];
	pointer new_start = advance_raw_ptr(base::start_, start_diff);
	
	return ndarray_wraparound_view<Dim - 1, T>(
		new_start,
		base::shape().erase(axis),
		base::strides().erase(axis),
		wrap_offsets_.erase(axis),
		wrap_circumferences_.erase(axis)
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


template<std::size_t Dim, typename T1, typename T2>
bool same(const ndarray_wraparound_view<Dim, T1>& a, const ndarray_wraparound_view<Dim, T2>& b) {
	if(a.is_null() && b.is_null()) return true;
	else return (a.start() == b.start()) && (a.shape() == b.shape()) && (a.strides() == b.strides()) && (a.wrap_offsets() == b.wrap_offsets()) && (a.wrap_circumferences() == b.wrap_circumferences());
}


template<std::size_t Dim, typename T>
ndarray_wraparound_view<Dim, T> swapaxis(const ndarray_wraparound_view<Dim, T>& vw, std::ptrdiff_t axis1, std::ptrdiff_t axis2) {
	Assert_crit(axis1 >= 0 && axis1 < vw.dimension());
	Assert_crit(axis2 >= 0 && axis2 < vw.dimension());
	auto new_strides = vw.strides(); std::swap(new_strides[axis1], new_strides[axis2]);
	auto new_shape = vw.shape(); std::swap(new_shape[axis1], new_shape[axis2]);
	auto new_wrap_offsets = vw.wrap_offsets(); std::swap(new_wrap_offsets[axis1], new_wrap_offsets[axis2]);
	auto new_wrap_circumferences = vw.wrap_offsets(); std::swap(new_wrap_circumferences[axis1], new_wrap_circumferences[axis2]);
	return ndarray_wraparound_view<Dim, T>(vw.start(), new_shape, new_strides, new_wrap_offsets, new_wrap_circumferences);
}



}
