#include "ndarray_view_operations.h"

namespace tff {


template<std::size_t Dim, typename T>
ndarray_view<1 + Dim, T> add_front_axis(const ndarray_view<Dim, T>& vw) {
	auto new_shape = ndcoord_cat(1, vw.shape());
	auto new_strides = ndcoord_cat(0, vw.strides());
	return ndarray_view<1 + Dim, T>(vw.start(), new_shape, new_strides);
}


template<std::size_t Dim, typename T>
ndarray_view<Dim + 1, T> add_front_axis(const ndarray_view<Dim, T>& vw) {
	auto new_shape = ndcoord_cat(vw.shape(), 1);
	auto new_strides = ndcoord_cat(vw.strides(), 0);
	return ndarray_view<Dim + 1, T>(vw.start(), new_shape, new_strides);
}


template<std::size_t Dim, typename T>
ndarray_view<Dim, T> swapaxis(const ndarray_view<Dim, T>& vw, std::ptrdiff_t axis1, std::ptrdiff_t axis2) {
	Assert_crit(axis1 >= 0 && axis1 < vw.dimension());
	Assert_crit(axis2 >= 0 && axis2 < vw.dimension());
	auto new_strides = vw.strides();
	auto new_shape = vw.shape();
	std::swap(new_strides[axis1], new_strides[axis2]);
	std::swap(new_shape[axis1], new_shape[axis2]);
	return ndarray_view<Dim, T>(vw.start(), new_shape, new_strides);
}


template<std::size_t Dim, typename T, std::size_t New_dim>
ndarray_view<New_dim, T> reshape(const ndarray_view<Dim, T>& vw, const ndsize<New_dim>& new_shape) {
	Assert_crit(vw.strides() == vw.default_strides(vw.shape()), "can reshape only with default strides");
	Assert_crit(new_shape.product() == vw.shape().product(), "new shape must have same product");
	return ndarray_view<New_dim, T>(vw.start(), new_shape);
}


template<std::size_t Dim, typename T>
ndarray_view<1, T> flatten(const ndarray_view<Dim, T>& vw) {
	auto new_shape = make_ndsize(vw.shape().product());
	return reshape(vw, new_shape);
}

}