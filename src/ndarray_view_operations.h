#ifndef TLZ_NDARRAY_VIEW_OPERATIONS_H_
#define TLZ_NDARRAY_VIEW_OPERATIONS_H_

#include "ndarray_view.h"

namespace tlz {

template<typename T>
ndarray_view<2, T> flip(const ndarray_view<2, T>& vw) { return swapaxis(vw, 0, 1); }

template<std::size_t Dim, typename T>
ndarray_view<1 + Dim, T> add_front_axis(const ndarray_view<Dim, T>&);

template<std::size_t Dim, typename T>
ndarray_view<Dim + 1, T> add_back_axis(const ndarray_view<Dim, T>&);

template<std::size_t Dim, typename T>
ndarray_view<Dim, T> swapaxis(const ndarray_view<Dim, T>&, std::ptrdiff_t axis1, std::ptrdiff_t axis2);


template<std::size_t Dim, typename T, std::size_t New_dim>
ndarray_view<New_dim, T> reshape(const ndarray_view<Dim, T>&, const ndsize<New_dim>&);


template<std::size_t Dim, typename T>
ndarray_view<1, T> flatten(const ndarray_view<Dim, T>&);


template<std::size_t Dim, typename T>
ndarray_view<Dim, T> step(const ndarray_view<Dim, T>& vw, std::ptrdiff_t axis, std::ptrdiff_t step) {
	return vw.axis_section(axis, 0, vw.shape()[axis], step);
}

template<typename T>
ndarray_view<1, T> step(const ndarray_view<1, T>& vw, std::ptrdiff_t step) {
	return vw.axis_section(0, 0, vw.shape()[0], step);
}

template<std::size_t Dim, typename T>
ndarray_view<Dim, T> reverse(const ndarray_view<Dim, T>& vw, std::ptrdiff_t axis = 0) {
	return step(vw, axis, -1);
}

template<std::size_t Dim, typename T>
ndarray_view<Dim, T> reverse_all(const ndarray_view<Dim, T>& vw) {
	return vw.section(vw.full_span(), ndptrdiff<Dim>(-1));
}

}

#include "ndarray_view_operations.tcc"

#endif