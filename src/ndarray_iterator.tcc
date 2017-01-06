#include "common.h"
#include "ndarray_view.h"

namespace tff {


template<typename View>
inline void ndarray_iterator<View>::forward_(std::ptrdiff_t d) {
	Assert_crit(d >= 0);
	std::ptrdiff_t contiguous_limit = view_.contiguous_length() - (index_ % view_.contiguous_length());
	index_ += d;
	if(d < contiguous_limit) {
		pointer_ = advance_raw_ptr(pointer_, d * view_.strides().back());
	} else {
		auto new_coord = view_.index_to_coordinates(index_);
		pointer_ = view_.coordinates_to_pointer(new_coord);
	}
}


template<typename View>
inline void ndarray_iterator<View>::backward_(std::ptrdiff_t d) {
	Assert_crit(d >= 0);
	std::ptrdiff_t contiguous_limit = index_ % view_.contiguous_length();
	index_ -= d;
	if(d <= contiguous_limit) {
		pointer_ = advance_raw_ptr(pointer_, -d * view_.strides().back());
	} else {
		auto new_coord = view_.index_to_coordinates(index_);
		pointer_ = view_.coordinates_to_pointer(new_coord);
	}
}


template<typename View>
ndarray_iterator<View>::ndarray_iterator(const view_type& vw, index_type index, pointer ptr) :
	view_(vw),
	pointer_(ptr),
	index_(index) { }


template<typename View>
auto ndarray_iterator<View>::operator=(const ndarray_iterator& it) -> ndarray_iterator& {
	pointer_ = it.pointer_;
	index_ = it.index_;
	return *this;
}


template<typename View>
auto ndarray_iterator<View>::operator++() -> ndarray_iterator& {
	forward_(1);
	return *this;
}


template<typename View>
auto ndarray_iterator<View>::operator++(int) -> ndarray_iterator {
	auto copy = *this;
	forward_(1);
	return copy;
}


template<typename View>
auto ndarray_iterator<View>::operator--() -> ndarray_iterator& {
	backward_(1);
	return *this;
}


template<typename View>
auto ndarray_iterator<View>::operator--(int) -> ndarray_iterator {
	auto copy = *this;
	backward_(1);
	return copy;
}


template<typename View>
auto ndarray_iterator<View>::operator+=(std::ptrdiff_t n) -> ndarray_iterator& {
	if(n > 0) forward_(n);
	else backward_(-n);
	return *this;
}


template<typename View>
auto ndarray_iterator<View>::operator-=(std::ptrdiff_t n) -> ndarray_iterator& {
	if(n > 0) backward_(n);
	else forward_(-n);
	return *this;
}


}
