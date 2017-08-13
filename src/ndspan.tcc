#include "ndspan_iterator.h"

namespace tlz {

template<std::size_t Dim, typename T>
bool ndspan<Dim, T>::invariant_() const {
	for(std::ptrdiff_t i = 0; i < Dim; ++i) if(end_[i] < start_[i]) return false;
	return true;
}



template<std::size_t Dim, typename T>
auto ndspan<Dim, T>::begin() const -> iterator {
	return iterator(*this, start_);
}


template<std::size_t Dim, typename T>
auto ndspan<Dim, T>::end() const -> iterator {
	coordinates_type iterator_end_coord = start_;
	iterator_end_coord.front() = end_.front();
	return iterator(*this, iterator_end_coord);
}



template<std::size_t Dim, typename T>
bool ndspan<Dim, T>::includes(const coordinates_type& c) const {
	for(std::ptrdiff_t i = 0; i < Dim; ++i)
		if( (start_[i] > c[i]) || (end_[i] <= c[i]) ) return false;
	return true;
}


template<std::size_t Dim, typename T>
bool ndspan<Dim, T>::includes(const ndspan& sub) const {
	for(std::ptrdiff_t i = 0; i < Dim; ++i)
		if( (start_[i] > sub.start_[i]) || (end_[i] < sub.end_[i]) ) return false;
	return true;
}


template<std::size_t Dim, typename T>
bool ndspan<Dim, T>::includes_strict(const ndspan& sub) const {
	for(std::ptrdiff_t i = 0; i < Dim; ++i)
		if( (start_[i] >= sub.start_[i]) || (end_[i] <= sub.end_[i]) ) return false;
	return true;
}


template<std::size_t Dim, typename T>
ndspan<Dim, T> span_intersection(const ndspan<Dim, T>& a, const ndspan<Dim, T>& b) {
	ndcoord<Dim, T> new_start, new_end;
	for(std::ptrdiff_t i = 0; i < Dim; ++i) {
		new_start[i] = std::max(a.start_pos()[i], b.start_pos()[i]);
		new_end[i] = std::min(a.end_pos()[i], b.end_pos()[i]);
		if(new_end[i] < new_start[i])
			new_start[i] = new_end[i] = 0;
	}
	return ndspan<Dim, T>(new_start, new_end);

}


}
