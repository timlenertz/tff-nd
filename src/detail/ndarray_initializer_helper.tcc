#include "ndarray_initializer_helper.h"

namespace tlz { namespace detail {

template<std::size_t Dim, typename Elem>
ndsize<Dim> ndarray_initializer_helper<Dim, Elem>::shape(const initializer_list_type& init) {
	Assert(init.size() > 0, "ndarray initializer_list cannot be empty");
	ndsize<Dim - 1> nested_shape = ndarray_initializer_helper<Dim - 1, Elem>::shape(*init.begin());
	return ndcoord_cat(init.size(), nested_shape);
}
		
template<std::size_t Dim, typename Elem>
bool ndarray_initializer_helper<Dim, Elem>::is_valid(const initializer_list_type& init) {
	auto it = init.begin();
	ndsize<Dim - 1> first_nested_shape = ndarray_initializer_helper<Dim - 1, Elem>::shape(*it);
	if(! ndarray_initializer_helper<Dim - 1, Elem>::is_valid(*it)) return false;
	for(++it; it != init.end(); ++it) {
		ndsize<Dim - 1> nested_shape = ndarray_initializer_helper<Dim - 1, Elem>::shape(*it);
		if(nested_shape != first_nested_shape) return false;
		if(! ndarray_initializer_helper<Dim - 1, Elem>::is_valid(*it)) return false;
	}
	return true;
}

template<std::size_t Dim, typename Elem> template<typename View>
void ndarray_initializer_helper<Dim, Elem>::copy_into(const initializer_list_type& init, const View& view) {
	std::ptrdiff_t view_slice_index = 0;
	for(auto init_it = init.begin(); init_it != init.end(); ++init_it, ++view_slice_index)
		ndarray_initializer_helper<Dim - 1, Elem>::copy_into(*init_it, view[view_slice_index]);
}

}}