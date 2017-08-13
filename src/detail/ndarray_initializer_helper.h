#ifndef TLZ_NDARRAY_INITIALIZER_HELPER_H_
#define TLZ_NDARRAY_INITIALIZER_HELPER_H_

namespace tlz { namespace detail {

template<std::size_t Dim, typename Elem>
struct ndarray_initializer_helper {
	using initializer_list_type = std::initializer_list<typename ndarray_initializer_helper<Dim - 1, Elem>::initializer_list_type>;
	
	static ndsize<Dim> shape(const initializer_list_type&);
	static bool is_valid(const initializer_list_type&);
	template<typename View> static void copy_into(const initializer_list_type& init, const View& view);
};


template<typename Elem>
struct ndarray_initializer_helper<0, Elem> {
	using initializer_list_type = Elem;

	static ndsize<0> shape(const initializer_list_type&) { return make_ndsize(); }
	static bool is_valid(const initializer_list_type&) { return true; }
	static void copy_into(const Elem& init, Elem& view) { view = init; }
};

}}

#include "ndarray_initializer_helper.tcc"

#endif