#ifndef TFF_NDARRAY_OPAQUE_VIEW_DERIVED_H_
#define TFF_NDARRAY_OPAQUE_VIEW_DERIVED_H_

namespace tff { namespace detail {

template<std::size_t Dim, bool Mutable, typename Frame_format, template<std::size_t,typename> Concrete_view>
class ndarray_opaque_view : private Concrete_view<Dim + 1, const_if<!Mutable, byte>> {
	using base = Concrete_view<Dim + 1, const_if<!Mutable, byte>>;
	
public:

};

}}

#endif
