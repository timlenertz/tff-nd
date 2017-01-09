#ifndef TFF_NDARRAY_OPAQUE_VIEW_CAST_H_
#define TFF_NDARRAY_OPAQUE_VIEW_CAST_H_

#include "../config.h"
#if TFF_ND_WITH_OPAQUE

#include "ndarray_opaque_view.h"
#include "../ndarray_view.h"

namespace tff {

template<std::size_t Opaque_dim, typename Opaque_frame_format, std::size_t Concrete_dim, typename Concrete_elem>
auto to_opaque(const ndarray_view<Concrete_dim, Concrete_elem>& concrete_view, const Opaque_frame_format& frm) {
	static_assert(Opaque_dim <= Concrete_dim,
				  "opaque dimension must be lower or equal to concrete dimension");
	constexpr std::size_t frame_dim = Concrete_dim - Opaque_dim;
	constexpr bool opaque_mutable = ! std::is_const<Concrete_elem>::value;
	
	using opaque_view_type = ndarray_opaque_view<Opaque_dim, opaque_mutable, Opaque_frame_format>;
	if(concrete_view.is_null()) return opaque_view_type::null();
	
	if(frm.is_ndarray()) {
		Assert(frm.elem_size() == sizeof(Concrete_elem));
		Assert(is_multiple_of(alignof(Concrete_elem), frm.elem_alignment_requirement()));
		Assert(frm.dimension() == frame_dim);
		Assert(frm.shape() == tail<frame_dim>(concrete_view.shape()));
		Assert(frm.elem_stride() - frm.elem_size() == concrete_view.default_strides_padding(Opaque_dim));
	}
	
	using frame_pointer_type = typename opaque_view_type::frame_pointer_type;
	auto new_start = reinterpret_cast<frame_pointer_type>(concrete_view.start());
	auto new_shape = head<Opaque_dim>(concrete_view.shape());
	auto new_strides = head<Opaque_dim>(concrete_view.strides());
		
	Assert(frm.size() <= new_strides.back());
	Assert(is_multiple_of(frm.alignment_requirement(), alignof(Concrete_elem)));
	
	return opaque_view_type(new_start, new_shape, new_strides, frm);
}


template<std::size_t Opaque_dim, std::size_t Concrete_dim, typename Concrete_elem>
auto to_opaque(const ndarray_view<Concrete_dim, Concrete_elem>& concrete_view) {
		Assert(concrete_view.has_default_strides());
	opaque_ndarray_format frame_format(
		sizeof(Concrete_elem),
		alignof(Concrete_elem),
		concrete_view.strides().back(),
		std::is_pod<Concrete_elem>::value,
		tail<Concrete_dim - Opaque_dim>(concrete_view.shape())
	);
	return to_opaque<Opaque_dim, opaque_ndarray_format>(concrete_view, frame_format);
}


template<std::size_t Concrete_dim, typename Concrete_elem, std::size_t Opaque_dim, bool Opaque_mutable, typename Opaque_frame_format>
auto from_opaque(
	const ndarray_opaque_view<Opaque_dim, Opaque_mutable, Opaque_frame_format>& opaque_view
) {
	using concrete_view_type = ndarray_view<Concrete_dim, Concrete_elem>;
	if(opaque_view.is_null()) return ndarray_view<Concrete_dim, Concrete_elem>::null();
	
	constexpr std::size_t frame_dim = Concrete_dim - Opaque_dim;
	
	static_assert(std::is_const<Concrete_elem>::value || Opaque_mutable,
				  "cannot cast const ndarray_view_opaque to mutable concrete ndarray_view");
	
	const Opaque_frame_format& format = opaque_view.frame_format();
		Assert(format.is_ndarray(), "opaque frame format must be ndarray");
		
		Assert(format.dimension() == frame_dim, "opaque frame format has wrong dimension");
		
		Assert(format.elem_size() == sizeof(Concrete_elem),
			   "opaque frame format has incorrect element size");
		Assert(is_multiple_of(format.elem_alignment_requirement(), alignof(Concrete_elem)),
			   "opaque frame format has incorrect element alignment");
	
	auto new_start = reinterpret_cast<Concrete_elem*>(opaque_view.start());
	
	ndsize<frame_dim> frame_shape;
	for(std::ptrdiff_t i = 0; i < frame_dim; ++i) frame_shape[i] = format.shape()[i];
	auto new_shape = ndcoord_cat(opaque_view.shape(), frame_shape);
	
	auto frame_strides = ndarray_view<frame_dim, Concrete_elem>::default_strides(frame_shape, format.elem_stride() - sizeof(Concrete_elem));
	auto new_strides = ndcoord_cat(opaque_view.strides(), frame_strides);
	
	return concrete_view_type(new_start, new_shape, new_strides);
}

}

#endif

#endif