#include "../opaque_format/ndarray.h"

namespace tff {

template<std::size_t Dim, bool Mutable, typename Frame_format>
auto ndarray_opaque_view<Dim, Mutable, Frame_format>::section_
(std::ptrdiff_t dim, std::ptrdiff_t start, std::ptrdiff_t end, std::ptrdiff_t step) const -> ndarray_opaque_view {
	Assert(dim < Dim);
	return ndarray_opaque_view(base::section_(dim, start, end, step), frame_format_);
}


template<std::size_t Dim, bool Mutable, typename Frame_format>
ndarray_opaque_view<Dim, Mutable, Frame_format>::ndarray_opaque_view
(pointer start, const shape_type& shape, const strides_type& strides, const frame_format_type& frm) :
	base(
		static_cast<typename base::pointer>(start),
		ndcoord_cat(shape, 1),
		ndcoord_cat(strides, frm.size())
	),
	frame_format_(frm) { }


template<std::size_t Dim, bool Mutable, typename Frame_format>
ndarray_opaque_view<Dim, Mutable, Frame_format>::ndarray_opaque_view
(pointer start, const shape_type& shape, const frame_format_type& frm) :
	ndarray_opaque_view(start, shape, default_strides(shape, frm), frm) { }


template<std::size_t Dim, bool Mutable, typename Frame_format>
void ndarray_opaque_view<Dim, Mutable, Frame_format>::reset(const ndarray_opaque_view& other) {
	base::reset(other.base_view());
	frame_format_ = other.frame_format_;
}


template<std::size_t Dim, bool Mutable, typename Frame_format>
auto ndarray_opaque_view<Dim, Mutable, Frame_format>::default_strides
(const shape_type& shape, const frame_format_type& frm, std::size_t frame_padding) -> strides_type {
	if(Dim == 0) return ndptrdiff<Dim>();
	Assert(is_multiple_of(frame_padding, frm.alignment_requirement()));
	strides_type strides;
	strides[Dim - 1] = frm.size() + frame_padding;
	for(std::ptrdiff_t i = Dim - 1; i > 0; --i)
		strides[i - 1] = strides[i] * shape[i];
	return strides;
}


template<std::size_t Dim, bool Mutable, typename Frame_format>
bool ndarray_opaque_view<Dim, Mutable, Frame_format>::has_default_strides(std::ptrdiff_t minimal_dimension) const {
	if(Dim == 0) return true;
	if(strides().back() < frame_format().size()) return false;
	for(std::ptrdiff_t i = Dim - 2; i >= minimal_dimension; --i) {
		std::ptrdiff_t expected_stride = shape()[i + 1] * strides()[i + 1];
		if(strides()[i] != expected_stride) return false;
	}
	return true;
}


template<std::size_t Dim, bool Mutable, typename Frame_format>
std::size_t ndarray_opaque_view<Dim, Mutable, Frame_format>::default_strides_padding(std::ptrdiff_t minimal_dimension) const {
	if(Dim == 0) return 0;
	Assert(has_default_strides(minimal_dimension));
	return (strides().back() - frame_format().size());
}


template<std::size_t Dim, bool Mutable, typename Frame_format>
bool ndarray_opaque_view<Dim, Mutable, Frame_format>::has_default_strides_without_padding(std::ptrdiff_t minimal_dimension) const {
	if(Dim == 0) return true;
	else if(has_default_strides(minimal_dimension)) return (default_strides_padding(minimal_dimension) == 0);
	else return false;
}


template<std::size_t Dim, bool Mutable, typename Frame_format>
void ndarray_opaque_view<Dim, Mutable, Frame_format>::assign(const ndarray_opaque_view<Dim, false, Frame_format>& vw) const {
	Assert(!is_null() && !vw.is_null());
	Assert(vw.frame_format() == frame_format());
	Assert(vw.shape() == shape());
	
	if(frame_format().is_pod() && frame_format().pod_format().is_contiguous()
	   && has_default_strides_without_padding() && vw.strides() == strides()) {
		// directly copy entire memory segment
		std::memcpy(start(), vw.start(), frame_format().size() * size());
	} else {
		// copy frame-by-frame, using assign function of frame handle
		auto it = begin();
		auto it_end = end();
		auto other_it = vw.begin();
		for(; it != it_end; ++it, ++other_it) {
			auto&& vw = *it;
			auto&& other_vw = *other_it;
			vw.frame_handle().assign(other_vw.frame_handle());
		}
	}
}



template<std::size_t Dim, bool Mutable, typename Frame_format>
bool ndarray_opaque_view<Dim, Mutable, Frame_format>::compare(const ndarray_opaque_view<Dim, false, Frame_format>& vw) const {
	Assert(!is_null() && !vw.is_null());
	Assert(vw.frame_format() == frame_format());
	Assert(vw.shape() == shape());
	
	if(frame_format().is_pod() && frame_format().pod_format().is_contiguous()
	   && has_default_strides_without_padding() && vw.strides() == strides()) {
		// directly compare entire memory segment
		return (std::memcmp(start(), vw.start(), frame_format().size() * size()) == 0);
	} else {
		// compare frame-by-frame, using compare function of frame handle
		auto it = begin();
		auto it_end = end();
		auto other_it = vw.begin();
		for(; it != it_end; ++it, ++other_it) {
			const auto&& vw = *it;
			const auto&& other_vw = *other_it;			
			bool frame_equal = vw.frame_handle().compare(other_vw.frame_handle());
			if(! frame_equal) return false;
		}
		return true;
	}
}


template<std::size_t Dim, bool Mutable, typename Frame_format>
auto ndarray_opaque_view<Dim, Mutable, Frame_format>::at(const coordinates_type& coord) const -> frame_view_type {
	auto base_coord = ndcoord_cat(coord, 0);
	auto ptr = static_cast<frame_pointer_type>(&base::at(base_coord));
	return frame_view_type(ptr, make_ndsize(), frame_format_);
}


template<std::size_t Dim, bool Mutable1, bool Mutable2, typename Frame_format>
bool same
(const ndarray_opaque_view<Dim, Mutable1, Frame_format>& a, const ndarray_opaque_view<Dim, Mutable2, Frame_format>& b) {
	return same(a.base_view(), b.base_view()) && (a.frame_format() == b.frame_format());
}


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


template<std::size_t Tail_dim, std::size_t Dim, bool Mutable, typename Frame_format>
bool tail_has_pod_format(const ndarray_opaque_view<Dim, Mutable, Frame_format>& vw) {
	pod_array_format frame_pod_format = vw.frame_format().pod_format();
	if(frame_pod_format.is_contiguous()) return vw.has_default_strides(Dim - Tail_dim);
	else return vw.has_default_strides_without_padding(Dim - Tail_dim);
}

template<std::size_t Dim, bool Mutable, typename Frame_format>
bool has_pod_format(const ndarray_opaque_view<Dim, Mutable, Frame_format>& vw) {
	return tail_has_pod_format<Dim>(vw);
}


template<std::size_t Tail_dim, std::size_t Dim, bool Mutable, typename Frame_format>
pod_array_format tail_pod_format(const ndarray_opaque_view<Dim, Mutable, Frame_format>& vw) {
	pod_array_format frame_pod_format = vw.frame_format().pod_format();
	
	if(frame_pod_format.is_contiguous()) {
		std::size_t frame_padding = vw.default_strides_padding(Dim - Tail_dim);
		std::size_t elem_size = frame_pod_format.size();
		std::size_t elem_align = frame_pod_format.elem_alignment();
		std::size_t length = tail<Tail_dim>(vw.shape()).product();
		std::size_t stride = vw.strides().back();
		return pod_array_format(elem_size, elem_align, length, stride);
		
	} else {
		Assert(vw.has_default_strides_without_padding(Dim - Tail_dim));
		std::size_t elem_size = frame_pod_format.elem_size();
		std::size_t elem_align = frame_pod_format.elem_alignment();
		std::size_t length = tail<Tail_dim>(vw.shape()).product() * frame_pod_format.length();
		std::size_t stride = frame_pod_format.stride();
		return pod_array_format(elem_size, elem_align, length, stride);
	}
}


template<std::size_t Dim, bool Mutable, typename Frame_format>
pod_array_format pod_format(const ndarray_opaque_view<Dim, Mutable, Frame_format>& vw) {
	return tail_pod_format<Dim>(vw);
}


}
