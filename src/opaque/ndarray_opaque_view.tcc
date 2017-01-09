#include "../opaque_format/ndarray.h"

namespace tff {


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



}
