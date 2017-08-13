#ifndef TLZ_NDARRAY_OPAQUE_VIEW_WRAPPER_H_
#define TLZ_NDARRAY_OPAQUE_VIEW_WRAPPER_H_

#include "../config.h"
#if TLZ_ND_WITH_OPAQUE

#include <utility>
#include <type_traits>
#include "../common.h"
#include "../ndarray_iterator.h"
#include "ndarray_view_fcall.h"
#include "../opaque/ndarray_opaque_traits.h"


namespace tlz { namespace detail {

template<std::size_t Dim, bool Mutable, typename Frame_format, template<std::size_t,typename> class Base_view>
class ndarray_opaque_view_wrapper;


template<std::size_t Dim, bool Mutable, typename Frame_format, template<std::size_t,typename> class Base_view>
using ndarray_opaque_view_wrapper_base = Base_view<Dim + 1, std::conditional_t<Mutable, byte, const byte>>;



template<std::size_t Dim, bool Mutable, typename Frame_format, template<std::size_t,typename> class Base_view>
class ndarray_opaque_view_wrapper : private ndarray_opaque_view_wrapper_base<Dim, Mutable, Frame_format, Base_view> {
	using base = ndarray_opaque_view_wrapper_base<Dim, Mutable, Frame_format, Base_view>;

public:
	using frame_format_type = Frame_format;
	using frame_view_type = ndarray_opaque_view_wrapper<0, Mutable, Frame_format, Base_view>;
	using frame_handle_type = std::conditional_t
	<Mutable, typename frame_format_type::frame_handle_type, typename frame_format_type::const_frame_handle_type>;
	using frame_pointer_type = std::conditional_t
	<Mutable, typename frame_format_type::frame_pointer_type, typename frame_format_type::const_frame_pointer_type>;
	
	using value_type = frame_view_type;
	using reference = frame_view_type;
	using pointer = frame_pointer_type;
	
	using index_type = std::ptrdiff_t;
	using coordinates_type = ndptrdiff<Dim>;
	using shape_type = ndsize<Dim>;
	using strides_type = ndptrdiff<Dim>;
	using span_type = ndspan<Dim>;
	
	using iterator = ndarray_iterator<ndarray_opaque_view_wrapper>;
	
	static constexpr std::size_t dimension() { return Dim; }
	static constexpr bool is_mutable() { return Mutable; }

private:
	template<typename Other_view, typename U = void>
	using enable_if_convertible_ = std::enable_if_t<is_convertible_ndarray_opaque_view<Other_view, ndarray_opaque_view_wrapper>::value, U>;

	using fcall_type = ndarray_view_fcall<ndarray_opaque_view_wrapper, 1>;

	frame_format_type frame_format_;
	
protected:
	using base::fix_coordinate_;
	// required by ndarray_timed_view_derived<ndarray_opaque_view_wrapper>

public:
	/// \name Construction
	///@{
	ndarray_opaque_view_wrapper() = default;
	
	ndarray_opaque_view_wrapper(const base& base_vw, const frame_format_type& frm) :
		base(base_vw), frame_format_(frm) { }
	
	ndarray_opaque_view_wrapper(const ndarray_opaque_view_wrapper<Dim, true, Frame_format, Base_view>& vw) :
		base(vw.base_view()), frame_format_(vw.frame_format()) { }
	
	ndarray_opaque_view_wrapper(pointer start, const shape_type& shp, const strides_type& str, const frame_format_type& frm) :
		base(
			static_cast<typename base::pointer>(start),
			ndcoord_cat(shp, 1),
			ndcoord_cat(str, frm.size())
		),
		frame_format_(frm) { }
	
	ndarray_opaque_view_wrapper(pointer start, const shape_type& shp, const frame_format_type& frm) :
	ndarray_opaque_view_wrapper(start, shp, default_strides(shp, frm), frm) { }
	
	static ndarray_opaque_view_wrapper null() { return ndarray_opaque_view_wrapper(); }
	bool is_null() const { return base::is_null(); }
	explicit operator bool () const { return ! is_null(); }
	
	template<typename... Args> void reset(const Args&... args) { reset(ndarray_opaque_view_wrapper(args...)); }
	void reset(const ndarray_opaque_view_wrapper& other) {
		base::reset(other.base_view());
		frame_format_ = other.frame_format_;
	}
	
	const base& base_view() const { return *this; }
	///@}
	
	
	/// \name Attributes
	///@{
	frame_pointer_type start() const { return static_cast<pointer>(base::start()); }
	shape_type shape() const { return head<Dim>(base::shape()); }
	strides_type strides() const { return head<Dim>(base::strides()); }
	
	std::size_t size() const { return shape().product(); }
	span_type full_span() const { return span_type(0, shape()); }
	
	static strides_type default_strides(const shape_type& shp, const frame_format_type& frm, std::size_t frame_padding = 0) {
		if(Dim == 0) return ndptrdiff<Dim>();
		Assert(is_multiple_of(frame_padding, frm.alignment_requirement()));
		strides_type strides;
		strides[Dim - 1] = frm.size() + frame_padding;
		for(std::ptrdiff_t i = Dim - 1; i > 0; --i)
			strides[i - 1] = strides[i] * shp[i];
		return strides;
	}
	
	bool has_default_strides(std::ptrdiff_t minimal_dimension = 0) const {
		if(Dim == 0) return true;
		if(strides().back() < frame_format().size()) return false;
		for(std::ptrdiff_t i = Dim - 2; i >= minimal_dimension; --i) {
			std::ptrdiff_t expected_stride = shape()[i + 1] * strides()[i + 1];
			if(strides()[i] != expected_stride) return false;
		}
		return true;
	}
	
	std::size_t default_strides_padding(std::ptrdiff_t minimal_dimension = 0) const {
		if(Dim == 0) return 0;
		Assert(has_default_strides(minimal_dimension));
		return (strides().back() - frame_format().size());
	}
	
	bool has_default_strides_without_padding(std::ptrdiff_t minimal_dimension = 0) const {
		if(Dim == 0) return true;
		else if(has_default_strides(minimal_dimension)) return (default_strides_padding(minimal_dimension) == 0);
		else return false;
	}
	
	const frame_format_type& frame_format() const { return frame_format_; }
	///@}
	
	
	/// \name Frame handle
	///@{
	frame_handle_type frame_handle() const
	{ static_assert(Dim == 0, "frame handle needs 0 dim view"); return frame_handle_type(start(), frame_format_); }
	
	operator frame_handle_type () const { return frame_handle(); }
	///@}
	
	
	/// \name Deep assignment
	///@{
	template<typename Other_view>
	enable_if_convertible_<Other_view> assign(const Other_view& other) const {
		Assert(other.frame_format() == frame_format());
		Assert(other.shape() == shape());
		
		if(has_pod_format() && other.has_pod_format() && pod_format() == other.pod_format()) {
			pod_array_copy(start(), other.start(), pod_format());
		} else {
			auto it = begin();
			for(const auto& other_view : other) {
				(it++)->frame_handle().assign(other_view.frame_handle());
			}
		}
	}
	
	template<typename Arg>
	const ndarray_opaque_view_wrapper& operator=(Arg&& arg) const
		{ assign(std::forward<Arg>(arg)); return *this; }
	const ndarray_opaque_view_wrapper& operator=(const ndarray_opaque_view_wrapper& vw) const
		{ assign(vw); return *this; }
	///@}
	
	
	/// \name Deep comparison
	///@{
	template<typename Other_view>
	enable_if_convertible_<Other_view, bool> compare(const Other_view& other) const {
		// TODO support nullable Other_view, check is_null, also assign()
		Assert(other.frame_format() == frame_format());
		Assert(other.shape() == shape());
		
		if(has_pod_format() && other.has_pod_format() && pod_format() == other.pod_format()) {
			return pod_array_compare(start(), other.start(), pod_format());
		} else {
			auto it = begin();
			for(const auto& other_view : other) {
				bool frame_equal = (it++)->frame_handle().compare(other_view.frame_handle());
				if(! frame_equal) return false;
			}
			return true;
		}
	}
	
	template<typename Arg> bool operator==(const Arg& arg) const { return compare(arg); }
	template<typename Arg> bool operator!=(const Arg& arg) const { return ! compare(arg); }
	///@}
	
	
	/// \name Iteration
	///@{
	frame_view_type dereference(pointer ptr) const {
		return frame_view_type(ptr, make_ndsize(), frame_format_);
	}
	
	std::ptrdiff_t contiguous_length() const { return base::contiguous_length(); }
	
	iterator begin() const {
		return iterator(*this, 0, start());
	}
	iterator end() const {
		index_type end_index = shape().product();
		coordinates_type end_coord = index_to_coordinates(end_index);
		return iterator(*this, end_index, coordinates_to_pointer(end_coord));
	}
	///@}
	
	/// \name Indexing
	///@{
	coordinates_type index_to_coordinates(index_type idx) const {
		return head<Dim>(base::index_to_coordinates(idx));
	}
	index_type coordinates_to_index(const coordinates_type& coord) const {
		return base::coordinates_to_index(ndcoord_cat(coord, 0));
	}
	pointer coordinates_to_pointer(const coordinates_type& coord) const {
		return base::coordinates_to_pointer(ndcoord_cat(coord, 0));
	}
	
	frame_view_type at(const coordinates_type& coord) const {
		auto base_coord = ndcoord_cat(coord, 0);
		auto ptr = static_cast<frame_pointer_type>(&base::at(base_coord));
		return frame_view_type(ptr, make_ndsize(), frame_format_);
	}
	
	ndarray_opaque_view_wrapper axis_section(std::ptrdiff_t dim, std::ptrdiff_t start, std::ptrdiff_t end, std::ptrdiff_t step) const {
		// required by ndarray_view_fcall
		Assert(dim < Dim);
		return ndarray_opaque_view_wrapper(base::axis_section(dim, start, end, step), frame_format_);
	}
	
	ndarray_opaque_view_wrapper section
	(const coordinates_type& start, const coordinates_type& end, const strides_type& steps = strides_type(1)) const {
		return ndarray_opaque_view_wrapper(base::section(ndcoord_cat(start, 0), ndcoord_cat(end, 1), ndcoord_cat(steps, 1)), frame_format_);
	}
	ndarray_opaque_view_wrapper section(const span_type& span, const strides_type& steps = strides_type(1)) const {
		return section(span.start_pos(), span.end_pos(), steps);
	}
	
	auto slice(std::ptrdiff_t c, std::ptrdiff_t dimension) const {
		return ndarray_opaque_view_wrapper<Dim - 1, Mutable, Frame_format, Base_view>(base::slice(c, dimension), frame_format_);
	}
	auto operator[](std::ptrdiff_t c) const {
		return slice(c, 0);
	}
	
	fcall_type operator()(std::ptrdiff_t start, std::ptrdiff_t end, std::ptrdiff_t step = 1) const {
		return ndarray_opaque_view_wrapper(base::operator()(start, end, step), frame_format_);
	}
	fcall_type operator()(std::ptrdiff_t c) const {
		return ndarray_opaque_view_wrapper(base::operator()(c), frame_format_);
	}
	fcall_type operator()() const {
		return ndarray_opaque_view_wrapper(base::operator()(), frame_format_);
	}
	///@}
	
	
	/// \name POD format
	///@{
	template<std::size_t Tail_dim>
	bool tail_has_pod_format() const {
		if(! frame_format().is_pod()) return false;
		pod_array_format frame_pod_format = frame_format().pod_format();
		if(frame_pod_format.is_contiguous()) return has_default_strides(Dim - Tail_dim);
		else return has_default_strides_without_padding(Dim - Tail_dim);
	}
	
	
	bool has_pod_format() const {
		return tail_has_pod_format<Dim>();
	}
	
	template<std::size_t Tail_dim>
	pod_array_format tail_pod_format() const {
		pod_array_format frame_pod_format = frame_format().pod_format();
		
		if(Tail_dim == 0) {
			return frame_pod_format;
			
		} else if(frame_pod_format.is_contiguous()) {
			std::size_t frame_padding = default_strides_padding(Dim - Tail_dim);
			std::size_t elem_size = frame_pod_format.size();
			std::size_t elem_align = frame_pod_format.elem_alignment();
			std::size_t length = tail<Tail_dim>(shape()).product();
			std::size_t stride = strides().back();
			return pod_array_format(elem_size, elem_align, length, stride);
			
		} else {
			Assert(has_default_strides_without_padding(Dim - Tail_dim));
			std::size_t elem_size = frame_pod_format.elem_size();
			std::size_t elem_align = frame_pod_format.elem_alignment();
			std::size_t length = tail<Tail_dim>(shape()).product() * frame_pod_format.length();
			std::size_t stride = frame_pod_format.stride();
			return pod_array_format(elem_size, elem_align, length, stride);
		}
	}
	
	pod_array_format pod_format() const {
		return tail_pod_format<Dim>();
	}
	///@}
};


template<std::size_t Dim, bool Mutable1, bool Mutable2, typename Frame_format, template<std::size_t,typename> class Base_view>
bool same(const ndarray_opaque_view_wrapper<Dim, Mutable1, Frame_format, Base_view>& a, const ndarray_opaque_view_wrapper<Dim, Mutable2, Frame_format, Base_view>& b) {
	return same(a.base_view(), b.base_view()) && (a.frame_format() == b.frame_format());
}

///////////////
	
	
}}

#endif
#endif
