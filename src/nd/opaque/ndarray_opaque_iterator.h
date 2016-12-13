#ifndef TFF_NDARRAY_OPAQUE_ITERATOR_H_
#define TFF_NDARRAY_OPAQUE_ITERATOR_H_

#include "../config.h"
#if TFF_ND_WITH_OPAQUE

#include <utility>
#include "../ndarray_iterator.h"
#include "../ndcoord.h"

namespace tff {

template<std::size_t Dim, bool Mutable, typename Frame_format>
class ndarray_opaque_view;


template<std::size_t Dim, bool Mutable, typename Frame_format>
class ndarray_opaque_iterator : public ndarray_iterator<Dim + 1, std::conditional_t<Mutable, byte, const byte>> {
	using base = ndarray_iterator<Dim + 1, std::conditional_t<Mutable, byte, const byte>>;

public:
	using frame_format_type = Frame_format;
	using frame_view_type = ndarray_opaque_view<0, Mutable, Frame_format>;
	using frame_pointer_type = std::conditional_t
		<Mutable, typename frame_format_type::frame_pointer_type, typename frame_format_type::const_frame_pointer_type>;

private:
	struct member_to_pointer_wrapper_ {
		frame_view_type view;
		member_to_pointer_wrapper_(const frame_view_type& vw) : view(vw) { }
		const frame_view_type* operator->() const { return &view; }
	};
	
	frame_format_type frame_format_;
	
	const base& base_() const { return *this; }

public:
	using value_type = frame_view_type;
	using reference = frame_view_type;
	using pointer = member_to_pointer_wrapper_;

	using view_type = ndarray_opaque_view<Dim, Mutable, Frame_format>;
	using coordinates_type = typename view_type::coordinates_type;
	constexpr static std::size_t dimension = view_type::dimension;


public:
	ndarray_opaque_iterator() : base() { }
	ndarray_opaque_iterator(const base& it, const frame_format_type& frm) : base(it), frame_format_(frm) { }
	ndarray_opaque_iterator(const ndarray_opaque_iterator&) = default;
	
	ndarray_opaque_iterator& operator=(const ndarray_opaque_iterator& it)
		{ base::operator=(it); return *this; }
	
	view_type view() const { return view_type(base::view(), frame_format_); }
	
	coordinates_type coordinates() const { return base::coordinates().head(); }
	
	frame_pointer_type ptr() const { return static_cast<frame_pointer_type>(base::ptr()); }
	reference operator*() const { return frame_view_type(ptr(), make_ndsize(), frame_format_); }
	member_to_pointer_wrapper_ operator->() const { return operator*(); }
	reference operator[](std::ptrdiff_t n) const { return *(*this + n); }
		
	ndarray_opaque_iterator& operator++() { base::operator++(); return *this; }
	ndarray_opaque_iterator operator++(int) { return ndarray_opaque_iterator(base::operator++(int()), frame_format_); }
	ndarray_opaque_iterator& operator--() { base::operator--(); return *this; }
	ndarray_opaque_iterator operator--(int) { return ndarray_opaque_iterator(base::operator--(int()), frame_format_); }
	ndarray_opaque_iterator& operator+=(std::ptrdiff_t n) { base::operator+=(n); return *this; }
	ndarray_opaque_iterator& operator-=(std::ptrdiff_t n) { base::operator-=(n); return *this; }
	
	friend ndarray_opaque_iterator operator+(const ndarray_opaque_iterator& it, std::ptrdiff_t n)
		{ return ndarray_opaque_iterator(it.base_() + n, it.frame_format_); }
	friend ndarray_opaque_iterator operator+(std::ptrdiff_t n, const ndarray_opaque_iterator& it)
		{ return ndarray_opaque_iterator(n + it.base_(), it.frame_format_); }
	friend ndarray_opaque_iterator operator-(const ndarray_opaque_iterator& it, std::ptrdiff_t n)
		{ return ndarray_opaque_iterator(it.base_() - n, it.frame_format_); }
};

}

#endif
#endif
