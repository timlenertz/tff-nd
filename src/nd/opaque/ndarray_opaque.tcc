namespace tff {


template<std::size_t Dim, typename Frame_format, typename Allocator>
void ndarray_opaque<Dim, Frame_format, Allocator>::construct_frames_() {
	if(frame_format().is_pod()) return;
	
	auto it = base::begin();
	auto it_end = base::end();
	for(; it != it_end; ++it) it->frame_handle().construct();
}


template<std::size_t Dim, typename Frame_format, typename Allocator>
void ndarray_opaque<Dim, Frame_format, Allocator>::destruct_frames_() {	
	if(frame_format().is_pod()) return;
	
	auto it = base::begin();
	auto it_end = base::end();
	for(; it != it_end; ++it) it->frame_handle().destruct();
}


template<std::size_t Dim, typename Frame_format, typename Allocator>
ndarray_opaque<Dim, Frame_format, Allocator>::ndarray_opaque
(const shape_type& shape, const frame_format_type& frm, std::size_t frame_padding, const Allocator& alloc) :
base(
	shape,
	view_type::default_strides(shape, frm, frame_padding),
	(frm.size() + frame_padding) * shape.product(),
	frm.alignment_requirement(),
	alloc,
	frm
) {
	construct_frames_();
}


template<std::size_t Dim, typename Frame_format, typename Allocator>
ndarray_opaque<Dim, Frame_format, Allocator>::ndarray_opaque
(const const_view_type& vw, std::size_t frame_padding, const Allocator& alloc) :
base(
	vw.shape(),
	view_type::default_strides(vw.shape(), vw.frame_format(), frame_padding),
	(vw.frame_format().size() + frame_padding) * vw.shape().product(),
	vw.frame_format().alignment_requirement(),
	alloc,
	vw.frame_format()
) {
	Assert(! vw.is_null());
	construct_frames_();
	base::view().assign(vw);
}


template<std::size_t Dim, typename Frame_format, typename Allocator>
ndarray_opaque<Dim, Frame_format, Allocator>::ndarray_opaque(const ndarray_opaque& arr) :
base(
	arr.shape(),
	arr.strides(),
	arr.allocated_byte_size(),
	arr.frame_format().alignment_requirement(),
	arr.get_allocator(),
	arr.frame_format()
) {
	construct_frames_();
	base::view().assign(arr.cview());
}


template<std::size_t Dim, typename Frame_format, typename Allocator>
ndarray_opaque<Dim, Frame_format, Allocator>::ndarray_opaque(ndarray_opaque&& arr) :
	base(std::move(arr)) { }


template<std::size_t Dim, typename Frame_format, typename Allocator>
ndarray_opaque<Dim, Frame_format, Allocator>::~ndarray_opaque() {
	// when invalidated after move assignment/construction: allocated_size set to 0
	if(base::allocated_size() > 0) destruct_frames_();
}


template<std::size_t Dim, typename Frame_format, typename Allocator>
void ndarray_opaque<Dim, Frame_format, Allocator>::assign(const const_view_type& vw, std::size_t frame_padding) {
	Assert(! vw.is_null());
	destruct_frames_();
	base::reset_(
		vw.shape(),
		view_type::default_strides(vw.shape(), vw.frame_format(), frame_padding),
		(vw.frame_format().size() + frame_padding) * vw.shape().product(),
		vw.frame_format().alignment_requirement(),
		vw.frame_format()
	);
	construct_frames_();
	base::view().assign(vw);
}


template<std::size_t Dim, typename Frame_format, typename Allocator>
auto ndarray_opaque<Dim, Frame_format, Allocator>::operator=(const ndarray_opaque& arr) -> ndarray_opaque& {
	if(&arr == this) return *this;
	destruct_frames_();
	base::reset_(
		arr.shape(),
		arr.strides(),
		arr.allocated_size(),
		arr.frame_format().alignment_requirement(),
		arr.frame_format()
	);
	construct_frames_();
	base::view().assign(arr.cview());
	return *this;
}


template<std::size_t Dim, typename Frame_format, typename Allocator>
auto ndarray_opaque<Dim, Frame_format, Allocator>::operator=(ndarray_opaque&& arr) -> ndarray_opaque& {
	base::operator=(std::move(arr));
	return *this;
}


}

