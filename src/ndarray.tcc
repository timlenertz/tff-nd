namespace tff {

template<std::size_t Dim, typename Elem, typename Allocator>
ndarray<Dim, Elem, Allocator>::ndarray(const shape_type& shape, std::size_t elem_padding, const Allocator& allocator) :
base(
	shape,
	view_type::default_strides(shape, elem_padding),
	(sizeof(Elem) + elem_padding) * shape.product(),
	alignof(Elem),
	allocator
) {
	construct_elems_();
}
	

template<std::size_t Dim, typename Elem, typename Allocator> template<typename Other_view, typename>
ndarray<Dim, Elem, Allocator>::ndarray
(const Other_view& vw, std::size_t elem_padding, const Allocator& allocator) :
base(
	vw.shape(),
	view_type::default_strides(vw.shape(), elem_padding),
	(sizeof(Elem) + elem_padding) * vw.shape().product(),
	alignof(Elem),
	allocator
) {
	Assert(! vw.is_null());
	construct_elems_();
	base::view().assign(vw);
}


template<std::size_t Dim, typename Elem, typename Allocator>
ndarray<Dim, Elem, Allocator>::ndarray(const ndarray& arr) :
base(
	arr.shape(),
	arr.strides(),
	arr.allocated_byte_size(),
	alignof(Elem),
	arr.get_allocator()
) {
	construct_elems_();
	base::view().assign(arr.cview());
}


template<std::size_t Dim, typename Elem, typename Allocator>
ndarray<Dim, Elem, Allocator>::ndarray(ndarray&& arr) :
base(std::move(arr)) { }


template<std::size_t Dim, typename Elem, typename Allocator>
ndarray<Dim, Elem, Allocator>::~ndarray() {
	destruct_elems_();
}


template<std::size_t Dim, typename Elem, typename Allocator>
void ndarray<Dim, Elem, Allocator>::construct_elems_() {
	if(! std::is_pod<Elem>::value)
		for(Elem& elem : *this) new (&elem) Elem;
}


template<std::size_t Dim, typename Elem, typename Allocator>
void ndarray<Dim, Elem, Allocator>::destruct_elems_() {
	if(! std::is_pod<Elem>::value)
		for(Elem& elem : *this) elem.~Elem();
}


template<std::size_t Dim, typename Elem, typename Allocator> template<typename Other_view>
auto ndarray<Dim, Elem, Allocator>::assign(const Other_view& vw, std::size_t elem_padding)
-> enable_if_convertible_<Other_view> {
	Assert(! vw.is_null());
	base::reset_(
		vw.shape(),
		view_type::default_strides(vw.shape(), elem_padding),
		(sizeof(Elem) + elem_padding) * vw.shape().product(),
		alignof(Elem)
	);
	base::view().assign(vw);
}


template<std::size_t Dim, typename Elem, typename Allocator>
auto ndarray<Dim, Elem, Allocator>::operator=(const ndarray& arr) -> ndarray& {
	if(&arr == this) return *this;
	base::reset_(
		arr.shape(),
		arr.strides(),
		arr.allocated_size(),
		alignof(Elem)
	);
	base::view().assign(arr);
	return *this;
}

	
template<std::size_t Dim, typename Elem, typename Allocator>
auto ndarray<Dim, Elem, Allocator>::operator=(ndarray&& arr) -> ndarray& {
	base::operator=(std::move(arr));
	return *this;
}




}

