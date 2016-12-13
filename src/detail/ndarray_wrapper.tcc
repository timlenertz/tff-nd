namespace tff { namespace detail {


template<typename Std_allocator, typename = void>
class hybrid_allocator_traits {
public:
	using value_type = typename Std_allocator::value_type;

	static void* allocate(Std_allocator& alloc, std::size_t size, std::size_t alignment) {
		Assert(is_multiple_of(size, sizeof(value_type)));
		Assert(alignment == alignof(value_type));
		std::size_t n = size / sizeof(value_type);
		return std::allocator_traits<Std_allocator>::allocate(alloc, n);
	}
	
	static void deallocate(Std_allocator& alloc, void* ptr, std::size_t size) {
		Assert(is_multiple_of(size, sizeof(value_type)));
		std::size_t n = size / sizeof(value_type);
		std::allocator_traits<Std_allocator>::deallocate(alloc, static_cast<value_type*>(ptr), n);
	}
};

template<typename Raw_allocator>
class hybrid_allocator_traits<Raw_allocator, std::enable_if_t<is_raw_allocator<Raw_allocator>>> {
public:
	static void* allocate(Raw_allocator& alloc, std::size_t size, std::size_t alignment) {
		return alloc.raw_allocate(size, alignment);
	}
	
	static void deallocate(Raw_allocator& alloc, void* ptr, std::size_t size) {
		alloc.raw_deallocate(ptr, size);
	}
};


///////////////


template<typename View, typename Const_view, typename Allocator>
void ndarray_wrapper<View, Const_view, Allocator>::allocate_(std::size_t size, std::size_t alignment) {
	if(size > 0) {
		void* buf = hybrid_allocator_traits<Allocator>::allocate(allocator_, size, alignment);
		allocated_size_ = size;
		allocated_buffer_ = buf;
	}
}


template<typename View, typename Const_view, typename Allocator>
void ndarray_wrapper<View, Const_view, Allocator>::deallocate_() {
	if(allocated_size_ != 0) {
		hybrid_allocator_traits<Allocator>::deallocate(allocator_, allocated_buffer_, allocated_size_);
		allocated_size_ = 0;
		allocated_buffer_ = nullptr;
	}
}


template<typename View, typename Const_view, typename Allocator> template<typename... Arg>
ndarray_wrapper<View, Const_view, Allocator>::ndarray_wrapper(
	const shape_type& shape,
	const strides_type& strides,
	std::size_t allocate_size,
	std::size_t allocate_alignment,
	const Allocator& allocator,
	const Arg&... view_arguments
) :
	allocator_(allocator)
{
	allocate_(allocate_size, allocate_alignment);
	view_.reset(view_type(
		static_cast<typename view_type::pointer>(allocated_buffer_),
		shape,
		strides,
		view_arguments...
	));
	Assert(static_cast<void*>(view_.start()) == allocated_buffer_, "first element in ndarray must be at buffer start");
}
	

template<typename View, typename Const_view, typename Allocator>
ndarray_wrapper<View, Const_view, Allocator>::ndarray_wrapper(ndarray_wrapper&& arr) :
	allocator_(),
	allocated_size_(arr.allocated_size_),
	allocated_buffer_(arr.allocated_buffer_),
	view_(arr.view_)
{
	arr.view_.reset();
	arr.allocated_size_ = 0;
	arr.allocated_buffer_ = nullptr;
}
		

template<typename View, typename Const_view, typename Allocator>
ndarray_wrapper<View, Const_view, Allocator>::~ndarray_wrapper() {
	deallocate_();
}


template<typename View, typename Const_view, typename Allocator>
auto ndarray_wrapper<View, Const_view, Allocator>::operator=(ndarray_wrapper&& arr) -> ndarray_wrapper& {
	if(&arr == this) return *this;
	
	deallocate_();
	
	allocated_size_ = arr.allocated_size_;
	allocated_buffer_ = arr.allocated_buffer_;
	view_.reset(arr.view_);
	
	arr.allocated_size_ = 0;
	arr.allocated_buffer_ = nullptr;
	arr.view_.reset();
	
	return *this;
}


template<typename View, typename Const_view, typename Allocator> template<typename... Arg>
void ndarray_wrapper<View, Const_view, Allocator>::reset_(
	const shape_type& shape,
	const strides_type& strides,
	std::size_t allocate_size,
	std::size_t allocate_alignment,
	const Arg&... view_arguments
) {	
	// reallocate memory only if necessary
	if(allocate_size != allocated_size() || ! is_aligned(allocated_buffer_, allocate_alignment)) {
		deallocate_();
		allocate_(allocate_size, allocate_alignment);
	}
	
	view_.reset(view_type(
		static_cast<typename view_type::pointer>(allocated_buffer_),
		shape,
		strides,
		view_arguments...
	));
}


}}
