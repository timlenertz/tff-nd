#ifndef TFF_NDARRAY_VIEW_FCALL_H_
#define TFF_NDARRAY_VIEW_FCALL_H_

#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <type_traits>

namespace tff { namespace detail {

template<typename View, std::ptrdiff_t Target_dim>
class ndarray_view_fcall : public View {
	static_assert(Target_dim <= View::dimension, "detail::ndarray_view_fcall target dimension out of bounds");
	using base = View;

private:
	using fcall_type = ndarray_view_fcall<View, Target_dim + 1>;

public:
	using base::base;
	
	ndarray_view_fcall(const base& vw) : base(vw) { }
	
	fcall_type operator()(std::ptrdiff_t start, std::ptrdiff_t end, std::ptrdiff_t step = 1) const {
		return base::section_(Target_dim, start, end, step);
	}
	fcall_type operator()(std::ptrdiff_t c) const {
		return base::section_(Target_dim, c, c + 1, 1);
	}
	fcall_type operator()() const {
		return *this;
	}		
};

}}

#endif
