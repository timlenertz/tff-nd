#ifndef TFF_NDARRAY_VIEW_FCALL_H_
#define TFF_NDARRAY_VIEW_FCALL_H_

#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <type_traits>
#include "../ndarray_traits.h"

namespace tff { namespace detail {

template<typename View, std::ptrdiff_t Target_dim>
class ndarray_view_fcall : public View {
	static_assert(Target_dim <= View::dimension(), "detail::ndarray_view_fcall target dimension out of bounds");
	using base = View;

private:
	using fcall_type = ndarray_view_fcall<View, Target_dim + 1>;

public:
	using base::base;
	
	ndarray_view_fcall(const base& vw) : base(vw) { }
	
	fcall_type operator()(std::ptrdiff_t start, std::ptrdiff_t end, std::ptrdiff_t step = 1) const {
		return base::axis_section(Target_dim, start, end, step);
	}
	fcall_type operator()(std::ptrdiff_t c) const {
		if(c != -1) return base::axis_section(Target_dim, c, c + 1, 1);
		else return base::axis_section(Target_dim, base::shape()[Target_dim] - 1, base::shape()[Target_dim], 1);
	}
	fcall_type operator()() const {
		return *this;
	}		
};

}}

namespace tff {

template<typename View, std::ptrdiff_t Target_dim>
struct is_ndarray_view<detail::ndarray_view_fcall<View, Target_dim>> : std::true_type {};

}

#endif
