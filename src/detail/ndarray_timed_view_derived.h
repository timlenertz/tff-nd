#ifndef TFF_NDARRAY_TIMED_VIEW_DERIVED_H_
#define TFF_NDARRAY_TIMED_VIEW_DERIVED_H_

#include "../config.h"
#if TFF_ND_WITH_TIMED

#include "../common.h"
#include "../ndarray_view.h"
#include "ndarray_view_fcall.h"

namespace tff { namespace detail {


template<typename Base>
class ndarray_timed_view_derived : public Base {
	using base = Base;

private:
	time_unit start_time_;

	using fcall_type = detail::ndarray_view_fcall<ndarray_timed_view_derived, 1>;
	
public:
	using non_timed_view_type = Base;
	
	using typename base::coordinates_type;
	using typename base::strides_type;
	using typename base::span_type;

	/// \name Construction
	///@{
	ndarray_timed_view_derived() : base(), start_time_(-1) { }

	ndarray_timed_view_derived(const base& vw, time_unit start_time) :
		base(vw), start_time_(vw.is_null() ? -1 : start_time) { }
	
	template<typename... Args> void reset(const Args&... args) {
		reset(ndarray_timed_view_derived(args...));
	}
	void reset(const ndarray_timed_view_derived& vw) {
		start_time_ = vw.start_time_;
		base::reset(vw);
	}

	static ndarray_timed_view_derived null() { return ndarray_timed_view_derived(); }
	
	friend bool same(const ndarray_timed_view_derived& a, const ndarray_timed_view_derived& b) {
		return (a.start_time() == b.start_time()) && same(a.non_timed(), b.non_timed());
	}
	
	const non_timed_view_type& non_timed() const { return *this; }
	///@}
	
	
	
	/// \name Attributes
	///@{
	time_unit start_time() const { return start_time_; }
	time_unit end_time() const { return start_time_ + base::shape().front(); }
	time_unit duration() const { return base::shape().front(); }
	
	std::ptrdiff_t time_to_coordinate(time_unit t) const { return t - start_time_; }
	time_unit coordinate_to_time(std::ptrdiff_t i) const { return start_time_ + base::fix_coordinate_(i, 0); }

	time_span tspan() const { return time_span(start_time(), start_time() + duration()); }
	///@}
	
	
	
	/// \name Indexing
	///@{
	ndarray_timed_view_derived axis_section
		(std::ptrdiff_t dim, std::ptrdiff_t start, std::ptrdiff_t end, std::ptrdiff_t step) const {
		return ndarray_timed_view_derived(base::axis_section(dim, start, end, step), start_time_ + (dim == 0 ? start : 0));
	}
	// required by ndarray_view_fcall
	
	
	decltype(auto) at_time(time_unit t) const { return base::operator[](time_to_coordinate(t)); }
	
	auto tsection(time_span span) {
		std::ptrdiff_t start = time_to_coordinate(span.start_time());
		std::ptrdiff_t end = time_to_coordinate(span.end_time());
		return operator()(start, end);
	}

	auto section
	(const coordinates_type& start, const coordinates_type& end, const strides_type& steps = strides_type(1)) const {
		return ndarray_timed_view_derived(base::section(start, end, steps), start_time_ + start.front());
	}

	auto section(const span_type& span, const strides_type& steps = strides_type(1)) const {
		return section(span.start_pos(), span.end_pos(), steps);
	}
	
	fcall_type operator()(std::ptrdiff_t start, std::ptrdiff_t end, std::ptrdiff_t step = 1) const {
		return ndarray_timed_view_derived(base::axis_section(0, start, end, step), start_time_ + start);
	}
	fcall_type operator()(std::ptrdiff_t c) const {
		return ndarray_timed_view_derived(base::axis_section(0, c, c + 1, 1), start_time_ + c);
	}
	fcall_type operator()() const {
		return *this;
	}
	///@}
};

}}

#endif
#endif
