#ifndef TLZ_NDSPAN_H_
#define TLZ_NDSPAN_H_

#include <ostream>
#include "common.h"
#include "ndcoord.h"

namespace tlz {

template<std::size_t Dim, typename T> class ndspan_iterator;

/// Cuboid n-dimensional span delimited by two \ref ndcoord vectors.
/** Represents the interval, rectangular, or in general `Dim`-dimensional cuboid region where for all coordinates `c`
 ** inside it and for each dimension `0 <= i < Dim`, one has `start_pos()[i] <= c[i] < end_pos()[i]`.
 ** Can be zero-length on any axis (possibly on all). */
template<std::size_t Dim, typename T = std::ptrdiff_t>
class ndspan {
public:
	using coordinates_type = ndcoord<Dim, T>;
	using shape_type = ndsize<Dim>;
	
	using iterator = ndspan_iterator<Dim, T>;
	
private:
	coordinates_type start_;
	coordinates_type end_;

	bool invariant_() const;

public:
	ndspan() = default;
	ndspan(const ndspan&) = default;
	ndspan(const coordinates_type& start, const coordinates_type& end) :
		start_(start), end_(end) { Assert_crit(invariant_()); }
		
	template<typename T2>
	ndspan(const ndspan<Dim, T2>& other) :
		start_(other.start_pos()), end_(other.end_pos()) { }
	
	ndspan(const time_span& tspan) :
		start_(tspan.begin), end_(tspan.end) { static_assert(Dim == 1, "only ndspan<1> from time_span"); }
		
	ndspan& operator=(const ndspan&) = default;

	ndspan& operator=(const time_span& tspan) {
		static_assert(Dim == 1, "only ndspan<1> from time_span");
		start_[0] = tspan.begin;
		end_[0] = tspan.end;
		return *this;
	}
	
	const coordinates_type& start_pos() const { return start_; }
	const coordinates_type& end_pos() const { return end_; }
	void set_start_pos(const coordinates_type& pos) { start_ = pos; Assert_crit(invariant_()); }
	void set_end_pos(const coordinates_type& pos) { end_ = pos; Assert_crit(invariant_()); }

	friend bool operator==(const ndspan& a, const ndspan& b) {
		return (a.start_ == b.start_) && (a.end_ == b.end_);
	}
	friend bool operator!=(const ndspan& a, const ndspan& b) {
		return (a.start_ != b.start_) || (a.end_ != b.end_);
	}
	
	bool includes(const coordinates_type&) const;
	
	bool includes(const ndspan& sub) const;
	bool includes_strict(const ndspan& sub) const;
			
	shape_type shape() const { return end_ - start_; }
	std::size_t size() const { return shape().product(); }

	iterator begin() const ;
	iterator end() const ;
};


template<std::size_t Dim, typename T>
ndspan<Dim, T> make_ndspan(const ndcoord<Dim, T>& start, const ndcoord<Dim, T>& end) {
	return ndspan<Dim, T>(start, end);
}


template<std::size_t Dim, typename T>
ndspan<Dim, T> make_ndspan(const ndcoord<Dim, T>& end) {
	ndcoord<Dim, T> start;
	return ndspan<Dim, T>(start, end);
}


template<std::size_t Dim, typename T>
std::ostream& operator<<(std::ostream& str, const ndspan<Dim, T>& span) {
	str << '[' << span.start_pos() << ", " << span.end_pos() << '[';
	return str;
}


template<std::size_t Dim, typename T>
ndspan<Dim, T> span_intersection(const ndspan<Dim, T>& a, const ndspan<Dim, T>& b);

}


#include "ndspan.tcc"

#endif
