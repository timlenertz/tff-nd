#ifndef TLZ_NDCOORD_DYN_H_
#define TLZ_NDCOORD_DYN_H_

#include <cstddef>
#include <array>
#include <initializer_list>
#include <functional>
#include <ostream>
#include <type_traits>
#include <limits>
#include <iterator>
#include <algorithm>
#include "common.h"
#include "ndcoord.h"

namespace tlz {


template<typename T, std::size_t Max_dim = 4>
class ndcoord_dyn {
	static_assert(std::is_arithmetic<T>::value, "ndcoord component type must be arithmetic");

private:
	std::size_t size_ = 0;
	std::array<T, Max_dim> components_;

public:
	using value_type = T;
	using reference = T&;
	using const_reference = const T&;
	using iterator = typename std::array<T, Max_dim>::iterator;
	using const_iterator = typename std::array<T, Max_dim>::const_iterator;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;
		
	ndcoord_dyn() = default;
	
	ndcoord_dyn(std::size_t dim, const T& val = T()) : size_(dim) {
		for(std::ptrdiff_t i = 0; i < size_; ++i) components_[i] = val;
	}

	template<typename It, typename = typename std::iterator_traits<It>::value_type>
	ndcoord_dyn(It begin, It end) {
		Assert_crit(std::distance(begin, end) <= Max_dim);
		auto out = components_.begin();
		size_ = 0;
		for(auto in = begin; in != end; ++in, ++out, ++size_) *out = static_cast<T>(*in);
	}
		
	ndcoord_dyn(std::initializer_list<T> l) :
		ndcoord_dyn(l.begin(), l.end()) { }
			
	ndcoord_dyn(const ndcoord_dyn& other) : size_(other.size_) {
		std::copy_n(other.components_.cbegin(), size_, components_.begin());
	}
	
	template<typename T2, std::size_t Max_dim2>
	ndcoord_dyn(const ndcoord_dyn<T2, Max_dim2>& other) :
		ndcoord_dyn(other.cbegin(), other.cend()) { }
		
	ndcoord_dyn& operator=(const ndcoord_dyn& other) {
		size_ = other.size_;
		std::copy_n(other.components_.cbegin(), size_, components_.begin());
		return *this;
	}
	
	template<typename T2, std::size_t Max_dim2>
	ndcoord_dyn& operator=(const ndcoord_dyn<T2, Max_dim2>& other) {
		Assert_crit(other.size() <= Max_dim);
		size_ = other.size();
		std::copy_n(other.cbegin(), size_, components_.begin());
		return *this;
	}
	
	T& operator[](std::ptrdiff_t i) {
		Assert_crit(i >= 0 && i < size_);
		return components_[i];
	}
	const T& operator[](std::ptrdiff_t i) const {
		Assert_crit(i >= 0 && i < size_);
		return components_[i];
	}
	
	std::size_t dimension() const { return size_; }
	std::size_t maximal_dimension() const { return Max_dim; }
	
	iterator begin() { return components_.begin(); }
	const_iterator begin() const { return components_.cbegin(); }
	const_iterator cbegin() const { return components_.cbegin(); }
	iterator end() { return components_.begin() + size_; }
	const_iterator end() const { return components_.cbegin() + size_; }
	const_iterator cend() const { return components_.cbegin() + size_; }
	size_type size() const { return size_; }
	
	template<typename Unary>
	ndcoord_dyn& transform_inplace(Unary fct) {
		for(T& c : *this) c = fct(c);
		return *this;
	}
	
	template<typename Binary>
	ndcoord_dyn& transform_inplace(const ndcoord_dyn& c, Binary fct) {
		for(std::ptrdiff_t i = 0; i < size_; ++i)
			components_[i] = fct(components_[i], c[i]);
		return *this;
	}
	
	ndcoord_dyn& operator+=(const ndcoord_dyn& c) { return transform_inplace(c, std::plus<T>()); }
	ndcoord_dyn& operator-=(const ndcoord_dyn& c) { return transform_inplace(c, std::minus<T>()); }
	ndcoord_dyn& operator*=(const ndcoord_dyn& c) { return transform_inplace(c, std::multiplies<T>()); }
	ndcoord_dyn& operator/=(const ndcoord_dyn& c) { return transform_inplace(c, std::divides<T>()); }

	ndcoord_dyn& operator*=(T val) { return operator*=(ndcoord_dyn(val)); };
	ndcoord_dyn& operator/=(T val) { return operator/=(ndcoord_dyn(val)); };

	ndcoord_dyn operator+() { return *this; }
	ndcoord_dyn operator-() { return transform_inplace(std::negate<T>()); }

	friend ndcoord_dyn operator+(const ndcoord_dyn& a, const ndcoord_dyn& b) noexcept
		{ return transform(a, b, std::plus<T>()); }
	friend ndcoord_dyn operator-(const ndcoord_dyn& a, const ndcoord_dyn& b) noexcept
		{ return transform(a, b, std::minus<T>()); }
	friend ndcoord_dyn operator*(const ndcoord_dyn& a, const ndcoord_dyn& b) noexcept
		{ return transform(a, b, std::multiplies<T>()); }
	friend ndcoord_dyn operator/(const ndcoord_dyn& a, const ndcoord_dyn& b) noexcept
		{ return transform(a, b, std::divides<T>()); }

	friend ndcoord_dyn operator*(const ndcoord_dyn& a, T val) noexcept
		{ return a * ndcoord_dyn(a.dimension(), val); }
	friend ndcoord_dyn operator/(const ndcoord_dyn& a, T val) noexcept
		{ return a / ndcoord_dyn(a.dimension(), val); }
		
	friend bool operator==(const ndcoord_dyn& a, const ndcoord_dyn& b) noexcept
		{ return std::equal(a.cbegin(), a.cend(), b.cbegin()); }
	friend bool operator!=(const ndcoord_dyn& a, const ndcoord_dyn& b) noexcept
		{ return ! std::equal(a.cbegin(), a.cend(), b.cbegin()); }
	
	T product() const {
		T prod = 1;
		for(T c : *this) prod *= c;
		return prod;
	}
	
	const T& front() const { return components_.front(); }
	T& front() { return components_.front(); }
	const T& back() const { return components_[size_ - 1]; }
	T& back() { return components_[size_ - 1]; }	
	
	ndcoord_dyn tail(std::size_t section_dim) const
		{ return ndcoord_dyn(begin() + (size_ - section_dim), end()); }
	ndcoord_dyn tail() const
		{ return tail(size() - 1); }

	ndcoord_dyn head(std::size_t section_dim) const
		{ return ndcoord_dyn(begin(), end() - (size_ - section_dim)); }
	ndcoord_dyn head() const
		{ return head(size() - 1); }

	ndcoord_dyn erase(std::ptrdiff_t i) const {
		ndcoord_dyn result;
		result.size_ = size_ - 1;
		for(std::ptrdiff_t j = 0; j < i; ++j) result[j] = components_[j];
		for(std::ptrdiff_t j = i + 1; j < size_; ++j) result[j - 1] = components_[j];
		return result;
	}
	
	
	
	template<std::size_t Static_dim>
	ndcoord_dyn(const ndcoord<Static_dim, T>& other) : size_(Static_dim) {
		std::copy_n(other.cbegin(), size_, components_.begin());
	}
	
	template<std::size_t Static_dim>
	ndcoord_dyn& operator=(const ndcoord<Static_dim, T>& other) {
		static_assert(Static_dim <= Max_dim, "Static_dim <= Max_dim");
		size_ = Static_dim;
		std::copy_n(other.cbegin(), size_, components_.begin());
		return *this;
	}
};


using ndsize_dyn = ndcoord_dyn<std::size_t>;
using ndptrdiff_dyn = ndcoord_dyn<std::ptrdiff_t>;


template<typename T, typename... Components>
auto make_ndcoord_dyn(Components... c) {
	return ndcoord_dyn<T>({ static_cast<T>(c)... });
}

template<typename... Components>
auto make_ndsize_dyn(Components... c) {
	return make_ndcoord_dyn<std::size_t>(c...);
}

template<typename... Components>
auto make_ndptrdiff_dyn(Components... c) {
	return make_ndcoord_dyn<std::ptrdiff_t>(c...);
}


template<typename T, std::size_t Max_dim, typename Unary>
ndcoord_dyn<T, Max_dim> transform(const ndcoord_dyn<T, Max_dim>& original, Unary fct) {
	ndcoord_dyn<T, Max_dim> result = original;
	for(T& val : result) val = fct(val);
	return result;
}


template<typename T, std::size_t Max_dim, typename Binary>
ndcoord_dyn<T, Max_dim> transform(const ndcoord_dyn<T, Max_dim>& a, const ndcoord_dyn<T, Max_dim>& b, Binary fct) {
	Assert_crit(a.size() == b.size());
	ndcoord_dyn<T, Max_dim> result(a.size());
	for(std::ptrdiff_t i = 0; i < a.size(); ++i) result[i] = fct(a[i], b[i]);
	return result;
}



template<typename T, std::size_t Max_dim>
std::ostream& operator<<(std::ostream& str, const ndcoord_dyn<T, Max_dim>& coord) {
	str << '(';
	for(std::ptrdiff_t i = 0; i < coord.size() - 1; i++) str << coord[i] << ", ";
	if(coord.size() >= 1) str << coord.back();
	str << ')';
	return str;
}



template<typename T, std::size_t Max_dim>
auto tail(const ndcoord_dyn<T, Max_dim>& coord, std::size_t section_dim) { return coord.tail(section_dim); }

template<typename T, std::size_t Max_dim>
auto tail(const ndcoord_dyn<T, Max_dim>& coord) { return coord.tail(); }

template<typename T, std::size_t Max_dim>
auto head(const ndcoord_dyn<T, Max_dim>& coord, std::size_t section_dim) { return coord.tail(section_dim);  }

template<typename T, std::size_t Max_dim>
auto head(const ndcoord_dyn<T, Max_dim>& coord) { return coord.tail();  }



}

#endif
