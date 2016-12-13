#ifndef TFF_ELEM_H_
#define TFF_ELEM_H_

#include "config.h"
#if TFF_ND_WITH_ELEM

#include <cstddef>
#include <array>
#include <complex>
#include <type_traits>
#include "common.h"

namespace tff {

/// Elem traits base class with the required members.
template<typename Elem, typename Scalar = Elem, std::size_t Components = 1, bool Nullable = false>
struct elem_traits_base {
	static_assert(std::is_standard_layout<Elem>::value, "elem must be standard layout type");
	static_assert(std::is_standard_layout<Scalar>::value, "elem scalar must be standard layout type");
	
	using scalar_type = Scalar;
	constexpr static bool is_tuple = false;
	constexpr static std::size_t components = Components;
	constexpr static std::size_t stride = sizeof(Scalar);
	constexpr static bool is_nullable = Nullable;
};


/// Default elem traits, using Elem as standard layout scalar type.
/** `Elem` must be standard layout type. */
template<typename Elem>
struct elem_traits : elem_traits_base<Elem> { };


/// Elem traits specialization for `std::array<T, N>`.
/** `T` must be standard layout type. */
template<typename T, std::size_t N>
struct elem_traits<std::array<T, N>> :
	elem_traits_base<std::array<T, N>, T, N> { };


/// Elem traits specialization for `std::complex<T>`.
template<typename T>
struct elem_traits<std::complex<T>> :
	elem_traits_base<std::complex<T>, T, 2> { };


/// Type for null element.
class nullelem_t {
public:
	constexpr explicit nullelem_t(int) { }
};


/// Null element constant.
/** Can be assigned to, or used to construct nullable element types. */
constexpr nullelem_t nullelem { 0 };


template<typename Elem> inline bool operator==(const Elem& elem, nullelem_t) { return is_null(elem); }
template<typename Elem> inline bool operator==(nullelem_t, const Elem& elem) { return is_null(elem); }
template<typename Elem> inline bool operator==(nullelem_t, nullelem_t) { return true; }
template<typename Elem> inline bool operator!=(const Elem& elem, nullelem_t) { return not is_null(elem); }
template<typename Elem> inline bool operator!=(nullelem_t, const Elem& elem) { return not is_null(elem); }
template<typename Elem> inline bool operator!=(nullelem_t, nullelem_t) { return false; }


/// Test if elem is null.
template<typename Elem>
std::enable_if_t<elem_traits<Elem>::is_nullable, bool> is_null(const Elem& elem) {
	return elem.is_null();
}


/// Test if elem is null.
template<typename Elem>
std::enable_if_t<! elem_traits<Elem>::is_nullable, bool> is_null(const Elem& elem) {
	return false;
}

}

#endif
#endif
