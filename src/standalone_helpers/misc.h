#ifndef TFF_ND_MISC_H_
#define TFF_ND_MISC_H_

namespace tff {

using time_unit = std::ptrdiff_t;
constexpr static time_unit undefined_time = -1;

using byte = std::uint8_t;

	
template<typename...> using void_t = void;

template<bool Const, typename T> using const_if = std::conditional_t<Const, const T, T>;

template<typename T> T* advance_raw_ptr(T* ptr, std::ptrdiff_t diff) {
	std::uintptr_t raw_ptr = reinterpret_cast<std::uintptr_t>(ptr);
	raw_ptr += diff;
	return reinterpret_cast<T*>(raw_ptr);
}

template<typename T> bool is_aligned(T* ptr, std::size_t alignment_requirement) {
	std::uintptr_t raw_ptr = reinterpret_cast<std::uintptr_t>(ptr);
	return (raw_ptr % alignment_requirement == 0);
}

template<typename T>
bool is_power_of_two(T x) {
	return (x != 0) && !(x & (x - 1));
}

template<typename T, typename T2>
bool is_multiple_of(T x, T2 base) {
	return (x % base == 0);
}

template<typename T>
T round_up(T x, T base) {
	T remainder = x % base;
	if(remainder == 0) return x;
	else return x + (base - remainder);
}

template<typename T, typename T2>
bool is_nonzero_multiple_of(T x, T2 base) {
	return (x != 0) && is_multiple_of(x, base);
}

template<typename T>
bool is_odd(T x) { return (x % 2) != 0; }

template<typename T>
bool is_even(T x) { return (x % 2) == 0; }

}

#endif
