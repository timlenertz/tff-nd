#include <catch.hpp>
#include <array>
#include "../src/ndarray_view.h"
#include "../src/pod_array_format.h"
#include "support/ndarray.h"

using namespace tlz;
using namespace tlz::test;

constexpr std::size_t len = 3 * 4 * 4;
ndsize<3> shp{3, 4, 4};


template<std::size_t Elem_size, std::size_t Padding_size>
void test_non_contiguous_() {
	struct elem_t {
		std::array<byte, Elem_size> data;
		bool operator==(const elem_t& other) const
			{ return (data == other.data); }
	};
	struct padded_elem_t {
		elem_t elem;
		std::array<byte, Padding_size> padding;
		bool operator==(const padded_elem_t& other) const
			{ return (elem.data == other.elem.data); }
	};
	REQUIRE(sizeof(elem_t) == Elem_size);
	REQUIRE(sizeof(padded_elem_t) == Elem_size + Padding_size);
	pod_array_format frm = make_pod_array_format<elem_t>(len, Elem_size + Padding_size);
	REQUIRE_FALSE(frm.is_contiguous());
	
	std::vector<padded_elem_t> raw1(len), raw2(len);
	for(int i = 0; i < len; ++i) {
		for(int j = 0; j < Elem_size; ++j) {
			raw1[i].elem.data[j] = i + j;
			raw2[i].elem.data[j] = 2*i + 1 + j;
		}
		raw1[i].padding.fill(123);
		raw2[i].padding.fill(231);
	}
	
	REQUIRE(pod_array_compare(
		static_cast<const void*>(raw1.data()),
		static_cast<const void*>(raw1.data()),
		frm
	));
	REQUIRE_FALSE(pod_array_compare(
		static_cast<const void*>(raw1.data()),
		static_cast<const void*>(raw2.data()),
		frm
	));
	pod_array_copy(
		static_cast<void*>(&raw2[0]),
		static_cast<const void*>(raw1.data()),
		frm
	);
	REQUIRE(raw1 == raw2);
	for(int i = 0; i < len; ++i) {
		std::array<byte, Padding_size> expected_pad; expected_pad.fill(123);
		REQUIRE(raw1[i].padding == expected_pad);
	}
	REQUIRE(pod_array_compare(
		static_cast<const void*>(raw1.data()),
		static_cast<const void*>(raw2.data()),
		frm
	));
	raw1[2].padding[0] = 77;
	raw2[5].padding[0] = 99;
	REQUIRE(pod_array_compare(
		static_cast<const void*>(raw1.data()),
		static_cast<const void*>(raw2.data()),
		frm
	));
}


TEST_CASE("pod_array_format", "[nd][pod_array_format]") {	
	SECTION("data assignment and comparison") {
		SECTION("contiguous") {
			std::vector<int> raw1(len), raw2(len);
			for(int i = 0; i < len; ++i) {
				raw1[i] = i;
				raw2[i] = 2*i + 1;
			}
			pod_array_format frm = make_pod_array_format<int>(len);
			REQUIRE(frm.is_contiguous());
			
			REQUIRE(pod_array_compare(
				static_cast<const void*>(raw1.data()),
				static_cast<const void*>(raw1.data()),
				frm
			));
			REQUIRE_FALSE(pod_array_compare(
				static_cast<const void*>(raw1.data()),
				static_cast<const void*>(raw2.data()),
				frm
			));
			pod_array_copy(
				static_cast<void*>(&raw2[0]),
				static_cast<const void*>(raw1.data()),
				frm
			);
			REQUIRE(raw1 == raw2);
			REQUIRE(pod_array_compare(
				static_cast<const void*>(raw1.data()),
				static_cast<const void*>(raw2.data()),
				frm
			));
		}
		

		SECTION("non-contiguous") {
			SECTION("int8 boundary") { test_non_contiguous_<1, 5>(); }
			SECTION("int16 boundary") { test_non_contiguous_<2, 8>(); }
			SECTION("int32 boundary") { test_non_contiguous_<4, 4>(); }
			SECTION("int64 boundary") { test_non_contiguous_<8, 16>(); }
			SECTION("irregular") { test_non_contiguous_<105, 13>(); }
			SECTION("not int32 boundary") { test_non_contiguous_<4, 5>(); }
		}
	}
	
	SECTION("non-pod") {
		REQUIRE_THROWS(make_pod_array_format<obj_t>(100));
	}
}


TEST_CASE("pod_array_format coverage", "[nd][pod_array_format]") {	
	auto req = [](const pod_array_format& a, const pod_array_format& b) {
		REQUIRE(same_coverage(a, b));
		REQUIRE(same_coverage(b, a));
	};
	auto req_false = [](const pod_array_format& a, const pod_array_format& b) {
		REQUIRE_FALSE(same_coverage(a, b));
		REQUIRE_FALSE(same_coverage(b, a));
	};
	
	constexpr std::size_t l = sizeof(int);
	
	SECTION("arbitray") {
		req(make_pod_array_format<int>(10), make_pod_array_format<int>(10));
		req(make_pod_array_format<int>(10), make_pod_array_format<int>(10, 3*l, 3*l));
		req(make_pod_array_format<int>(10), make_pod_array_format<int>(10));
		req_false(make_pod_array_format<int>(10), make_pod_array_format<int>(11));
		req_false(make_pod_array_format<int>(10), make_pod_array_format<char>(11));
		req(make_pod_array_format<int>(10), make_pod_array_format<int>(10, l));
		req_false(make_pod_array_format<int>(10), make_pod_array_format<byte>(10, l));
	}
	
	SECTION("zero") {
		req(make_pod_array_format<int>(0), make_pod_array_format<byte>(0, 3));
		req(make_pod_array_format<int>(0), make_pod_array_format<byte>(0, 32, 32));
	}
	
	SECTION("contiguous") {
		req(make_pod_array_format<int>(10), make_pod_array_format<byte>(10*l));
	}
	
	SECTION("single elem") {
		req(make_pod_array_format<int>(1), make_pod_array_format<byte>(l));
		req(make_pod_array_format<int>(1, l), make_pod_array_format<byte>(l));
		req(make_pod_array_format<int>(1, 4*l, 4*l), make_pod_array_format<byte>(l));
		req(make_pod_array_format<int>(1, 4*l), make_pod_array_format<int>(1, 2*l));
		req(make_pod_array_format<int>(1, 4*l, 4*l), make_pod_array_format<int>(1, 4*l, l));
		req_false(make_pod_array_format<int>(1), make_pod_array_format<byte>(l+1));
	}
}
