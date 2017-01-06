#include <catch.hpp>
#include <vector>
#include "../src/ndarray_view.h"
#include "../src/ndarray_wraparound_view.h"
#include "support/ndarray.h"

using namespace tff;
using namespace tff::test;

TEST_CASE("ndarray_wraparound_view", "[nd][ndarray_wraparound_view]") {
	constexpr std::ptrdiff_t l = sizeof(int);
	constexpr std::ptrdiff_t pad = l;
	constexpr std::size_t len = 3 * 4 * 4;
	std::vector<int> raw(len);
	for(int i = 0; i < len; ++i) raw[i] = i;

	ndarray_view<3, int> arr3(raw.data(), make_ndsize(3, 4, 4));
	ndarray_view<1, int> row0 = arr3[0][0]; // 00 01 02 03
	ndarray_view<1, int> row1 = arr3[0][1]; // 04 05 06 07
	ndarray_view<1, int> col0 = arr3[0].slice(0, 1); // 00 04 08 0C
	ndarray_view<1, int> col1 = arr3[1].slice(2, 1); // 12 16 1A 1E
	
	REQUIRE(compare_sequence_(arr3, {
		0x00, 0x01, 0x02, 0x03,
		0x04, 0x05, 0x06, 0x07,
		0x08, 0x09, 0x0a, 0x0b,
		0x0c, 0x0d, 0x0e, 0x0f,
		
		0x10, 0x11, 0x12, 0x13,
		0x14, 0x15, 0x16, 0x17,
		0x18, 0x19, 0x1a, 0x1b,
		0x1c, 0x1d, 0x1e, 0x1f,
		
		0x20, 0x21, 0x22, 0x23,
		0x24, 0x25, 0x26, 0x27,
		0x28, 0x29, 0x2a, 0x2b,
		0x2c, 0x2d, 0x2e, 0x2f
	}));
	REQUIRE(compare_sequence_(row0, { 0x00, 0x01, 0x02, 0x03 }));
	REQUIRE(compare_sequence_(row1, { 0x04, 0x05, 0x06, 0x07 }));
	REQUIRE(compare_sequence_(col0, { 0x00, 0x04, 0x08, 0x0c }));
	REQUIRE(compare_sequence_(col1, { 0x12, 0x16, 0x1a, 0x1e }));
	
	SECTION("1D") {
		ndarray_wraparound_view<1, int> row0_w1 = row0.wraparound_view(make_ndptrdiff(-2), make_ndptrdiff(10));
		REQUIRE(row0_w1.shape() == make_ndsize(12));
		REQUIRE(row0_w1[0] == 0x02);
		REQUIRE(row0_w1[1] == 0x03);
		REQUIRE(row0_w1[2] == 0x00);
		REQUIRE(row0_w1[3] == 0x01);
		REQUIRE(row0_w1[4] == 0x02);
		REQUIRE(row0_w1[5] == 0x03);
		REQUIRE(row0_w1[6] == 0x00);
		REQUIRE(compare_sequence_(row0_w1, {
			            0x02, 0x03,
			0x00, 0x01, 0x02, 0x03,
			0x00, 0x01, 0x02, 0x03,
			0x00, 0x01
		}));
		
		ndarray_wraparound_view<1, int> col1_w1 = col1.wraparound_view(make_ndptrdiff(-2), make_ndptrdiff(10));
		REQUIRE(compare_sequence_(col1_w1, {
			            0x1a, 0x1e,
			0x12, 0x16, 0x1a, 0x1e,
			0x12, 0x16, 0x1a, 0x1e,
			0x12, 0x16
		}));
		
		col1_w1[1] = 123;
		REQUIRE(col1_w1[1] == 123);
		REQUIRE(col1_w1[5] == 123);
		REQUIRE(col1_w1[9] == 123);
		
		REQUIRE_FALSE(row0_w1 == col1_w1);
		REQUIRE(row0_w1 != col1_w1);
		
		col1_w1 = row0_w1;
		REQUIRE(row0_w1 == col1_w1);
		REQUIRE_FALSE(row0_w1 != col1_w1);
		
		REQUIRE(compare_sequence_(arr3, {
			0x00, 0x01, 0x02, 0x03,
			0x04, 0x05, 0x06, 0x07,
			0x08, 0x09, 0x0a, 0x0b,
			0x0c, 0x0d, 0x0e, 0x0f,
			
			0x10, 0x11, 0x00, 0x13,
			0x14, 0x15, 0x01, 0x17,
			0x18, 0x19, 0x02, 0x1b,
			0x1c, 0x1d, 0x03, 0x1f,
			
			0x20, 0x21, 0x22, 0x23,
			0x24, 0x25, 0x26, 0x27,
			0x28, 0x29, 0x2a, 0x2b,
			0x2c, 0x2d, 0x2e, 0x2f
		}));
		
		// subsection
		REQUIRE(compare_sequence_(row0_w1(1, 9), { 0x03, 0x00, 0x01, 0x02, 0x03, 0x00, 0x01, 0x02 }));
		REQUIRE(compare_sequence_(row0_w1(1, 9, -1), { 0x02, 0x01, 0x00, 0x03, 0x02, 0x01, 0x00, 0x03 }));
		
		REQUIRE(compare_sequence_(row0_w1(1, 9, 2), { 0x03, 0x01, 0x03, 0x01 }));
		REQUIRE(compare_sequence_(row0_w1(1, 9, -2), { 0x01, 0x03, 0x01, 0x03 }));
	}
	
	SECTION("1D stride") {
		ndarray_wraparound_view<1, int> row0_w2 = row0.wraparound_view(make_ndptrdiff(-2), make_ndptrdiff(11), make_ndptrdiff(3));
		REQUIRE(row0_w2.shape() == make_ndsize(5));
		REQUIRE(compare_sequence_(row0_w2, {
			0x02, 0x01, 0x00, 0x03, 0x02
		}));
		
		ndarray_wraparound_view<1, int> col1_w2 = col1.wraparound_view(make_ndptrdiff(0), make_ndptrdiff(8), make_ndptrdiff(2));
		REQUIRE(compare_sequence_(col1_w2, {
			0x12, 0x1a, 0x12, 0x1a
		}));
		
		REQUIRE(compare_sequence_(col1_w2(1, 3), { 0x1a, 0x12 }));
		REQUIRE(compare_sequence_(col1_w2(1, 4, 2), { 0x1a, 0x1a }));
	}
	
	SECTION("1D reverse") {
		ndarray_wraparound_view<1, int> row0_w3 = row0.wraparound_view(make_ndptrdiff(-2), make_ndptrdiff(10), make_ndptrdiff(-1));
		REQUIRE(row0_w3.shape() == make_ndsize(12));
		REQUIRE(compare_sequence_(row0_w3, {
			            0x01, 0x00,
			0x03, 0x02, 0x01, 0x00,
			0x03, 0x02, 0x01, 0x00,
			0x03, 0x02
		}));
		
		REQUIRE(compare_sequence_(row0_w3(3, 9), { 0x02, 0x01, 0x00, 0x03, 0x02, 0x01 }));
		REQUIRE(compare_sequence_(row0_w3(3, 9, -1), { 0x01, 0x02, 0x03, 0x00, 0x01, 0x02 }));
		REQUIRE(compare_sequence_(row0_w3(3, 9, 2), { 0x02, 0x00, 0x02 }));
		REQUIRE(compare_sequence_(row0_w3(3, 9, -2), { 0x02, 0x00, 0x02 }));
	}
}