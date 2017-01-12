#include <catch.hpp>
#include "../src/opaque/ndarray_timed_wraparound_opaque_view.h"
#include "support/ndarray.h"

using namespace tff;
using namespace tff::test;

TEST_CASE("ndarray_timed_wraparound_opaque_view", "[nd][ndarray_timed_wraparound_opaque_view]") {
	auto shp = make_ndsize(10, 3, 4);
	auto len = shp.product();
	std::vector<std::uint32_t> raw(len);
	for(int i = 0; i < len; ++i) raw[i] = i;

	opaque_raw_format frm(4);
	ndarray_opaque_view<3, true, opaque_raw_format> vw(
		raw.data(),
		shp,
		ndarray_opaque_view<3, true, opaque_raw_format>::default_strides(shp, frm),
		frm
	);

	ndarray_wraparound_opaque_view<3, true, opaque_raw_format> vw_w1 = wraparound(
		vw,
		make_ndptrdiff(-3, 2, 1),
		make_ndptrdiff(2, 7, 3),
		make_ndptrdiff(1, 2, -1)
	);

	ndarray_timed_wraparound_opaque_view<3, true, opaque_raw_format> vw_w1_t = timed(vw_w1, 100);
	
	SECTION("basics") {
		REQUIRE(vw_w1_t.start_time() == 100);
		REQUIRE(vw_w1_t.end_time() == 105);
		REQUIRE(vw_w1_t.duration() == 5);
		REQUIRE(vw_w1_t.tspan() == time_span(100, 105));
		
		ndarray_wraparound_opaque_view<3, true, opaque_raw_format> vw_w1_re = vw_w1_t.non_timed();
		REQUIRE(same(vw_w1, vw_w1_re));
		
		REQUIRE(vw_w1_t.time_to_coordinate(102) == 2);
		REQUIRE(vw_w1_t.coordinate_to_time(2) == 102);
	}


	SECTION("section, indexing") {
		REQUIRE(same(vw_w1_t.at_time(101), vw_w1[1]));
		REQUIRE(same(vw_w1_t.at_time(101), vw[8]));
		REQUIRE(same(vw_w1_t.at_time(104), vw_w1[4]));
		REQUIRE(same(vw_w1_t.at_time(104), vw[1]));
		
		REQUIRE(same(vw_w1_t.tsection(time_span(101, 104)), vw(1, 4)));
		REQUIRE(a1(2, 8)()(0, 1).start_time() == 102);
		REQUIRE(a1(3)(0)(3).start_time() == 103);
		REQUIRE(a1()(1, 2)(2).start_time() == 100);
	}
}

	
