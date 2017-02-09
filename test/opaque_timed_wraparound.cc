#include <catch.hpp>
#include "../src/opaque/ndarray_wraparound_opaque_view.h"
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

	ndarray_wraparound_opaque_view<3, true, opaque_raw_format> vw_w = wraparound(
		vw,
		make_ndptrdiff(-3, 2, 1),
		make_ndptrdiff(2, 7, 3),
		make_ndptrdiff(1, 2, -1)
	);

	ndarray_timed_wraparound_opaque_view<3, true, opaque_raw_format> vw_w_t = timed(vw_w, 100);
	
	SECTION("basics") {
		REQUIRE(vw_w_t.start_time() == 100);
		REQUIRE(vw_w_t.end_time() == 105);
		REQUIRE(vw_w_t.duration() == 5);
		
		ndarray_wraparound_opaque_view<3, true, opaque_raw_format> vw_w_re = vw_w_t.non_timed();
		REQUIRE(same(vw_w, vw_w_re));
		
		REQUIRE(vw_w_t.time_to_coordinate(102) == 2);
		REQUIRE(vw_w_t.coordinate_to_time(2) == 102);
	}


	SECTION("section, indexing") {
		REQUIRE(same(vw_w_t.at_time(101), vw_w[1]));
		REQUIRE(same(vw_w_t.at_time(104), vw_w[4]));
		
		REQUIRE(same(vw_w_t.tsection(101, 104), vw_w(1, 4)));
		REQUIRE(vw_w_t(2, 4)()(0, 1).start_time() == 102);
		REQUIRE(vw_w_t(3)(0)(1).start_time() == 103);
		REQUIRE(vw_w_t()(1, 3)(1).start_time() == 100);
	}
}

	
