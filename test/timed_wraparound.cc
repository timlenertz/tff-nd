#include <catch.hpp>
#include "../src/ndarray_wraparound_view.h"
#include "../src/ndarray_timed_wraparound_view.h"
#include "support/ndarray.h"

using namespace tff;
using namespace tff::test;

TEST_CASE("ndarray_timed_wraparound_view", "[nd][ndarray_timed_wraparound_view]") {
	constexpr std::ptrdiff_t l = sizeof(int);
	constexpr std::ptrdiff_t pad = l;
	constexpr std::size_t len = 10 * 3 * 4;
	std::vector<int> raw(len);
	for(int i = 0; i < len; ++i) raw[i] = i;

	ndarray_view<3, int> vw(raw.data(), make_ndsize(10, 3, 4));

	ndarray_wraparound_view<3, int> vw_w = wraparound(
		vw,
		make_ndptrdiff(-3, 2, 1),
		make_ndptrdiff(2, 7, 3),
		make_ndptrdiff(1, 2, -1)
	);
	
	ndarray_timed_wraparound_view<3, int> vw_w_t = timed(vw_w, 100);
	
	SECTION("basics") {
		REQUIRE(vw_w_t.start_time() == 100);
		REQUIRE(vw_w_t.end_time() == 105);
		REQUIRE(vw_w_t.duration() == 5);
		
		ndarray_wraparound_view<3, int> vw_w_re = vw_w_t.non_timed();
		REQUIRE(same(vw_w, vw_w_re));
		
		REQUIRE(vw_w_t.time_to_coordinate(102) == 2);
		REQUIRE(vw_w_t.coordinate_to_time(2) == 102);
	}
	
	
	SECTION("same, reset") {
		ndarray_wraparound_view<3, int> vw_w2 = wraparound(
			vw,
			make_ndptrdiff(-2, 2, 1),
			make_ndptrdiff(3, 7, 3),
			make_ndptrdiff(1, 2, -1)
		);

		ndarray_timed_wraparound_view<3, int> vw_w_t2 = timed(vw_w, 200);
		ndarray_timed_wraparound_view<3, int> vw_w_t3 = timed(vw_w2, 100);
		REQUIRE_FALSE(same(vw_w_t, vw_w_t2));
		REQUIRE_FALSE(same(vw_w_t, vw_w_t3));
		vw_w_t2.reset(vw_w_t);
		REQUIRE(same(vw_w_t, vw_w_t2));
		vw_w_t3.reset(vw_w_t);
		REQUIRE(same(vw_w_t, vw_w_t3));
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

	
