#include <catch.hpp>
#include "../src/ndarray_timed_view.h"
#include "support/ndarray.h"

using namespace tff;
using namespace tff::test;


TEST_CASE("ndarray_timed_view", "[nd][ndarray_timed_view]") {
	auto shp = make_ndsize(10, 3, 4);
	auto len = shp.product();
	std::vector<int> raw(len);
	for(int i = 0; i < len; ++i) raw[i] = i;

	ndarray_view<3, int> a1nt(raw.data(), shp);
	ndarray_timed_view<3, int> a1 = timed(a1nt, 100);

	SECTION("basics") {
		REQUIRE(a1.start() == raw.data());
		REQUIRE(a1.shape() == shp);
		REQUIRE(a1.has_default_strides());
		REQUIRE(a1.default_strides_padding() == 0);
		REQUIRE(a1.has_default_strides_without_padding());
		REQUIRE(a1.start_time() == 100);
		REQUIRE(a1.end_time() == 110);
		REQUIRE(a1.duration() == 10);
		REQUIRE(a1.tspan() == time_span(100, 110));
		
		ndarray_view<3, int> a1nt_re = a1.non_timed();
		REQUIRE(same(a1nt_re, a1nt));
		REQUIRE(a1.time_to_coordinate(102) == 2);
		REQUIRE(a1.coordinate_to_time(2) == 102);
	}
	
	SECTION("section, indexing") {
		REQUIRE(same(a1.at_time(105), a1[5]));
		REQUIRE(same(a1.tsection(time_span(101, 103)), a1(1, 3)));
		REQUIRE(same(
			a1.section(make_ndspan(make_ndptrdiff(1, 0, 1), make_ndptrdiff(4, 1, 2))),
			a1nt.section(make_ndspan(make_ndptrdiff(1, 0, 1), make_ndptrdiff(4, 1, 2)))
		));
		REQUIRE(a1.section(make_ndspan(make_ndptrdiff(1, 0, 1), make_ndptrdiff(4, 1, 2))).start_time() == 101);
		REQUIRE(same(
			a1.section(make_ndptrdiff(3, 0, 1), make_ndptrdiff(5, 1, 2)),
			a1nt.section(make_ndptrdiff(3, 0, 1), make_ndptrdiff(5, 1, 2))			
		));
		REQUIRE(a1.section(make_ndptrdiff(3, 0, 1), make_ndptrdiff(5, 1, 2)).start_time() == 103);
		REQUIRE(a1(2, 8)()(0, 1).start_time() == 102);
		REQUIRE(a1(3)(0)(3).start_time() == 103);
		REQUIRE(a1()(1, 2)(2).start_time() == 100);
	}
}
