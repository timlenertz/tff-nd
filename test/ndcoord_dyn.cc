#include <catch.hpp>
#include "../src/ndcoord_dyn.h"
#include "../src/ndcoord.h"

using namespace tff;


TEST_CASE("ndcoord_dyn_dyn", "[nd][ndcoord_dyn_dyn]") {
	SECTION("construction") {
		ndcoord_dyn<int> zero;
		REQUIRE(zero.dimension() == 0);
		
		
		ndcoord_dyn<int> constant(3, 7);
		REQUIRE(constant.dimension() == 3);
		REQUIRE(constant[0] == 7);
		REQUIRE(constant[1] == 7);
		REQUIRE(constant[2] == 7);
				
		ndcoord_dyn<int> list{4, 5, 6};
		REQUIRE(list.dimension() == 3);
		REQUIRE(list[0] == 4);
		REQUIRE(list[1] == 5);
		REQUIRE(list[2] == 6);
		
		ndcoord_dyn<int> copy = list;
		REQUIRE(copy.dimension() == 3);
		REQUIRE(copy[0] == 4);
		REQUIRE(copy[1] == 5);
		REQUIRE(copy[2] == 6);
		
		ndcoord_dyn<unsigned short> conv = copy;
		REQUIRE(conv.dimension() == 3);
		REQUIRE(conv[0] == 4);
		REQUIRE(conv[1] == 5);
		REQUIRE(conv[2] == 6);

		auto helper = make_ndcoord_dyn<int>(3, 5, 7);
		REQUIRE(helper.dimension() == 3);
		REQUIRE(helper[0] == 3);
		REQUIRE(helper[1] == 5);
		REQUIRE(helper[2] == 7);
		
		auto helper_size = make_ndsize_dyn(3, 5, 7);
		REQUIRE(helper_size.dimension() == 3);
		REQUIRE(helper_size[0] == 3);
		REQUIRE(helper_size[1] == 5);
		REQUIRE(helper_size[2] == 7);

		auto helper_ptrdiff = make_ndptrdiff_dyn(-3, 5, 7);
		REQUIRE(helper_ptrdiff.dimension() == 3);
		REQUIRE(helper_ptrdiff[0] == -3);
		REQUIRE(helper_ptrdiff[1] == 5);
		REQUIRE(helper_ptrdiff[2] == 7);	
	}
	
	SECTION("basics") {
		ndcoord_dyn<int> coord{1, 2, 3};
		ndcoord_dyn<int> coord2{4, 5, 6};

		// subscript
		REQUIRE(coord[1] == 2);
		coord[1] = 4;
		REQUIRE(coord[1] == 4);
		REQUIRE(coord.front() == 1);
		REQUIRE(coord.back() == 3);
		
		// compare, assign
		REQUIRE_FALSE(coord == coord2);
		REQUIRE(coord != coord2);
	
		coord = coord2;
		REQUIRE(coord[0] == 4);
		REQUIRE(coord[1] == 5);
		REQUIRE(coord[2] == 6);
		REQUIRE(coord == coord2);
		REQUIRE_FALSE(coord != coord2);
		
		// product
		REQUIRE(coord.product() == 4*5*6);
		
		ndcoord_dyn<int> zero;
		REQUIRE(zero.product() == 1);
	}
	
	SECTION("iterator") {
		auto c = make_ndsize_dyn(1, 2, 3, 4);
		
		std::vector<int> vec(c.begin(), c.end());
		REQUIRE(vec.size() == 4);
		REQUIRE(vec[0] == 1);
		REQUIRE(vec[1] == 2);
		REQUIRE(vec[2] == 3);
		REQUIRE(vec[3] == 4);
		
		REQUIRE(c.size() == 4);
	}
	
	SECTION("section, cat") {
		auto c1 = make_ndsize_dyn(1, 2, 3, 4);
		
		REQUIRE(c1.head() == make_ndsize_dyn(1, 2, 3));
		REQUIRE(c1.tail() == make_ndsize_dyn(2, 3, 4));
		
		REQUIRE(c1.erase(1) == make_ndsize_dyn(1, 3, 4));
		REQUIRE(c1.erase(0) == c1.tail());
		REQUIRE(c1.erase(3) == c1.head());
	}
	
	SECTION("transform") {
		ndsize_dyn c1{1, 2, 3}, c2{4, 5, 6};
		
		SECTION("transform_inplace") {
			c1.transform_inplace([](int i) { return i + 1; });
			REQUIRE(c1 == make_ndsize_dyn(2, 3, 4));
		
			c1.transform_inplace(c2, [](int a, int b) { return a * b; });
			REQUIRE(c1 == make_ndsize_dyn(2*4, 3*5, 4*6));
		}
		
		SECTION("plus") {
			REQUIRE(c1 + c2 == make_ndsize_dyn(1+4, 2+5, 3+6));
			c1 += c2;
			REQUIRE(c1 == make_ndsize_dyn(1+4, 2+5, 3+6));
		}
		
		SECTION("minus") {
			REQUIRE(c1 - c2 == make_ndsize_dyn(1-4, 2-5, 3-6));
			c1 -= c2;
			REQUIRE(c1 == make_ndsize_dyn(1-4, 2-5, 3-6));
		}

		SECTION("multiply") {
			REQUIRE(c1 * c2 == make_ndsize_dyn(1*4, 2*5, 3*6));
			c1 *= c2;
			REQUIRE(c1 == make_ndsize_dyn(1*4, 2*5, 3*6));
		}
		
		SECTION("divide") {
			REQUIRE(c2 / c1 == make_ndsize_dyn(4/1, 5/2, 6/3));
			c2 /= c1;
			REQUIRE(c2 == make_ndsize_dyn(4/1, 5/2, 6/3));
		}
		
		SECTION("unary") {
			REQUIRE(+c1 == c1);
			REQUIRE(-c1 == make_ndsize_dyn(-1, -2, -3));
		}
	}
}


TEST_CASE("ndcoord <-> ndcoord_dyn", "[nd][ndcoord][ndcoord_dyn]") {
	auto sta = make_ndsize(1, 2, 3);
	auto dyn = make_ndsize_dyn(4, 5, 6);
	
	REQUIRE_FALSE(sta == dyn);
	REQUIRE_FALSE(dyn == sta);
	REQUIRE(sta != dyn);
	REQUIRE(dyn != sta);
	
	dyn = sta;
	REQUIRE(sta == dyn);
	REQUIRE(dyn == sta);
	REQUIRE_FALSE(sta != dyn);
	REQUIRE_FALSE(dyn != sta);
}
