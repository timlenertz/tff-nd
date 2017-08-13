#include <catch.hpp>
#include <algorithm>
#include "../src/ndarray.h"
#include "../src/ndarray_view.h"
#include "support/ndarray.h"

using namespace tlz;
using namespace tlz::test;

static void verify_ndarray_memory_(ndarray<3, int>& arr) {
	const auto& shp = arr.shape();
	int i = 0;
	for(const auto& coord : make_ndspan(shp)) arr.at(coord) = i++;
	i = 0;
	for(const auto& coord : make_ndspan(shp)) REQUIRE(arr.at(coord) == i++);
}


TEST_CASE("ndarray", "[nd][ndarray]") {
	constexpr std::ptrdiff_t l = sizeof(int);
	constexpr std::ptrdiff_t pad = l;
	ndsize<3> shape{3, 4, 4};
	std::vector<int> raw(shape.product());
	for(int i = 0; i < raw.size(); ++i) raw[i] = i;
	ndarray_view<3, int> arr_vw(raw.data(), shape);

	ndarray_view<3, int> arr_vw_sec = arr_vw()(0, 4, 2)(0, 4, 2);
	REQUIRE_FALSE(arr_vw_sec.has_default_strides());

	SECTION("construction") {
		// construction with shape
		ndarray<3, int> arr(shape);
		REQUIRE(arr.shape() == shape);
		REQUIRE((arr.strides() == ndarray_view<3, int>::default_strides(shape)));
		verify_ndarray_memory_(arr);
		REQUIRE(arr.allocated_byte_size() >= arr_vw.size()*l);

		// construction with shape, padding
		ndarray<3, int> arr_pad(shape, pad);
		REQUIRE(arr_pad.shape() == shape);
		REQUIRE((arr_pad.strides() == ndarray_view<3, int>::default_strides(shape, pad)));
		verify_ndarray_memory_(arr_pad);

		// construction from view (ndarray gets default strides)
		ndarray<3, int> arr2(arr_vw_sec);
		REQUIRE(arr2.view().compare(arr_vw_sec));
		REQUIRE(arr2.shape() == arr_vw_sec.shape());
		REQUIRE((arr2.strides() == ndarray_view<3, int>::default_strides(arr_vw_sec.shape())));
		verify_ndarray_memory_(arr2);
		arr2[1][1][1] = 456;REQUIRE(arr2[1][1][1] == 456);
		REQUIRE_FALSE(arr_vw[1][1][1] == 456);
		
		// construction from view (ndarray gets padded default strides)
		ndarray<3, int> arr3(arr_vw_sec, pad);
		REQUIRE(arr3.view().compare(arr_vw_sec));
		REQUIRE(arr3.shape() == arr_vw_sec.shape());
		REQUIRE((arr3.strides() == ndarray_view<3, int>::default_strides(arr_vw_sec.shape(), pad)));
		verify_ndarray_memory_(arr3);
		arr3[1][1][1] = 456;
		REQUIRE(arr3[1][1][1] == 456);
		REQUIRE_FALSE(arr_vw[1][1][1] == 456);
		
		// construction from view with different type
		constexpr std::ptrdiff_t pad_f = 2*sizeof(float);
		ndarray<3, float> arr3_f(arr_vw_sec, pad_f);
		REQUIRE(arr3_f.view().compare(arr_vw_sec));
		REQUIRE(arr3_f.shape() == arr_vw_sec.shape());
		REQUIRE((arr3_f.strides() == ndarray_view<3, float>::default_strides(arr_vw_sec.shape(), pad_f)));

		// construction from null ndarray_view
		REQUIRE_THROWS((new ndarray<3, int>(ndarray_view<3, int>::null())));

		// copy-construction from another ndarray (strides get copied)
		ndarray<3, int> arr4 = arr3;
		REQUIRE(arr4.view().compare(arr3));
		REQUIRE(arr4.shape() == arr3.shape());
		REQUIRE(arr4.strides() == arr3.strides());
		REQUIRE(arr4[1][1][1] == 456);
		verify_ndarray_memory_(arr4);
		arr4[1][1][1] = 789;
		REQUIRE(arr4[1][1][1] == 789);
		REQUIRE_FALSE(arr3[1][1][1] == 789);
		
		// move construction from another ndarray (strides get copied)
		ndarray<3, int> arr5_cmp = arr4;
		ndarray<3, int> arr5 = std::move(arr4);
		REQUIRE(arr5.view().compare(arr5_cmp));
		REQUIRE(arr5.shape() == arr5_cmp.shape());
		REQUIRE(arr5.strides() == arr5_cmp.strides());
		REQUIRE(arr4.allocated_byte_size() == 0);
		verify_ndarray_memory_(arr5);

		// construction using initializer_list
		ndarray<2, int> arr6({
			{1, 2, 3},
			{4, 5, 6}
		});
		REQUIRE(arr6.shape() == make_ndsize(2, 3));
		REQUIRE(arr6[0][0] == 1);
		REQUIRE(arr6[1][1] == 5);
		
		// invalid initializer_list
		REQUIRE_THROWS(( new ndarray<2, int>{{1}, {1, 2}}  ));
		REQUIRE_THROWS(( new ndarray<2, int>{{}, {}}  ));
	};


	SECTION("assignment") {
		ndsize<3> previous_shape{5, 1, 1};
		ndsize<3> shp = arr_vw_sec.shape();
		
		// assignment from view (ndarray gets default strides)
		ndarray<3, int> arr_(previous_shape), arr_2(previous_shape);
		arr_ = arr_vw_sec;
		REQUIRE(arr_.view().compare(arr_vw_sec));
		REQUIRE(arr_.shape() == arr_vw_sec.shape());
		REQUIRE((arr_.strides() == ndarray_view<3, int>::default_strides(shp)));
		arr_[1][1][1] = 123; REQUIRE_FALSE(arr_vw[1][1][1] == 123);
		arr_2.assign(arr_vw_sec);
		REQUIRE(arr_2.view().compare(arr_vw_sec));
		REQUIRE(arr_2.shape() == arr_vw_sec.shape());
		REQUIRE((arr_2.strides() == ndarray_view<3, int>::default_strides(shp)));
		arr_2[1][1][1] = 123; REQUIRE_FALSE(arr_vw[1][1][1] == 123);
				
		// assignment from view (ndarray gets padded default strides)
		ndarray<3, int> arr_3(previous_shape);
		arr_3.assign(arr_vw_sec, pad);
		REQUIRE(arr_3.view().compare(arr_vw_sec));
		REQUIRE(arr_3.shape() == arr_vw_sec.shape());
		REQUIRE((arr_3.strides() == ndarray_view<3, int>::default_strides(shp, pad)));
		arr_3[1][1][1] = 456; REQUIRE_FALSE(arr_vw_sec[1][1][1] == 456);

		// assignment from null ndarray_view
		ndarray<3, int> arr_4(previous_shape), arr_4_(previous_shape);
		REQUIRE_THROWS((arr_4 = ndarray_view<3, int>::null()));
		REQUIRE_THROWS(arr_4_.assign(ndarray_view<3, int>::null()));

		// copy-assignment from another ndarray (strides get copied)
		ndarray<3, int> arr_5(previous_shape);
		arr_5 = arr_3;
		REQUIRE(arr_5.view().compare(arr_3));
		REQUIRE(arr_5.shape() == arr_3.shape());
		REQUIRE(arr_5.strides() == arr_3.strides());
		arr_5[1][1][1] = 789; REQUIRE_FALSE(arr_3[1][1][1] == 789);
		
		// move-assignment from another ndarray (strides get copied)
		ndarray<3, int> arr_7(previous_shape);
		ndarray<3, int> arr_3_cmp = arr_3;
		arr_7 = std::move(arr_3);
		REQUIRE(arr_7.view().compare(arr_3_cmp));
		REQUIRE(arr_7.shape() == arr_3_cmp.shape());
		REQUIRE(arr_7.strides() == arr_3_cmp.strides());
		verify_ndarray_memory_(arr_7);
		
		// assignment from ndarray_view with different element type
		ndarray<3, float> arr_f(arr_vw_sec);
		ndarray<3, int> arr_9(previous_shape), arr_9_(previous_shape);
		arr_9 = arr_f.cview();
		REQUIRE(arr_9.view().compare(arr_f.cview()));
		REQUIRE(arr_9.shape() == arr_f.shape());
		REQUIRE((arr_9.strides() == ndarray_view<3, int>::default_strides(shp)));
		arr_9_.assign(arr_f.cview(), pad);
		REQUIRE(arr_9_.view().compare(arr_f.cview()));
		REQUIRE(arr_9_.shape() == arr_f.shape());
		REQUIRE((arr_9_.strides() == ndarray_view<3, int>::default_strides(shp, pad)));
		
		// assign using initializer_list (operator= : new ndarray shape can be different)
		ndarray<3, int> arr_init(previous_shape);
		REQUIRE(arr_init.shape() == previous_shape);
		arr_init = {{{0, 1, 2} , {3, 4, 5}}};
		REQUIRE(arr_init.shape() == make_ndsize(1, 2, 3));
		REQUIRE(arr_init[0][0][0] == 0);
		REQUIRE(arr_init[0][1][1] == 4);

		// assign using initializer_list (assign() : assign to view, shape must be same)
		ndarray<2, int> arr_init2(make_ndsize(2, 3));
		REQUIRE(arr_init2.shape() == make_ndsize(2, 3));
		arr_init2.assign({{0, 1, 2} , {3, 4, 5}});
		REQUIRE_THROWS((arr_init2.assign({{0, 1, 2} , {3, 4, 5, 6}})));
		REQUIRE_THROWS((arr_init2.assign({{0, 1, 2} , {3, 4, 5}, {6, 7, 8}})));
		REQUIRE_THROWS((arr_init2.assign({{0, 1} , {3, 4}})));
	}
	
	
	SECTION("wrapper") {
		auto arr = make_ndarray(arr_vw);
		REQUIRE(arr.size() == shape.product());
		REQUIRE(arr.start());
		REQUIRE(arr.shape() == shape);
		REQUIRE((arr.strides() == ndarray_view<3, int>::default_strides(shape)));
		REQUIRE(arr.full_span() == make_ndspan(make_ndptrdiff(0, 0, 0), make_ndptrdiff(3, 4, 4)));
		REQUIRE(arr.compare(arr_vw));
		REQUIRE(arr == arr_vw);
		REQUIRE_FALSE(arr != arr_vw);
		REQUIRE(std::equal(arr.begin(), arr.end(), arr_vw.begin()));
		REQUIRE(std::equal(arr.cbegin(), arr.cend(), arr_vw.begin()));
		REQUIRE(arr(1, 3, -1)(0, 3, -2)(2, 4, -1) == arr_vw(1, 3, -1)(0, 3, -2)(2, 4, -1));
		REQUIRE(arr.at(make_ndptrdiff(1, 2, 2)) == arr_vw[1][2][2]);
		REQUIRE(arr()(0, 3, -2) == arr_vw()(0, 3, -2));
		REQUIRE(arr(1)() == arr_vw(1)());
		REQUIRE(arr[1][0][1] == arr_vw[1][0][1]);
		REQUIRE(arr.section(make_ndptrdiff(1, 0, 2), make_ndptrdiff(3, 3, 4), make_ndptrdiff(-1, -2, -1)) ==
		        arr_vw.section(make_ndptrdiff(1, 0, 2), make_ndptrdiff(3, 3, 4), make_ndptrdiff(-1, -2, -1)));
		REQUIRE(arr.section(make_ndspan(make_ndptrdiff(1, 0, 2), make_ndptrdiff(3, 3, 4)), make_ndptrdiff(-1, -2, -1)) ==
		        arr_vw.section(make_ndspan(make_ndptrdiff(1, 0, 2), make_ndptrdiff(3, 3, 4)), make_ndptrdiff(-1, -2, -1)));
		REQUIRE(arr[2] == arr_vw[2]);
		REQUIRE(arr.slice(1, 1) == arr_vw.slice(1, 1));
		REQUIRE(std::equal(arr.cbegin(), arr.cend(), arr_vw.begin()));
	}
	

	SECTION("make") {
		// make from arr_vw (same as copy-construction)
		auto arr = make_ndarray(arr_vw);
		REQUIRE(arr[1][2][3] == arr_vw[1][2][3]);
		arr[1][2][3] = 7890;
		REQUIRE(arr[1][2][3] == 7890);
		REQUIRE_FALSE(arr_vw[1][2][3] == 7890);
	
		// make from slice of arr_vw
		auto arr2 = make_ndarray(arr_vw[1]);
		REQUIRE(arr2[2][3] == arr_vw[1][2][3]);
		REQUIRE(arr2.shape() == arr_vw[1].shape());
		arr2[2][3] = 1230;
		REQUIRE(arr2[2][3] == 1230);
		REQUIRE_FALSE(arr_vw[1][2][3] == 1230);
	}
	

	SECTION("non-pod") {			
		// construction
		REQUIRE(obj_t::counter == 0);
		ndarray<3, obj_t> arr(shape);
		REQUIRE(obj_t::counter == arr.size());
		int i = 0; for(obj_t& obj : arr) obj.i = i++;
		
		// copy-construction
		ndarray<3, obj_t> arr2 = arr;
		REQUIRE(obj_t::counter == 2*arr.size());
		
		// comparation
		REQUIRE(arr == arr2);
		i = 0; for(const obj_t& obj : arr2) REQUIRE(obj.i == i++);
		
		for(obj_t& obj : arr2) obj.unchanged = 2;
		REQUIRE(arr == arr2);
		
		for(obj_t& obj : arr2) obj.i = 0;
		REQUIRE(arr != arr2);
		
		// assignment
		arr2 = arr;
		REQUIRE(arr == arr2);
		
		// destruction
		{
			ndarray<3, obj_t> arr3(shape);
			REQUIRE(obj_t::counter == 3*arr.size());
		}
		REQUIRE(obj_t::counter == 2*arr.size());
	}
}
