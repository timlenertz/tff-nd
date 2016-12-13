#include <catch.hpp>
#include "../src/opaque/ndarray_opaque.h"
#include "../src/opaque_format/raw.h"
#include "support/ndarray.h"

using namespace tff;
using namespace tff::test;

TEST_CASE("ndarray_opaque", "[nd][ndarray_opaque]") {
	constexpr std::size_t l = sizeof(int);
	auto shape = make_ndsize(3, 4);
	opaque_raw_format frm(l);
	std::size_t pad = l;
	
	using array_type = ndarray_opaque<2, opaque_raw_format>;
	using view_type = ndarray_opaque_view<2, true, opaque_raw_format>;
	
	auto make_opaque_frame = [&frm](int value) {
		auto frame = make_ndarray_opaque_frame(frm);
		int* i = reinterpret_cast<int*>(frame.frame_handle().ptr());
		*i = value;
		return frame;
	};
	
	auto verify_ndarray_memory_ = [&make_opaque_frame](array_type& arr) {
		const auto& shp = arr.shape();
		int i = 0;
		
		for(const auto& coord : make_ndspan(shp))
			arr.at(coord) = make_opaque_frame(i++);
		
		i = 0;
		for(const auto& coord : make_ndspan(shp))
			REQUIRE(arr.at(coord) == make_opaque_frame(i++));
	};
	
	ndarray_opaque<2, opaque_raw_format> og_arr(shape, frm, 0);
	ndarray_opaque_view<2, true, opaque_raw_format> arr_vw = og_arr.view();
	ndarray_opaque_view<2, true, opaque_raw_format> arr_vw_sec = arr_vw()(0, 2);
	REQUIRE_FALSE(arr_vw_sec.has_default_strides());

	
	SECTION("construction") {
		// construction with shape
		array_type arr(shape, frm);
		REQUIRE(arr.shape() == shape);
		REQUIRE((arr.strides() == view_type::default_strides(shape, frm)));
		REQUIRE(arr.allocated_byte_size() >= arr_vw.size()*frm.size());
		verify_ndarray_memory_(arr);

		// construction with shape, padding
		array_type arr_pad(shape, frm, pad);
		REQUIRE(arr_pad.shape() == shape);
		REQUIRE((arr_pad.strides() == view_type::default_strides(shape, frm, pad)));
		verify_ndarray_memory_(arr_pad);

		// construction from view (ndarray gets default strides)
		array_type arr2(arr_vw_sec);
		REQUIRE(arr2.view().compare(arr_vw_sec));
		REQUIRE(arr2.shape() == arr_vw_sec.shape());
		REQUIRE((arr2.strides() == view_type::default_strides(arr_vw_sec.shape(), frm)));
		verify_ndarray_memory_(arr2);
		arr2[1][1] = make_opaque_frame(456);
		REQUIRE(arr2[1][1] == make_opaque_frame(456));
		REQUIRE_FALSE(arr_vw[1][1] == make_opaque_frame(456));

		// construction from view (ndarray gets padded default strides)
		array_type arr3(arr_vw_sec, pad);
		REQUIRE(arr3.view().compare(arr_vw_sec));
		REQUIRE(arr3.shape() == arr_vw_sec.shape());
		REQUIRE((arr3.strides() == view_type::default_strides(arr_vw_sec.shape(), frm, pad)));
		verify_ndarray_memory_(arr3);
		arr3[1][1] = make_opaque_frame(456);
		REQUIRE(arr3[1][1] == make_opaque_frame(456));
		REQUIRE_FALSE(arr_vw[1][1] == make_opaque_frame(456));

		// construction from null ndarray_view
		//REQUIRE_THROWS(new ndarray_opaque<2>(ndarray_view_opaque<2>::null())); Catch bug?
	
		// copy-construction from another ndarray (strides get copied)
		array_type arr4 = arr3;
		REQUIRE(arr4.view().compare(arr3));
		REQUIRE(arr4.shape() == arr3.shape());
		REQUIRE(arr4.strides() == arr3.strides());
		REQUIRE(arr4[1][1] == make_opaque_frame(456));
		verify_ndarray_memory_(arr4);
		arr4[1][1] = make_opaque_frame(789);
		REQUIRE(arr4[1][1] == make_opaque_frame(789));
		REQUIRE_FALSE(arr3[1][1] == make_opaque_frame(789));
	
		// move construction from another ndarray (strides get copied)
		array_type arr5_cmp = arr4;
		array_type arr5 = std::move(arr4);
		REQUIRE(arr5.view().compare(arr5_cmp));
		REQUIRE(arr5.shape() == arr5_cmp.shape());
		REQUIRE(arr5.strides() == arr5_cmp.strides());
		verify_ndarray_memory_(arr5);
	}


	SECTION("assignment") {
		auto previous_shape = make_ndsize(5, 1);
		auto shp = arr_vw_sec.shape();
		
		// assignment from view (ndarray gets default strides)
		array_type arr_(previous_shape, frm), arr_2(previous_shape, frm);
		arr_ = arr_vw_sec;
		REQUIRE(arr_.view().compare(arr_vw_sec));
		REQUIRE(arr_.shape() == arr_vw_sec.shape());
		REQUIRE((arr_.strides() == view_type::default_strides(shp, frm)));
		arr_[1][1] = make_opaque_frame(123);
		REQUIRE_FALSE(arr_vw[1][1] == make_opaque_frame(123));
		arr_2.assign(arr_vw_sec);
		REQUIRE(arr_2.view().compare(arr_vw_sec));
		REQUIRE(arr_2.shape() == arr_vw_sec.shape());
		REQUIRE((arr_2.strides() == view_type::default_strides(shp, frm)));
		arr_2[1][1] = make_opaque_frame(123);
		REQUIRE_FALSE(arr_vw[1][1] == make_opaque_frame(123));
				
		// assignment from view (ndarray gets padded default strides)
		array_type arr_3(previous_shape, frm);
		arr_3.assign(arr_vw_sec, pad);
		REQUIRE(arr_3.view().compare(arr_vw_sec));
		REQUIRE(arr_3.shape() == arr_vw_sec.shape());
		REQUIRE((arr_3.strides() == view_type::default_strides(shp, frm, pad)));
		arr_3[1][1] = make_opaque_frame(456);
		REQUIRE_FALSE(arr_vw_sec[1][1] == make_opaque_frame(456));

		// assignment from null ndarray_view
		array_type arr_4(previous_shape, frm), arr_4_(previous_shape, frm);
		REQUIRE_THROWS(arr_4 = view_type::null());
		REQUIRE_THROWS(arr_4_.assign(view_type::null()));

		// copy-assignment from another ndarray (strides get copied)
		array_type arr_5(previous_shape, frm);
		arr_5 = arr_3;
		REQUIRE(arr_5.view().compare(arr_3));
		REQUIRE(arr_5.shape() == arr_3.shape());
		REQUIRE(arr_5.strides() == arr_3.strides());
		arr_5[1][1] = make_opaque_frame(789);
		REQUIRE_FALSE(arr_3[1][1] == make_opaque_frame(789));
		
		// move-assignment from another ndarray (strides get copied)
		array_type arr_7(previous_shape, frm);
		array_type arr_3_cmp = arr_3;
		arr_7 = std::move(arr_3);
		REQUIRE(arr_7.view().compare(arr_3_cmp));
		REQUIRE(arr_7.shape() == arr_3_cmp.shape());
		REQUIRE(arr_7.strides() == arr_3_cmp.strides());
		verify_ndarray_memory_(arr_7);
	}
	
	
	SECTION("non-pod frames") {
		nonpod_frame_format frm(100);
		REQUIRE(nonpod_frame_handle::counter == 0);
		
		{
			ndarray_opaque<2, nonpod_frame_format> arr(make_ndsize(2, 2), frm);
			REQUIRE(nonpod_frame_handle::counter == 4);
			
			ndarray_opaque<2, nonpod_frame_format> arr2 = arr;
			REQUIRE(nonpod_frame_handle::counter == 8);
			
			ndarray_opaque<2, nonpod_frame_format> arr3 = std::move(arr2);
			REQUIRE(nonpod_frame_handle::counter == 8);
		}
		
		REQUIRE(nonpod_frame_handle::counter == 0);
	}
}
