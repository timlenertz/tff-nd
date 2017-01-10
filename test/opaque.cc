#include <catch.hpp>
#include "../src/opaque/ndarray_opaque_view.h"
#include "../src/opaque/ndarray_opaque_view_cast.h"
#include "../src/opaque_format/ndarray.h"
#include "../src/opaque_format/raw.h"

using namespace tff;

constexpr std::size_t sz = 243;
constexpr std::size_t al = 32;


TEST_CASE("opaque_raw_format, alignment padding", "[nd][opaque_raw_format]") {		
	SECTION("format") {
		opaque_raw_format frm(sz, al);
		
		REQUIRE(frm.size() > sz);
		REQUIRE(is_multiple_of(frm.size(), al));
		REQUIRE(frm.alignment_requirement() == al);
		
		REQUIRE(frm.is_pod());
		REQUIRE(frm.pod_format().size() == frm.size());
		REQUIRE(frm.pod_format().length() == 1);
		REQUIRE(frm.pod_format().stride() == frm.size());
		REQUIRE(frm.pod_format().elem_size() == sz);
		REQUIRE(frm.pod_format().elem_alignment() == al);
		REQUIRE(frm.pod_format().elem_padding() > 0);
		REQUIRE_FALSE(frm.pod_format().is_contiguous());
		
		REQUIRE_FALSE(frm.is_ndarray());
	}
	
	SECTION("handle") {
		struct storage_t {
			alignas(al) std::array<byte, sz> data;
			bool operator==(const storage_t& other) const { return (data == other.data); }
			bool operator!=(const storage_t& other) const { return (data != other.data); }
		};
		
		opaque_raw_format frm(sz, al);
		storage_t raw; storage_t raw2;
		raw.data.fill(1); raw2.data.fill(2);
		REQUIRE(is_aligned(&raw, al));
		REQUIRE(sizeof(raw) >= sz);
			
		opaque_raw_frame_handle<true> vw(&raw, frm);
		opaque_raw_frame_handle<false> vw2(&raw2, frm);

		REQUIRE_FALSE(vw.compare(vw2));
		REQUIRE_FALSE(vw2.compare(vw));
		REQUIRE(vw.compare(vw));
		REQUIRE(vw2.compare(vw2));
		
		vw.assign(vw2);

		REQUIRE(vw.compare(vw2));
		REQUIRE(vw2.compare(vw));

		vw.assign(vw);
	}
}

TEST_CASE("opaque_raw_format", "[nd][opaque_raw_format]") {	
	constexpr std::size_t sz = 100;
	
	SECTION("format") {
		opaque_raw_format frm(sz);

		REQUIRE(frm.size() == sz);
		REQUIRE(frm.alignment_requirement() == 1);
		
		REQUIRE(frm.is_pod());
		REQUIRE(frm.pod_format().size() == sz);
		REQUIRE(frm.pod_format().length() == 1);
		REQUIRE(frm.pod_format().stride() == sz);
		REQUIRE(frm.pod_format().elem_size() == sz);
		REQUIRE(frm.pod_format().elem_alignment() == 1);
		REQUIRE(frm.pod_format().elem_padding() == 0);
		REQUIRE(frm.pod_format().is_contiguous());
	
		REQUIRE_FALSE(frm.is_ndarray());
	}
	
	SECTION("handle") {
		opaque_raw_format frm(sz);
		std::vector<byte> raw(sz, 1), raw2(sz, 2);
		
		opaque_raw_frame_handle<true> vw(raw.data(), frm);
		opaque_raw_frame_handle<false> vw2(raw2.data(), frm);

		REQUIRE_FALSE(vw.compare(vw2));
		REQUIRE_FALSE(vw2.compare(vw));
		REQUIRE(vw.compare(vw));
		REQUIRE(vw2.compare(vw2));
		
		vw.assign(vw2);

		REQUIRE(vw.compare(vw2));
		REQUIRE(vw2.compare(vw));

		vw.assign(vw);
	}

	SECTION("cast, pod") {
		std::size_t sz = 3*2*sizeof(int);
		auto shp = make_ndsize(2, 3, 2);
		std::size_t len = shp.product();
		std::vector<int> raw(len, 1), raw2(len, 2);
		ndarray_view<3, int> vw(raw.data(), shp);
		ndarray_view<3, const int> vw2(raw2.data(), shp);
		
		opaque_raw_format op_frm(sz, alignof(int));
		
		auto op_vw = to_opaque<1>(vw, op_frm);
		auto op_vw2 = to_opaque<1>(vw2, op_frm);
				
		REQUIRE(op_vw.frame_format() == op_frm);

		REQUIRE(op_vw == op_vw);
		REQUIRE_FALSE(op_vw == op_vw2);
		REQUIRE_FALSE(op_vw2 == op_vw);
		
		op_vw = op_vw2;
		
		REQUIRE(raw == raw2);
		
		REQUIRE(op_vw == op_vw2);
		REQUIRE(op_vw2 == op_vw);
	}
}

TEST_CASE("opaque_ndarray_format, pod", "[nd][opaque_ndarray_format]") {
	auto shp = make_ndsize(3, 4);
	std::size_t len = 3*4;
	opaque_ndarray_format frm(sizeof(int), alignof(int), 2*sizeof(int), true, shp);
	
	SECTION("format") {		
		REQUIRE(frm.size() == len * sizeof(int) * 2);
		REQUIRE(frm.alignment_requirement() == alignof(int));
		
		REQUIRE(frm.is_pod());
		REQUIRE(frm.pod_format().size() == frm.size());
		REQUIRE(frm.pod_format().length() == len);
		REQUIRE(frm.pod_format().stride() == 2*sizeof(int));
		REQUIRE(frm.pod_format().elem_size() == sizeof(int));
		REQUIRE(frm.pod_format().elem_alignment() == alignof(int));
		REQUIRE(frm.pod_format().elem_padding() == 1*sizeof(int));
		REQUIRE_FALSE(frm.pod_format().is_contiguous());
		
		REQUIRE(frm.is_ndarray());
		REQUIRE(frm.dimension() == 2);
		REQUIRE(frm.shape() == make_ndsize(3, 4));
		REQUIRE(frm.elem_size() == sizeof(int));
		REQUIRE(frm.elem_alignment_requirement() == alignof(int));
		REQUIRE(frm.elem_stride() == 2*sizeof(int));
	}

	SECTION("handle") {
		std::vector<int> raw(len*2, 1);
		std::vector<int> raw2(len*2, 2);
		
		for(std::ptrdiff_t i = 1; i < 2*len; i += 2) {
			// padding between elements
			// (ignored for compare, not touched by assign)
			raw[i] = 123; raw2[i] = 456;
		}

		opaque_ndarray_frame_handle<true> vw(raw.data(), frm);
		opaque_ndarray_frame_handle<false> vw2(raw2.data(), frm);

		REQUIRE_FALSE(vw.compare(vw2));
		REQUIRE_FALSE(vw2.compare(vw));
		REQUIRE(vw.compare(vw));
		REQUIRE(vw2.compare(vw2));
		
		vw.assign(vw2);
		
		REQUIRE(vw.compare(vw2));
		REQUIRE(vw2.compare(vw));

		for(std::ptrdiff_t i = 1; i < 2*len; i += 2) {
			REQUIRE(raw[i] == 123);
			REQUIRE(raw2[i] == 456);
		}
	}
	
	SECTION("cast, pod") {
		auto concrete_shp = make_ndsize(2, 3, 3, 4);
		std::size_t concrete_len = concrete_shp.product();		
		auto concrete_str = ndarray_view<4, int>::default_strides(concrete_shp, sizeof(int));
		
		std::vector<int> raw(2*concrete_len, 1), raw2(2*concrete_len, 2);
		for(std::ptrdiff_t i = 1; i < 2*concrete_len; i += 2) { raw[i] = 123; raw2[i] = 456; }
		
		ndarray_view<4, int> vw(raw.data(), concrete_shp, concrete_str);
		ndarray_view<4, const int> vw2(raw2.data(), concrete_shp, concrete_str);
		
		auto op_vw = to_opaque<2>(vw);
		auto op_vw2 = to_opaque<2>(vw2);
		
		REQUIRE(op_vw.frame_format().size() == 3*4*sizeof(int)*2);
		REQUIRE(op_vw.frame_format().alignment_requirement() == alignof(int));
		REQUIRE(op_vw.frame_format().is_pod());
		REQUIRE(op_vw.frame_format().pod_format().size() == op_vw.frame_format().size());
		REQUIRE(op_vw.frame_format().pod_format().length() == 3*4);
		REQUIRE(op_vw.frame_format().pod_format().stride() == sizeof(int)*2);
		REQUIRE(op_vw.frame_format().pod_format().elem_size() == sizeof(int));
		REQUIRE(op_vw.frame_format().pod_format().elem_alignment() == alignof(int));
		REQUIRE(op_vw.frame_format().pod_format().elem_padding() == sizeof(int));
		REQUIRE_FALSE(op_vw.frame_format().pod_format().is_contiguous());
		REQUIRE(op_vw.frame_format().is_ndarray());
		REQUIRE(op_vw.frame_format().dimension() == 2);
		REQUIRE(op_vw.frame_format().shape() == make_ndsize(3, 4));
		REQUIRE(op_vw.frame_format().elem_size() == sizeof(int));
		REQUIRE(op_vw.frame_format().elem_alignment_requirement() == alignof(int));
		REQUIRE(op_vw.frame_format().elem_stride() == sizeof(int)*2);
		
		REQUIRE(op_vw == op_vw);
		REQUIRE_FALSE(op_vw == op_vw2);
		REQUIRE_FALSE(op_vw2 == op_vw);
		
		op_vw = op_vw2;
		for(std::ptrdiff_t i = 1; i < 2*concrete_len; i += 2) REQUIRE(raw[i] == 123);
		
		REQUIRE(op_vw == op_vw2);
		REQUIRE(op_vw2 == op_vw);
		
		ndarray_view<4, int> re_vw = from_opaque<4, int>(op_vw);
		ndarray_view<4, const int> re_vw2 = from_opaque<4, const int>(op_vw2);
		REQUIRE(same(vw, re_vw));
		REQUIRE(same(vw2, re_vw2));
	}
}

TEST_CASE("opaque_ndarray_format, non-pod", "[nd][opaque_ndarray_format]") {
	auto shp = make_ndsize(3, 4);
	std::size_t len = 3*4;
	opaque_ndarray_format frm(sizeof(int), alignof(int), 2*sizeof(int), false, shp);
	
	SECTION("format") {		
		REQUIRE(frm.size() == len * sizeof(int) * 2);
		REQUIRE(frm.alignment_requirement() == alignof(int));
		
		REQUIRE_FALSE(frm.is_pod());
		
		REQUIRE(frm.is_ndarray());
		REQUIRE(frm.dimension() == 2);
		REQUIRE(frm.shape() == make_ndsize(3, 4));
		REQUIRE(frm.elem_size() == sizeof(int));
		REQUIRE(frm.elem_alignment_requirement() == alignof(int));
		REQUIRE(frm.elem_stride() == 2*sizeof(int));
	}

	SECTION("handle") {
		std::vector<int> raw(len*2, 1);
		std::vector<int> raw2(len*2, 2);

		opaque_ndarray_frame_handle<true> vw(raw.data(), frm);
		opaque_ndarray_frame_handle<false> vw2(raw2.data(), frm);

		REQUIRE_THROWS(vw.compare(vw2));
		REQUIRE_THROWS(vw2.compare(vw));
		REQUIRE_THROWS(vw.compare(vw));
		REQUIRE_THROWS(vw2.compare(vw2));
		
		REQUIRE_THROWS(vw.assign(vw2));
	}
	
	SECTION("cast") {
		struct obj_t {
			int i;
			obj_t() : i(4) { }
		};
		REQUIRE(sizeof(obj_t) == sizeof(int));
		REQUIRE(alignof(obj_t) == alignof(int));
		REQUIRE_FALSE(std::is_pod<obj_t>::value);
		
		auto concrete_shp = make_ndsize(2, 3, 3, 4);
		std::size_t concrete_len = concrete_shp.product();		
		auto concrete_str = ndarray_view<4, obj_t>::default_strides(concrete_shp, sizeof(int));
		
		std::vector<int> raw(2*concrete_len, 1), raw2(2*concrete_len, 2);
		for(std::ptrdiff_t i = 1; i < 2*concrete_len; i += 2) { raw[i] = 123; raw2[i] = 456; }
		
		ndarray_view<4, obj_t> vw(reinterpret_cast<obj_t*>(raw.data()), concrete_shp, concrete_str);
		ndarray_view<4, const obj_t> vw2(reinterpret_cast<const obj_t*>(raw2.data()), concrete_shp, concrete_str);
		
		auto op_vw = to_opaque<2>(vw);
		auto op_vw2 = to_opaque<2>(vw2);
		
		REQUIRE(op_vw.frame_format().size() == 3*4*sizeof(int)*2);
		REQUIRE(op_vw.frame_format().alignment_requirement() == alignof(int));
		REQUIRE_FALSE(op_vw.frame_format().is_pod());
		REQUIRE(op_vw.frame_format().is_ndarray());
		REQUIRE(op_vw.frame_format().dimension() == 2);
		REQUIRE(op_vw.frame_format().shape() == make_ndsize(3, 4));
		REQUIRE(op_vw.frame_format().elem_size() == sizeof(int));
		REQUIRE(op_vw.frame_format().elem_alignment_requirement() == alignof(int));
		REQUIRE(op_vw.frame_format().elem_stride() == sizeof(int)*2);
		
		REQUIRE_THROWS(op_vw == op_vw);
		REQUIRE_THROWS(op_vw == op_vw2);
		REQUIRE_THROWS(op_vw2 == op_vw);
		REQUIRE_THROWS(op_vw = op_vw2);
		
		ndarray_view<4, obj_t> re_vw = from_opaque<4, obj_t>(op_vw);
		ndarray_view<4, const obj_t> re_vw2 = from_opaque<4, const obj_t>(op_vw2);
		REQUIRE(same(vw, re_vw));
		REQUIRE(same(vw2, re_vw2));
	}

}


TEST_CASE("ndarray_opaque_view", "[nd][ndarray_opaque_view]") {	
	SECTION("basics") {
		using opaque_view_type = ndarray_opaque_view<2, true, opaque_raw_format>;
		opaque_raw_format frm(100, 1), frm2(100, 2);
		std::vector<byte> raw(1000);
		
		// default strides
		REQUIRE(( opaque_view_type::default_strides(make_ndsize(3, 2), frm) == make_ndptrdiff(2*100, 100) ));
		REQUIRE(( opaque_view_type::default_strides(make_ndsize(3, 2), frm, 1) == make_ndptrdiff(2*101, 101) ));

		// default strides view
		auto shp = make_ndsize(2, 3);
		opaque_view_type a1(raw.data(), shp, frm);
		REQUIRE(a1.start() == raw.data());
		REQUIRE(a1.shape() == shp);
		REQUIRE(a1.strides() == make_ndptrdiff(300, 100));
		REQUIRE(a1.size() == shp.product());
		REQUIRE_FALSE(a1.is_null());
		REQUIRE(a1.full_span() == make_ndspan(make_ndptrdiff(2, 3)));
		REQUIRE(a1.frame_format() == frm);
		REQUIRE(a1.has_default_strides());
		REQUIRE(a1.default_strides_padding() == 0);
		REQUIRE(a1.has_default_strides_without_padding());
		
		// padded default strides view
		opaque_view_type a1pad(raw.data(), shp, opaque_view_type::default_strides(shp, frm, 10), frm);
		REQUIRE(a1pad.start() == raw.data());
		REQUIRE(a1pad.shape() == shp);
		REQUIRE(a1pad.strides() == make_ndptrdiff(330, 110));
		REQUIRE(a1pad.size() == shp.product());
		REQUIRE(a1pad.has_default_strides());
		REQUIRE(a1pad.default_strides_padding() == 10);
		REQUIRE_FALSE(a1pad.has_default_strides_without_padding());
		
		// non-default strides
		auto str = make_ndptrdiff(320, 100);
		opaque_view_type a2(raw.data(), shp, str, frm);
		REQUIRE(a2.start() == raw.data());
		REQUIRE(a2.shape() == shp);
		REQUIRE(a2.strides() == str);
		REQUIRE(a2.size() == shp.product());
		REQUIRE_FALSE(a2.has_default_strides());
		REQUIRE_THROWS(a2.default_strides_padding());
		REQUIRE_FALSE(a2.has_default_strides_without_padding());
		
		// comparison and assignment (shallow)
		opaque_view_type a3(raw.data() + 13, shp, frm);
		REQUIRE(same(a1, a1));
		REQUIRE_FALSE(same(a1, a3));
		REQUIRE_FALSE(same(a3, a1));
		opaque_view_type a3_;
		REQUIRE(a3_.is_null());
		REQUIRE_FALSE(same(a3_, a1));
		a3_.reset(raw.data() + 13, shp, frm);
		REQUIRE(same(a3_, a3));
		a3_.reset(a1);
		REQUIRE(same(a3_, a1));
		REQUIRE(a3_.start() == raw.data());
		REQUIRE(a3_.shape() == shp);
		REQUIRE(a3_.strides() == a1.strides());
		REQUIRE(a3_.frame_format() == frm);
		opaque_view_type a1_f(raw.data(), shp, frm2);
		REQUIRE(a1_f.start() == a1.start());
		REQUIRE(a1_f.shape() == a1.shape());
		REQUIRE(a1_f.strides() == a1.strides());
		REQUIRE_FALSE(same(a1_f, a1));
		
		// frame view/handle
		ndarray_opaque_view<0, true, opaque_raw_format> fvw = a1[0][0];
		opaque_raw_frame_handle<true> fhd = fvw.frame_handle();
		REQUIRE(fvw.start() == fhd.ptr());
		
		// copy-construction
		opaque_view_type a1copy = a1;
		REQUIRE(same(a1copy, a1));
		
		// const and non-const
		ndarray_opaque_view<2, false, opaque_raw_format> a1c = a1;
		REQUIRE(same(a1, a1c));
		REQUIRE(same(a1c, a1));
		a1c.reset(a3);
		REQUIRE(same(a3, a1c));
		REQUIRE(same(a1c, a3));
	}

	SECTION("null view; zero-size view") {
		std::vector<byte> raw(2);
		using opaque_view_type = ndarray_opaque_view<1, true, opaque_raw_format>;
		opaque_raw_format frm(100);

		opaque_view_type zero_size_vw(raw.data(), make_ndsize(0), frm);
		opaque_view_type null_vw;
		
		SECTION("null view") {
			REQUIRE(null_vw.is_null());
			REQUIRE(! null_vw);
			REQUIRE(null_vw.start() == nullptr);
			REQUIRE(null_vw.shape() == make_ndsize(0));
			REQUIRE(null_vw.size() == 0);
			REQUIRE(null_vw.full_span() == make_ndspan(make_ndptrdiff(0)));
			REQUIRE(null_vw.frame_format() == opaque_raw_format());
			REQUIRE(null_vw.begin() == null_vw.end());
			REQUIRE(same(null_vw, null_vw));
			REQUIRE_FALSE(same(null_vw, zero_size_vw));
		}
		
		SECTION("zero-size view") {
			REQUIRE_FALSE(zero_size_vw.is_null());
			REQUIRE(zero_size_vw);
			REQUIRE(zero_size_vw.start() == raw.data());
			REQUIRE(zero_size_vw.shape() == make_ndsize(0));
			REQUIRE(zero_size_vw.size() == 0);
			REQUIRE_FALSE(zero_size_vw.frame_format() == opaque_raw_format());
			REQUIRE(zero_size_vw.begin() == zero_size_vw.end());
			REQUIRE(same(zero_size_vw, zero_size_vw));
			
			opaque_view_type zero_size_vw2(raw.data()+1, make_ndsize(0), frm);
			REQUIRE_FALSE(same(zero_size_vw2, zero_size_vw));
		}
	}

	SECTION("3dim") {
		// 01 02 03
		// 04 05 06
		// 
		//   07 08 09
		//   10 11 12
		// 
		//     13 14 15
		//     16 17 18
		// 
		//       19 20 21
		//       22 23 24
		
		std::vector<std::uint32_t> raw(24*2);
		for(std::uint32_t i = 0; i < raw.size(); i += 2) raw[i] = 1 + i/2;
		
		auto shp = make_ndsize(4, 2, 3);
		opaque_raw_format frm(4);
		using opaque_view_type = ndarray_opaque_view<3, true, opaque_raw_format>;
		opaque_view_type vw(raw.data(), shp, opaque_view_type::default_strides(shp, frm, 4), frm);
		
		auto frame_val = [](const opaque_raw_frame_handle<true>& fr) -> std::uint32_t& {
			return *static_cast<std::uint32_t*>(fr.ptr());
		};
		
		auto compare_seq = [&frame_val](const auto& vw, const std::vector<std::uint32_t>& values) {
			int i = 0;
			for(auto&& fr : vw) {
				REQUIRE(i < values.size());
				REQUIRE(frame_val(fr.frame_handle()) == values[i++]);
			}
			REQUIRE(i == values.size());
		};
		
		SECTION("iterator") {
			compare_seq(vw, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
			compare_seq(vw[2], { 13, 14, 15, 16, 17, 18 });
			compare_seq(vw[2][0], { 13, 14, 15 });

			auto it = vw.begin();
			REQUIRE(same(it.view(), vw));
			// ++, --, +=, -=
			REQUIRE(it.coordinates() == make_ndptrdiff(0, 0, 0));
			REQUIRE(it->frame_handle().ptr() == raw.data());
			REQUIRE((*it).frame_handle().ptr() == raw.data());
			REQUIRE(it->frame_handle().ptr() == raw.data());
			REQUIRE((++it).coordinates() == make_ndptrdiff(0, 0, 1));
			REQUIRE((++it).coordinates() == make_ndptrdiff(0, 0, 2));
			REQUIRE((--it).coordinates() == make_ndptrdiff(0, 0, 1));
			REQUIRE((it++).coordinates() == make_ndptrdiff(0, 0, 1));
			REQUIRE(it.coordinates() == make_ndptrdiff(0, 0, 2));
			REQUIRE((it += 4).coordinates() == make_ndptrdiff(1, 0, 0));
			REQUIRE((++it).coordinates() == make_ndptrdiff(1, 0, 1));
			REQUIRE((it -= 3).coordinates() == make_ndptrdiff(0, 1, 1));
			REQUIRE((--it).coordinates() == make_ndptrdiff(0, 1, 0));
			REQUIRE((it--).coordinates() == make_ndptrdiff(0, 1, 0));
			REQUIRE(it.coordinates() == make_ndptrdiff(0, 0, 2));

			// ==, !=, >, >=, <, <=
			auto it2 = vw.begin();
			REQUIRE(it2 == vw.begin());
			REQUIRE(it != vw.begin());
			
			REQUIRE_FALSE(it == it2);
			REQUIRE_FALSE(it2 == it);
			REQUIRE(it != it2);
			REQUIRE(it2 != it);
			
			REQUIRE(it2 < it);
			REQUIRE(it2 <= it);
			REQUIRE_FALSE(it2 > it);
			REQUIRE_FALSE(it2 >= it);
			REQUIRE(it > it2);
			REQUIRE(it >= it2);
			REQUIRE_FALSE(it < it2);
			REQUIRE_FALSE(it <= it2);

			it2++;
			it--; // now it == it2
			REQUIRE(it == it2);
			REQUIRE(it2 == it);
			REQUIRE_FALSE(it != it2);
			REQUIRE_FALSE(it2 != it);
			REQUIRE_FALSE(it < it2);
			REQUIRE(it <= it2);
			REQUIRE_FALSE(it > it2);
			REQUIRE(it >= it2);
			REQUIRE_FALSE(it2 > it);
			REQUIRE(it2 >= it);
			REQUIRE_FALSE(it2 < it);
			REQUIRE(it2 <= it);
		
			// [], +, -
			it2 += 4;
			it+1;
			REQUIRE(it[4].frame_handle().ptr() == it2->frame_handle().ptr());
			REQUIRE((it + 4)->frame_handle().ptr() == it2->frame_handle().ptr());
			REQUIRE((4 + it)->frame_handle().ptr() == it2->frame_handle().ptr());
			REQUIRE((it2 - 4)->frame_handle().ptr() == it->frame_handle().ptr());
			REQUIRE(it2 - it == 4);
			REQUIRE(it - it2 == -4);
				
			auto it_end = vw.end();
			REQUIRE(same(it_end.view(), vw));
			std::ptrdiff_t dist = it_end - it;
			it += (it_end - it);
			REQUIRE(it == it_end);
		}
		
		SECTION("subscript") {
			REQUIRE(frame_val(vw[0][0][0]) == 1);
			REQUIRE(frame_val(vw.at({ 0, 0, 0 })) == 1);
			REQUIRE(frame_val(vw[1][1][1]) == 11);
			REQUIRE(frame_val(vw.at({ 1, 1, 1 })) == 11);
			REQUIRE(frame_val(vw[2][0][1]) == 14);
			REQUIRE(frame_val(vw.at({ 2, 0, 1 })) == 14);
			
			REQUIRE(frame_val(vw[-1][-1][-1]) == 24);
			REQUIRE(frame_val(vw.at({ -1, -1, -1 })) == 24);
			REQUIRE(frame_val(vw[-2][0][1]) == 14);
			REQUIRE(frame_val(vw.at({ -2, 0, 1 })) == 14);
			
			frame_val(vw[0][0][0]) = 123;
			REQUIRE(frame_val(vw[0][0][0]) == 123);
			frame_val(vw.at({ -2, 0, 1 })) = 456;
			REQUIRE(frame_val(vw[-2][0][1]) == 456);
		}
		
		SECTION("section, slice") {
			compare_seq(vw()()(), { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 });
			compare_seq(vw()(1)(), { 4, 5, 6, 10, 11, 12, 16, 17, 18, 22, 23, 24 });
			compare_seq(vw()(1)(0, 2), { 4, 5, 10, 11, 16, 17, 22, 23 });
			compare_seq(vw(0, 4)(0)(0), { 1, 7, 13, 19 });
			compare_seq(vw(0, -1)(0)(0), { 1, 7, 13 });
			compare_seq(vw(0, 4, -1)(0)(0), { 19, 13, 7, 1 });
			compare_seq(vw(1, 4, -2)(0)(0), { 19, 7 });

			compare_seq(vw.section({0, 1, 0}, {-1, 2, 3}, {2, 1, -1}), { 6, 5, 4, 18, 17, 16 });
			compare_seq(vw.section(make_ndspan(make_ndptrdiff(0, 1, 0), make_ndptrdiff(4, 2, 3)), {2, 1, -1}), { 6, 5, 4, 18, 17, 16 });
			
			compare_seq(vw.slice(1, 0), { 7, 8, 9, 10, 11, 12 });
		}
	}
	
	
	SECTION("deep assign and compare") {
		// creating two views to different data
		auto shp = make_ndsize(2, 3);
		std::vector<int> raw1(2 * 3 * 10);
		std::vector<int> raw2 = raw1;
		for(std::ptrdiff_t i = 0; i != raw1.size(); ++i) {
			raw1[i] = i;
			raw2[i] = 2*i + 1;
		}
		opaque_ndarray_format frm(sizeof(int), alignof(int), sizeof(int), true, make_ndsize(10));
		REQUIRE(frm.is_pod());
		REQUIRE(frm.pod_format().is_contiguous());
		auto str = ndarray_opaque_view<2, true, opaque_ndarray_format>::default_strides(shp, frm);
		ndarray_opaque_view<2, true, opaque_ndarray_format> vw1(raw1.data(), shp, str, frm);
		ndarray_opaque_view<2, false, opaque_ndarray_format> vw2(raw2.data(), shp, str, frm);
		
		REQUIRE(vw1 == vw1);
		REQUIRE(vw1 != vw2);

		// assign and compare 0-d section (= frame view)
		auto fr1 = vw1[1][1];
		auto fr2 = vw2[1][1];
		vw1[1][1] = vw2[1][1];
		REQUIRE(vw1[1][0] != vw2[1][0]);
		REQUIRE(vw1[1][1] == vw2[1][1]);
		REQUIRE(vw1[1][2] != vw2[1][2]);
		
		REQUIRE_FALSE(raw1[(1*3 + 1)*10 - 1] == raw2[(1*3 + 1)*10 - 1]);
		for(std::ptrdiff_t i = 0; i < 10; ++i)
			REQUIRE(raw1[(1*3 + 1)*10 + i] == raw2[(1*3 + 1)*10 + i]);
		REQUIRE_FALSE(raw1[(1*3 + 1)*10 + 10] == raw2[(1*3 + 1)*10 + 10]);

		// assign and compare 1-d section
		vw1[0] = vw2[0];
		REQUIRE(vw1[0] == vw2[0]);
		REQUIRE(vw1[1] != vw2[1]);

		// assign and compare full
		vw1 = vw2;
		REQUIRE(vw1 == vw2);	
	}
	
	
	SECTION("pod format") {
		SECTION("contiguous frames") {
			// frame = contigous, size 3*4*sizeof(int)
			// opaque dim 1 : padding 4 --> still default strides because frame has no internal padding
			// opaque dim 0 : padding 32 --> no longer default strides
			
			auto frame_shp = make_ndsize(3, 4);
			opaque_ndarray_format frm(sizeof(int), alignof(int), sizeof(int), true, frame_shp);
			auto op_shp = make_ndsize(2, 2);
			auto shp = ndcoord_cat(op_shp, frame_shp);
			REQUIRE(frm.is_pod());
			REQUIRE(frm.pod_format().is_contiguous());
		
			auto str = make_ndptrdiff(2*(frm.size() + 4) + 32, frm.size() + 4);
			std::vector<byte> raw(2*str.front());
			ndarray_opaque_view<2, true, opaque_ndarray_format> vw(raw.data(), op_shp, str, frm);
			
			REQUIRE_FALSE(vw.has_default_strides());
			REQUIRE(vw.has_default_strides(1));
			REQUIRE(vw.default_strides_padding(1) == 4);

			SECTION("0 dim (frame)") {
				REQUIRE(tail_has_pod_format<0>(vw));
				pod_array_format tfrm = tail_pod_format<0>(vw);
				REQUIRE(same_coverage(tfrm, frm.pod_format()));
				REQUIRE(tfrm.elem_alignment() == alignof(int));
			}
			
			SECTION("1 dim") {
				REQUIRE(tail_has_pod_format<1>(vw));
				pod_array_format tfrm = tail_pod_format<1>(vw);
				REQUIRE(tfrm.elem_size() == frm.size());
				REQUIRE(tfrm.stride() == frm.size() + 4);
				REQUIRE(tfrm.length() == 2);
				REQUIRE(tfrm.elem_alignment() == alignof(int));
			}
			
			SECTION("2 dim") {
				REQUIRE_FALSE(tail_has_pod_format<2>(vw));
				REQUIRE_THROWS(tail_pod_format<2>(vw));
				REQUIRE_FALSE(has_pod_format(vw));
				REQUIRE_THROWS(pod_format(vw));
			}
		}
		
		SECTION("non-contiguous frames") {
			// frame = non-contigous, length 3*4, padding sizeof(int)
			// opaque dim 1 : no padding --> default strides without padding (but frame has internal padding)
			// opaque dim 0 : padding 32 --> not default strides

			auto frame_shp = make_ndsize(3, 4);
			opaque_ndarray_format frm(sizeof(int), alignof(int), 2*sizeof(int), true, frame_shp);
			auto op_shp = make_ndsize(2, 2);
			auto shp = ndcoord_cat(op_shp, frame_shp);
			REQUIRE(frm.is_pod());
			REQUIRE_FALSE(frm.pod_format().is_contiguous());
			REQUIRE(frm.pod_format().elem_padding() == sizeof(int));
			
			auto str = make_ndptrdiff(2*frm.size() + 32, frm.size());
			std::vector<byte> raw(2*str.front());
			ndarray_opaque_view<2, true, opaque_ndarray_format> vw(raw.data(), op_shp, str, frm);

			REQUIRE_FALSE(vw.has_default_strides());
			REQUIRE(vw.has_default_strides(1));
			REQUIRE(vw.default_strides_padding(1) == 0);

			SECTION("0 dim (frame)") {
				REQUIRE(tail_has_pod_format<0>(vw));
				pod_array_format tfrm = tail_pod_format<0>(vw);
				REQUIRE(same_coverage(tfrm, frm.pod_format()));
				REQUIRE(tfrm.elem_alignment() == alignof(int));
				
				REQUIRE(tfrm.elem_size() == sizeof(int));
				REQUIRE(tfrm.stride() == 2*sizeof(int));
				REQUIRE(tfrm.length() == 3*4);
			}
			
			SECTION("1 dim") {
				REQUIRE(tail_has_pod_format<1>(vw));
				pod_array_format tfrm = tail_pod_format<1>(vw);
				REQUIRE(tfrm.elem_size() == sizeof(int));
				REQUIRE(tfrm.stride() == 2*sizeof(int));
				REQUIRE(tfrm.length() == 2 * 3*4);
				REQUIRE(tfrm.elem_alignment() == alignof(int));
			}
			
			SECTION("2 dim") {
				REQUIRE_FALSE(tail_has_pod_format<2>(vw));
				REQUIRE_THROWS(tail_pod_format<2>(vw));
				REQUIRE_FALSE(has_pod_format(vw));
				REQUIRE_THROWS(pod_format(vw));
			}
		}
	}
}
