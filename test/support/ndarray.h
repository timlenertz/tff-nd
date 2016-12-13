#ifndef TFF_TESTSUPPORT_NDARRAY_H_
#define TFF_TESTSUPPORT_NDARRAY_H_

#include <iostream>
#include <sstream>
#include <vector>
#include "../../src/ndarray.h"
#include "../../src/opaque_format/raw.h"

namespace tff { namespace test {

struct obj_t {
	int i = 0;
	int unchanged = 0;

	static int counter;
	obj_t() { ++counter; }
	obj_t(const obj_t&) { ++counter; }
	~obj_t() { --counter; }
	obj_t& operator=(const obj_t& obj) { i = obj.i; return *this; }
	bool operator==(const obj_t& obj) const { return (i == obj.i); }
};


struct nonpod_frame_handle;

struct nonpod_frame_format : opaque_raw_format {
	using frame_handle_type = nonpod_frame_handle;
	using const_frame_handle_type = nonpod_frame_handle;
	
	using opaque_raw_format::opaque_raw_format;
	
	bool is_pod() const { return false; }
	pod_array_format pod_format() const { throw 0; }
};

struct nonpod_frame_handle {
	static int counter;
	
	nonpod_frame_handle(const void*, const nonpod_frame_format&) { }
	void assign(const nonpod_frame_handle&) { }
	bool compare(const nonpod_frame_handle&) { return true; }
	void construct() const { counter++; }
	void destruct() const { counter--; }
	void initialize() const {  }
};



template<typename Buffer>
bool compare_sequence_forwards_(const Buffer& buf, const std::vector<int>& seq) {
	std::ostringstream str;

	auto vec_it = seq.begin();
	auto it = buf.begin();
	for(; (it != buf.end()) && (vec_it != seq.end()); ++it, ++vec_it) {
		str << std::hex << "got:" << *it << "  wanted:" << *vec_it << std::endl;
		if(*vec_it != *it) break;
	}
	if( (it == buf.end()) && (vec_it == seq.end()) ) return true;
	else if( (it == buf.end()) && (vec_it != seq.end()) ) str << "ended early" << std::endl;
	else if( (it != buf.end()) && (vec_it == seq.end()) ) str << "ended late" << std::endl;
	
	std::cerr << "sequence mismatch: " << std::endl << str.str() << std::endl;	
	
	return false;
}


template<typename Buffer>
bool compare_sequence_(const Buffer& buf, const std::vector<int>& seq) {
	if(! compare_sequence_forwards_(buf, seq)) return false;
	
	std::ostringstream str;

	auto vec_it = seq.end();
	auto it = buf.end();
	
	do {
		--it; --vec_it;
		str << std::hex << "got:" << *it << "  wanted:" << *vec_it << std::endl;
		if(*vec_it != *it) break;
	} while( (it != buf.begin()) && (vec_it != seq.begin()) );
	if( (it == buf.begin()) && (vec_it == seq.begin()) ) return true;
	else if( (it == buf.begin()) && (vec_it != seq.begin()) ) str << "arrived back early" << std::endl;
	else if( (it != buf.begin()) && (vec_it == seq.begin()) ) str << "arrived back late" << std::endl;
	
	std::cerr << "sequence mismatch (backwards): " << std::endl << str.str() << std::endl;	
	
	return false;
}

ndarray<2, int> make_frame(const ndsize<2>& shape, int i);

int frame_index(const ndarray_view<2, int>&, bool verify = false);

bool compare_frames(const ndsize<2>& shape, const ndarray_view<3, int>& frames, const std::vector<int>& is);

}}

#endif
