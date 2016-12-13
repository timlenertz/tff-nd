#include "ndarray.h"
#include <random>

namespace tff { namespace test {

int obj_t::counter = 0;
int nonpod_frame_handle::counter = 0;

ndarray<2, int> make_frame(const ndsize<2>& shape, int i) {
	std::mt19937 generator(i);
	std::uniform_int_distribution<int> dist;
	ndarray<2, int> frame(shape);
	for(std::ptrdiff_t y = 0; y < shape[0]; ++y)
	for(std::ptrdiff_t x = 0; x < shape[1]; ++x)
		frame[y][x] = 0;
	frame[0][0] = i;
	return frame;
}


int frame_index(const ndarray_view<2, int>& vw, bool verify) {
	const auto& shp = vw.shape();
	int i = vw[0][0];
	if(verify) {
		if(vw == make_frame(shp, i)) return i;
		else return -1;
	} else {
		return i;
	}
}


bool compare_frames(const ndsize<2>& shape, const ndarray_view<3, int>& frames, const std::vector<int>& is) {
	if(frames.shape().front() != is.size()) return false;
	for(std::ptrdiff_t i = 0; i < is.size(); ++i) {
		auto expected = make_frame(shape, is[i]);
		if(frames[i] != expected) return false;
	}
	return true;
}

}}
