#include "cthead.h"
#include <cstdio>

namespace tff_ex {

tff::ndarray<3, std::int16_t> read_cthead(const std::string& dirname) {
	auto shape = tff::make_ndsize(113, 256, 256);

	tff::ndarray<3, std::int16_t> out(tff::make_ndsize(113, 256, 256));
	for(std::ptrdiff_t z = 0; z < shape[0]; ++z) {
		std::string filename = dirname;
		if(filename.back() != '/') filename += '/';
		filename += "CThead." + std::to_string(z + 1);
		
		std::FILE* file = std::fopen(filename.c_str(), "rb");
		std::fread(out[z].start(), 2, shape[1]*shape[2], file);
		std::fclose(file);
	}
	
	return out;
}

}