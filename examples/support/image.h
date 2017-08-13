#ifndef TFF_ND_EX_IMAGE_H_
#define TFF_ND_EX_IMAGE_H_

#include <cstdint>
#include <string>
#include "../../src/elem.h"
#include "../../src/ndarray.h"
#include "../../src/ndarray_view.h"

namespace tlz_ex {

struct rgb_color {
	std::uint8_t r;
	std::uint8_t g;
	std::uint8_t b;
};

void write_png(const std::string& filename, const tff::ndarray_view<2, rgb_color>&);
tff::ndarray<2, rgb_color> read_png(const std::string& filename);

}

#endif


namespace tlz {

	template<>
	struct elem_traits<tff_ex::rgb_color> : elem_traits_base<tff_ex::rgb_color, std::uint8_t, 3, false> { };

}
