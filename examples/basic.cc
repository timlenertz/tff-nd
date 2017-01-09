#include "../src/nd.h"
#include "support/image.h"
#include "support/cthead.h"
#include <algorithm>

using namespace tff;
using namespace tff_ex;


int main() {
	auto lena = read_png("lena.png");
	using decomp_view_type = ndarray_view<3, std::uint8_t>;
	auto decomp_view = ndarray_view_cast<decomp_view_type>(lena.view());
	//decomp_view.slice(0, 2) = reverse(decomp_view.slice(1, 2), 0);
	//decomp_view.slice(1, 2) = reverse(decomp_view.slice(1, 2), 1);

	//swapaxis(decomp_view(100, 200)(100, 200)(), 0, 2) = wraparound(decomp_view, make_ndptrdiff(400, 200, 0), make_ndptrdiff(403, 300, 100));
	
	lena(0, 256)(0,-2,2) = lena(256, 512, -1)(1,-1,2);

	write_png("lena_out.png", lena.view());
}


/*
int main() {
	auto lena = read_png("lena.png");
	auto lena_wrap = wraparound(step(lena.view(), 0, 3), make_ndptrdiff(-200, -40), make_ndptrdiff(600, 2000));
	ndarray<2, rgb_color> lena_out(lena_wrap);
	write_png("lena_out.png", lena_out.view());
}
*/

/*
int main() {
	auto cthead = read_cthead("cthead/");

	for(std::ptrdiff_t z = 0; z < cthead.shape()[2]; ++z) {
		//auto slice = cthead[z];
		auto slice = cthead.slice(z, 2);
		ndarray<2, rgb_color> img(slice.shape());
		std::transform(slice.begin(), slice.end(), img.begin(), [](std::int16_t val_sw) {
			int byte_mask = (1 << 8) - 1;
			std::int16_t val = ((val_sw & byte_mask) << 8) + ((val_sw & (byte_mask<<8)) >> 8);
			if(val < 0) std::abort();
			std::uint16_t uval = val;
			std::uint8_t b = uval >> 5;
			b *= 1.80;
			return rgb_color{b, b, b};
		});
		write_png("out/head_slice_" + std::to_string(z) + ".png", img.view());
	
	}

}
*/