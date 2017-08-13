#include "image.h"
#include <cstdio>
#include <cassert>
#include <png.h>
#include <csetjmp>

namespace tlz_ex {

void write_png(const std::string& filename, const tff::ndarray_view<2, rgb_color>& vw) {
	std::FILE* file = std::fopen(filename.c_str(), "wb");
	assert(file != nullptr);
	
	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
	assert(png_ptr != nullptr);
	png_infop info_ptr = png_create_info_struct(png_ptr);
	assert(info_ptr != nullptr);

	if(setjmp(png_jmpbuf(png_ptr))) std::terminate();
	
	png_init_io(png_ptr, file);
	
	png_set_IHDR(
		png_ptr,
		info_ptr,
		vw.shape()[1],
		vw.shape()[0],
		8,
		PNG_COLOR_TYPE_RGB,
		PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_DEFAULT,
		PNG_FILTER_TYPE_DEFAULT
	);
	
	png_write_info(png_ptr, info_ptr);
	
	tff::ndarray<1, rgb_color> row_buffer(tff::make_ndsize(vw.shape()[1]));
	for(std::ptrdiff_t row = 0; row < vw.shape()[0]; ++row) {
		row_buffer = vw[row];
		png_write_row(png_ptr, reinterpret_cast<png_bytep>(row_buffer.start()));
	}
	
	png_write_end(png_ptr, nullptr);
	
	fclose(file);
	png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
	png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
}


tff::ndarray<2, rgb_color> read_png(const std::string& filename) {
	std::FILE* file = std::fopen(filename.c_str(), "rb");
	assert(file != nullptr);
	
	unsigned char header[8];
	std::fread(header, 1, 8, file);
	assert(png_sig_cmp(header, 0, 8) == 0);
	
	png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
	assert(png_ptr != nullptr);
	
	png_infop info_ptr = png_create_info_struct(png_ptr);
	assert(info_ptr != nullptr);
	
	if(setjmp(png_jmpbuf(png_ptr))) std::terminate();
	
 	png_init_io(png_ptr, file);
    png_set_sig_bytes(png_ptr, 8);

	png_read_png(
		png_ptr,
		info_ptr,
		PNG_TRANSFORM_STRIP_16 |
			PNG_TRANSFORM_STRIP_ALPHA |
			PNG_TRANSFORM_PACKING |
			PNG_TRANSFORM_PACKING |
			PNG_TRANSFORM_EXPAND,
		nullptr
	);
	
	png_uint_32 width, height;
	int bit_depth, color_type, interlace_type;
	png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type, &interlace_type, nullptr, nullptr);
	   
	png_bytepp row_pointers = png_get_rows(png_ptr, info_ptr);
	assert(png_get_rowbytes(png_ptr, info_ptr) == width * sizeof(rgb_color));
	
	tff::ndarray<2, rgb_color> arr(tff::make_ndsize(height, width));
	for(std::ptrdiff_t row = 0; row < height; ++row) {
		const rgb_color* row_ptr = reinterpret_cast<const rgb_color*>(row_pointers[row]);
		tff::ndarray_view<1, const rgb_color> row_vw(row_ptr, tff::make_ndsize(width));
		arr[row] = row_vw;
	}
	
	png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
	std::fclose(file);
	
	return arr;
}


}