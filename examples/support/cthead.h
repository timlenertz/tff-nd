#ifndef TFF_ND_EX_CTHEAD_H_
#define TFF_ND_EX_CTHEAD_H_

#include <cstdint>
#include <string>
#include "../../src/ndarray.h"

namespace tlz_ex {

tff::ndarray<3, std::int16_t> read_cthead(const std::string& dirname);

}

#endif