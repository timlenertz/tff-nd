#ifndef TLZ_ND_COMMON_H_
#define TLZ_ND_COMMON_H_

#ifdef TLZ_ND_STANDALONE

#include "standalone_helpers/assert.h"
#include "standalone_helpers/misc.h"
#include "standalone_helpers/raw_allocator.h"

#else

#include "../common.h"
#include "../utility/assert.h"
#include "../utility/misc.h"
#include "../utility/raw_allocator.h"

#endif

#endif
