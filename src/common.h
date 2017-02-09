#ifndef TFF_ND_COMMON_H_
#define TFF_ND_COMMON_H_

#ifdef TFF_ND_STANDALONE

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
