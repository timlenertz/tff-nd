#ifndef TFF_ND_COMMON_H_
#define TFF_ND_COMMON_H_

#ifdef TFF_ND_STANDALONE

#include "standalone_helpers/assert.h"
#include "standalone_helpers/misc.h"
#include "standalone_helpers/raw_allocator.h"

#else

#ifndef TFF_ND_HAS_DEPENDENCIES
#error TFF_ND_HAS_DEPENDENCIES not defined before including tff-nd
#endif

#endif

#endif
