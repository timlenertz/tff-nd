#ifndef TLZ_ND_CONFIG_H_
#define TLZ_ND_CONFIG_H_

#define TLZ_DEBUG_BUILD !defined(NDEBUG)

// TLZ_STANDALONE:
// if defined, helpers from standalone_helpers/ are included (from common.h)
// otherwise helpers are included from outside (tff framework)

#ifndef TLZ_ND_WITH_ELEM
#define TLZ_ND_WITH_ELEM 1
#endif

#ifndef TLZ_ND_WITH_EXCEPTIONS
#define TLZ_ND_WITH_EXCEPTIONS 1
#endif

#ifndef TLZ_ND_WITH_WRAPAROUND
#define TLZ_ND_WITH_WRAPAROUND 1
#endif

#ifndef TLZ_ND_WITH_ALLOCATION
#define TLZ_ND_WITH_ALLOCATION 1
#endif

#ifndef TLZ_ND_WITH_OPAQUE
#define TLZ_ND_WITH_OPAQUE 1
#endif

#endif
