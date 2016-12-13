#ifndef TFF_ND_CONFIG_H_
#define TFF_ND_CONFIG_H_

#define TFF_DEBUG_BUILD !defined(NDEBUG)

// TFF_STANDALONE: 
// if defined, helpers from standalone_helpers/ are included (from common.h)
// otherwise helpers are included from outside (tff framework)

#ifndef TFF_ND_WITH_ELEM
#define TFF_ND_WITH_ELEM 1
#endif

#ifndef TFF_ND_WITH_EXCEPTIONS
#define TFF_ND_WITH_EXCEPTIONS 1
#endif

#ifndef TFF_ND_WITH_TIMED
#define TFF_ND_WITH_TIMED 1
#endif

#ifndef TFF_ND_WITH_ALLOCATION
#define TFF_ND_WITH_ALLOCATION 1
#endif

#ifndef TFF_ND_WITH_OPAQUE
#define TFF_ND_WITH_OPAQUE 1
#endif

#endif
