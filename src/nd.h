#ifndef TFF_ND_H_
#define TFF_ND_H_

#include "config.h"
#include "common.h"

#if TFF_ND_WITH_ELEM
	#include "elem.h"
	#include "elem_tuple.h"
#endif

#include "ndcoord.h"
#include "ndcoord_dyn.h"
#include "ndspan.h"
#include "ndspan_iterator.h"

#include "pod_array_format.h"

#include "ndarray_view.h"
#include "ndarray_iterator.h"
#include "ndarray_view_cast.h"

#if TFF_ND_WITH_WRAPAROUND
	#include "ndarray_wraparound_view.h"
#endif

#if TFF_ND_WITH_TIMED
	#include "ndarray_timed_view.h"
	#if TFF_ND_WITH_WRAPAROUND
		#include "ndarray_timed_wraparound_view.h"
	#endif
#endif

#if TFF_ND_WITH_ALLOCATION
	#include "ndarray.h"
#endif

#if TFF_ND_WITH_OPAQUE
	#include "opaque/ndarray_opaque_view.h"
	#include "opaque/ndarray_opaque_iterator.h"
	#include "opaque/ndarray_opaque_view_cast.h"
	#if TFF_ND_WITH_ALLOCATION
		#include "opaque/ndarray_opaque.h"
	#endif
	#if TFF_ND_WITH_TIMED
		#include "opaque/ndarray_timed_opaque_view.h"
	#endif
	#include "opaque_format/ndarray.h"
	#include "opaque_format/raw.h"
#endif

#endif
