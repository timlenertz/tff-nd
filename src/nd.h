#ifndef TLZ_ND_H_
#define TLZ_ND_H_

#include "config.h"
#include "common.h"

#if TLZ_ND_WITH_ELEM
	#include "elem.h"
	#include "elem_tuple.h"
#endif

#include "ndcoord.h"
#include "ndcoord_dyn.h"
#include "ndspan.h"
#include "ndspan_iterator.h"

#include "pod_array_format.h"

#include "ndarray_traits.h"
#include "ndarray_view.h"
#include "ndarray_iterator.h"
#include "ndarray_view_cast.h"
#include "ndarray_view_operations.h"

#if TLZ_ND_WITH_WRAPAROUND
	#include "ndarray_wraparound_view.h"
#endif

#if TLZ_ND_WITH_ALLOCATION
	#include "ndarray.h"
#endif

#if TLZ_ND_WITH_OPAQUE
	#include "opaque/ndarray_opaque_traits.h"
	#include "detail/ndarray_opaque_view_wrapper.h"
	#include "opaque/ndarray_opaque_view_cast.h"
	#if TLZ_ND_WITH_WRAPAROUND
		#include "opaque/ndarray_wraparound_opaque_view.h"
		#include "opaque/ndarray_wraparound_opaque_view_cast.h"
	#endif
	#if TLZ_ND_WITH_ALLOCATION
		#include "opaque/ndarray_opaque.h"
	#endif
	#include "opaque_format/ndarray.h"
	#include "opaque_format/raw.h"
#endif

#endif
