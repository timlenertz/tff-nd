#ifndef TLZ_ND_ASSERT_H_
#define TLZ_ND_ASSERT_H_

#include "../config.h"

#define TLZ_STRINGIZE_(X) #X
#define TLZ_STRINGIZE(X) TLZ_STRINGIZE_(X)

#define TLZ_GET_NARG_MACRO_2(_1, _2, NAME, ...) NAME

#ifdef _MSC_VER
	#define TLZ_ASSUME(__invariant__) __assume(__invariant__)
#else
	#define TLZ_ASSUME(__invariant__) __builtin_assume(__invariant__)
#endif


#if 1||TLZ_ND_WITH_EXCEPTIONS

	#if TLZ_DEBUG_BUILD
		#define TLZ_ASSERT_CRIT_MSG_(__condition__, __msg__) \
			if(! (__condition__)) throw ::tlz::failed_assertion(__msg__ " at " __FILE__ ":" TLZ_STRINGIZE(__LINE__))
		#define TLZ_ASSERT_MSG_(__condition__, __msg__) \
			if(! (__condition__)) throw ::tlz::failed_assertion(__msg__ " at " __FILE__ ":" TLZ_STRINGIZE(__LINE__))
	#else 
		#define TLZ_ASSERT_CRIT_MSG_(__condition__, __msg__) \
			(void)0
		#define TLZ_ASSERT_MSG_(__condition__, __msg__) \
			if(! (__condition__)) throw ::tlz::failed_assertion(__msg__ " at " __FILE__ ":" TLZ_STRINGIZE(__LINE__))
	#endif

#else

	#if TLZ_DEBUG_BUILD
		#define TLZ_ASSERT_CRIT_MSG_(__condition__, __msg__) \
			if(! (__condition__)) std::terminate()
		#define TLZ_ASSERT_MSG_(__condition__, __msg__) \
			if(! (__condition__)) std::terminate()
	#else 
		#define TLZ_ASSERT_CRIT_MSG_(__condition__, __msg__) \
			(void)0
		#define TLZ_ASSERT_MSG_(__condition__, __msg__) \
			if(! (__condition__)) std::terminate()
	#endif

#endif


#define TLZ_ASSERT_(__condition__) TLZ_ASSERT_MSG_(__condition__, "`" #__condition__ "`")
#define TLZ_ASSERT_CRIT_(__condition__) TLZ_ASSERT_CRIT_MSG_(__condition__, "`" #__condition__ "`")

#ifdef _MSC_VER
	// workaround for MSVC: http://stackoverflow.com/a/5134656/4108376
	#define TLZ_EXPAND_(x) x
	#define TLZ_ASSERT(...) TLZ_EXPAND_( TLZ_GET_NARG_MACRO_2(__VA_ARGS__, TLZ_ASSERT_MSG_, TLZ_ASSERT_, IGNORE)(__VA_ARGS__) )
	#define TLZ_ASSERT_CRIT(...) TLZ_EXPAND_( TLZ_GET_NARG_MACRO_2(__VA_ARGS__, TLZ_ASSERT_CRIT_MSG_, TLZ_ASSERT_CRIT_, IGNORE)(__VA_ARGS__) )
#else
	#define TLZ_ASSERT(...) TLZ_GET_NARG_MACRO_2(__VA_ARGS__, TLZ_ASSERT_MSG_, TLZ_ASSERT_, IGNORE)(__VA_ARGS__)
	#define TLZ_ASSERT_CRIT(...) TLZ_GET_NARG_MACRO_2(__VA_ARGS__, TLZ_ASSERT_CRIT_MSG_, TLZ_ASSERT_CRIT_, IGNORE)(__VA_ARGS__)
#endif

#define Assert TLZ_ASSERT
#define Assert_crit TLZ_ASSERT_CRIT


///////////////


#if 1||TLZ_ND_WITH_EXCEPTIONS

#include <stdexcept>

namespace tlz {

class failed_assertion : public std::runtime_error {
public:
	using std::runtime_error::runtime_error;
};

}

#endif

#endif
