/*
 * Prepended to every program by oclc to work around IMG's OpenCL front end.
 *
 * The front end runs each work-item as its own work-group (get_local_size()
 * is 1), so nothing else can see an item's local memory and the local-memory
 * atomics are done as plain read-modify-write.  This also avoids two front
 * end bugs: a scalar __local variable whose address is taken shares its
 * register with its value, and atomics drop constant address offsets
 * (atomic_inc(&p[3]) operates on p[0]).  Global atomics keep the hardware
 * lock but get an offset the front end cannot fold: get_work_dim() is at
 * most 3, so the shift is always zero.
 */
#define __VITA_NOFOLD (get_work_dim() >> 8)

/* The front end defines this, but the device has no image support. */
#undef __IMAGE_SUPPORT__

#define __VITA_ATOMIC_1(name, op, T)							\
	T __attribute__((overloadable)) __vita_##name(__global volatile T *p)		\
	{ return name(p + __VITA_NOFOLD); }						\
	T __attribute__((overloadable)) __vita_##name(__local volatile T *vp)		\
	{ __local T *p = (__local T *)vp; T o = *p; *p = op; return o; }
#define __VITA_ATOMIC_2(name, op, T)							\
	T __attribute__((overloadable)) __vita_##name(__global volatile T *p, T v)	\
	{ return name(p + __VITA_NOFOLD, v); }						\
	T __attribute__((overloadable)) __vita_##name(__local volatile T *vp, T v)	\
	{ __local T *p = (__local T *)vp; T o = *p; *p = op; return o; }
#define __VITA_ATOMIC_INT(def, name, op)						\
	def(name, op, int)								\
	def(name, op, uint)

__VITA_ATOMIC_INT(__VITA_ATOMIC_2, atomic_add, o + v)
__VITA_ATOMIC_INT(__VITA_ATOMIC_2, atomic_sub, o - v)
__VITA_ATOMIC_INT(__VITA_ATOMIC_2, atomic_xchg, v)
__VITA_ATOMIC_2(atomic_xchg, v, float)
__VITA_ATOMIC_INT(__VITA_ATOMIC_1, atomic_inc, o + 1)
__VITA_ATOMIC_INT(__VITA_ATOMIC_1, atomic_dec, o - 1)
__VITA_ATOMIC_INT(__VITA_ATOMIC_2, atomic_min, min(o, v))
__VITA_ATOMIC_INT(__VITA_ATOMIC_2, atomic_max, max(o, v))
__VITA_ATOMIC_INT(__VITA_ATOMIC_2, atomic_and, o & v)
__VITA_ATOMIC_INT(__VITA_ATOMIC_2, atomic_or, o | v)
__VITA_ATOMIC_INT(__VITA_ATOMIC_2, atomic_xor, o ^ v)

#define __VITA_CMPXCHG(T)								\
	T __attribute__((overloadable)) __vita_atomic_cmpxchg(__global volatile T *p, T c, T v) \
	{ return atomic_cmpxchg(p + __VITA_NOFOLD, c, v); }				\
	T __attribute__((overloadable)) __vita_atomic_cmpxchg(__local volatile T *vp, T c, T v) \
	{ __local T *p = (__local T *)vp; T o = *p; *p = o == c ? v : o; return o; }
__VITA_CMPXCHG(int)
__VITA_CMPXCHG(uint)

#define atomic_add(p, v)	__vita_atomic_add(p, v)
#define atomic_sub(p, v)	__vita_atomic_sub(p, v)
#define atomic_xchg(p, v)	__vita_atomic_xchg(p, v)
#define atomic_inc(p)		__vita_atomic_inc(p)
#define atomic_dec(p)		__vita_atomic_dec(p)
#define atomic_cmpxchg(p, c, v)	__vita_atomic_cmpxchg(p, c, v)
#define atomic_min(p, v)	__vita_atomic_min(p, v)
#define atomic_max(p, v)	__vita_atomic_max(p, v)
#define atomic_and(p, v)	__vita_atomic_and(p, v)
#define atomic_or(p, v)		__vita_atomic_or(p, v)
#define atomic_xor(p, v)	__vita_atomic_xor(p, v)
#define atom_add(p, v)		__vita_atomic_add(p, v)
#define atom_sub(p, v)		__vita_atomic_sub(p, v)
#define atom_xchg(p, v)		__vita_atomic_xchg(p, v)
#define atom_inc(p)		__vita_atomic_inc(p)
#define atom_dec(p)		__vita_atomic_dec(p)
#define atom_cmpxchg(p, c, v)	__vita_atomic_cmpxchg(p, c, v)
#define atom_min(p, v)		__vita_atomic_min(p, v)
#define atom_max(p, v)		__vita_atomic_max(p, v)
#define atom_and(p, v)		__vita_atomic_and(p, v)
#define atom_or(p, v)		__vita_atomic_or(p, v)
#define atom_xor(p, v)		__vita_atomic_xor(p, v)
