#define N 4
typedef struct { uint a; uint b[3]; int c; } S;
__kernel void shapes(__global uint *u, __global S *s, __global int *o)
{
	o[0] = atomic_inc(&u[3]);
}
__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void loc(__global uint *o, __local uint *lp)
{
	__local uint count;
	__local volatile int arr[N], arr2[2][N];
	__local S st;
	__local float *p = (__local float *)lp;
	size_t i = get_global_id(0);
	count = 10;
	arr[2] = -1; arr2[1][3] = 5; st.c = 4;
	lp[1] = 7;
	uint a = atomic_inc(&count);
	int b = atomic_max(&arr[2], (int)i - 2);
	uint c = atomic_cmpxchg(&lp[1], 7u, 9u);
	o[i * 4 + 0] = a + count * 100;
	o[i * 4 + 1] = b + arr[2] * 100 + arr2[1][3] + st.c + sizeof(arr2);
	o[i * 4 + 2] = c + lp[1] * 100 + (uint)p[0];
	o[i * 4 + 3] = atomic_xchg(&count, 3u) + count * 100;
}
kernel void empty(void) { local int z; z = 1; }
