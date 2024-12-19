// gpu-side path tracing (wavefront)

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#include "traverse.cl"

#define PI			3.14159265358979323846264f
#define INVPI		0.31830988618379067153777f
#define INV2PI		0.15915494309189533576888f
#define TWOPI		6.28318530717958647692528f

// struct for rendering parameters - keep in sync with CPU side.
struct RenderData
{
	// camera setup
	float4 eye, C, p0, p1, p2;
	uint frameIdx, dummy1, dummy2, dummy3;
	// BVH data
	global float4* cwbvhNodes;
	global float4* cwbvhTris;
};

__global volatile int extendTasks, shadeTasks, connectTasks;
__global struct RenderData rd;


// Xor32 RNG
uint WangHash( uint s ) { s = (s ^ 61) ^ (s >> 16), s *= 9, s = s ^ (s >> 4), s *= 0x27d4eb2d; return s ^ (s >> 15); }
uint RandomUInt( uint* seed ) { *seed ^= *seed << 13, *seed ^= *seed >> 17, *seed ^= *seed << 5; return *seed; }
float RandomFloat( uint* seed ) { return RandomUInt( seed ) * 2.3283064365387e-10f; }

// DiffuseReflection: Uniform random bounce in the hemisphere
float3 DiffuseReflection( float3 N, uint* seed )
{
	float3 R;
	do
	{
		R = (float3)( RandomFloat( seed ) * 2 - 1, RandomFloat( seed ) * 2 - 1, RandomFloat( seed ) * 2 - 1 );
	} while (dot( R, R ) > 1);
	return normalize( dot( R, N ) > 0 ? R : -R );
}

// CosWeightedDiffReflection: Cosine-weighted random bounce in the hemisphere
float3 CosWeightedDiffReflection( const float3 N, uint* seed )
{
	float3 R;
	do
	{
		R = (float3)( RandomFloat( seed ) * 2 - 1, RandomFloat( seed ) * 2 - 1, RandomFloat( seed ) * 2 - 1 );
	} while (dot( R, R ) > 1);
	return normalize( N + normalize( R ) );
}

// PathState: path throughput, current extension ray, pixel index
struct PathState
{
	float4 T; // xyz = rgb, postponed pdf in w
	float4 O; // pixel index and path depth in O.w
	float4 D; // t in D.w
	float4 hit;
};

// Potential contribution: shadoww ray origin & dir, throughput
struct Potential 
{ 
	float4 T;
	float4 O; // pixel index in O.w
	float4 D; // t in D.w
};

// atomic counter management - prepare for primary ray wavefront
void kernel SetRenderData( 
	int primaryRayCount,
	float4 eye, float4 p0, float4 p1, float4 p2,
	uint frameIdx,
	global float4* cwbvhNodes,
	global float4* cwbvhTris
)
{
	if (get_global_id( 0 ) != 0) return;
	// set camera parameters
	rd.eye = eye;
	rd.p0 = p0, rd.p1 = p1, rd.p2 = p2;
	rd.frameIdx = frameIdx;
	// set BVH pointers
	rd.cwbvhNodes = cwbvhNodes;
	rd.cwbvhTris = cwbvhTris;
	// initialize atomic counters
	extendTasks = primaryRayCount;
	shadeTasks = primaryRayCount;
	connectTasks = 0;
}

// clear accumulator
void kernel Clear( global float4* accumulator )
{
	const uint pixelIdx = get_global_id( 0 );
	accumulator[pixelIdx] = (float4)( 0 );
}

// primary ray generation
void kernel Generate( global struct PathState* raysOut  )
{
	const uint x = get_global_id( 0 ), y = get_global_id( 1 );
	const uint id = x + y * get_global_size( 0 );
	const float u = (float)x / (float)get_global_size( 0 );
	const float v = (float)y / (float)get_global_size( 1 );
	const float4 P = rd.p0 + u * (rd.p1 - rd.p0) + v * (rd.p2 - rd.p0);
	raysOut[id].T = (float4)( 1, 1, 1, 1 /* pdf */ );
	raysOut[id].O = (float4)( rd.eye.xyz, as_float( id << 4 /* low bits: depth */ ) );
	raysOut[id].D = (float4)( normalize( P.xyz - rd.eye.xyz ), 1e30f );
	raysOut[id].hit = (float4)( 1e30f, 0, 0, as_float( 0 ) );
}

// extend: trace the generated rays to find the nearest intersection point.
void kernel Extend( global struct PathState* raysIn )
{
	while (1)
	{
		// obtain task
		if (extendTasks < 1) break;
		const int pathId = atomic_dec( &extendTasks ) - 1;
		if (pathId < 0) break;
		const float4 O4 = raysIn[pathId].O;
		const float4 D4 = raysIn[pathId].D;
		const float3 rD = (float3)( 1.0f / D4.x, 1.0f / D4.y, 1.0f / D4.z );
		raysIn[pathId].hit = traverse_cwbvh( rd.cwbvhNodes, rd.cwbvhTris, O4.xyz, D4.xyz, rD, 1e30f );
	}
}

// syncing counters: at this point, we need to reset the extendTasks counter.
void kernel UpdateCounters1()
{
	if (get_global_id( 0 ) != 0) return;
	extendTasks = 0;
}

// shade: process intersection results; this evaluates the BRDF and creates 
// extension rays and shadow rays.
void kernel Shade( 
	global float4* accumulator, 
	global struct PathState* raysIn, 
	global struct PathState* raysOut, 
	global struct Potential* shadowOut,
	global float4* verts
)
{
	while (1)
	{
		// obtain task
		if (shadeTasks < 1) break;
		const int pathId = atomic_dec( &shadeTasks ) - 1;
		if (pathId < 0) break;
		// fetch path data
		float4 data0 = raysIn[pathId].T;	// xyz = rgb, postponed pdf in w
		float4 data1 = raysIn[pathId].O;	// pixel index in O.w
		float4 data2 = raysIn[pathId].D;	// t in D.w
		float4 data3 = raysIn[pathId].hit;	// dist, u, v, prim
		// prepare for shading
		uint depth = as_uint( data1.w ) & 15;
		uint pixelIdx = as_uint( data1.w ) >> 4;
		uint seed = WangHash( as_uint( data1.w ) + rd.frameIdx * 17117);
		// end path on sky
		if (data3.x == 1e30f)
		{
			float3 skyColor = (float3)( 0.7f, 0.7f, 1.2f );
			accumulator[pixelIdx] += (float4)( data0.xyz * skyColor, 1 );
			continue;
		}
		// fetch geometry at intersection point
		uint vertIdx = as_uint( data3.w ) * 3;
		float4 v0 = verts[vertIdx];
		float3 vert0 = v0.xyz, vert1 = verts[vertIdx + 1].xyz, vert2 = verts[vertIdx + 2].xyz;
		float3 N = normalize( cross( vert1 - vert0, vert2 - vert0 ) );
		float3 D = data2.xyz;
		if (dot( N, D ) > 0) N *= -1;
		float3 T = data0.xyz;
		float3 O = data1.xyz;
		float t = data3.x;
		float3 I = O + t * D;
		// bounce
		if (depth < 4)
		{
			uint newRayIdx = atomic_inc( &extendTasks );
			float3 BRDF = (float3)(1) /* just white for now */ * INVPI;
		#if 0
			float3 R = DiffuseReflection( N, &seed );
			float PDF = INV2PI;
		#else
			float3 R = CosWeightedDiffReflection( N, &seed );
			float PDF = dot( N, R ) * INVPI;
		#endif
			T *= dot( N, R ) * BRDF * (1.0f / PDF);
			raysOut[newRayIdx].T = (float4)( T, 1 );
			raysOut[newRayIdx].O = (float4)( I + R * 0.001f, as_float( (pixelIdx << 4) + depth + 1 ) );
			raysOut[newRayIdx].D = (float4)( R, 1e30f );
		}
	}
}

// syncing counters: we generated extensions; those will need shading too.
void kernel UpdateCounters2()
{
	if (get_global_id( 0 ) != 0) return;
	shadeTasks = extendTasks;
}

// connect: trace shadow rays and deposit their potential contribution to the pixels
// if not occluded.
void kernel Connect( global float4* accumulator, global struct Potential* shadowIn )
{
	// obtain task
	const int rayId = atomic_dec( &connectTasks ) - 1;
	if (rayId < 0) return;
	const float4 T4 = shadowIn[rayId].T;
	const float4 O4 = shadowIn[rayId].O;
	const float4 D4 = shadowIn[rayId].D;
	const float3 rD = (float3)( 1.0f / D4.x, 1.0f / D4.y, 1.0f / D4.z );
	bool occluded = false; // isoccluded_cwbvh( rd.cwbvhNodes, rd.cwbvhTris, O4.xyz, D4.xyz, rD, D4.w );
	if (occluded) return;
	uint pixelIdx = as_uint( O4.w );
	accumulator[pixelIdx] += T4;
}

// finalize: convert the accumulated values into final pixel values.
// NOTE: rendering result is emitted to global uint array, which needs to be copied back 
// to the host. This is not efficient. A proper scheme should use OpenGL / D3D / Vulkan 
// interop do write directly to a texture.
void kernel Finalize( global float4* accumulator, const float scale, global uint* pixels )
{
	const uint x = get_global_id( 0 ), y = get_global_id( 1 );
	const uint pixelIdx = x + y * get_global_size( 0 );
	const float4 p = accumulator[pixelIdx] * scale;
	const int r = (int)(255.0f * min( 1.0f, sqrt( p.x ) ));
	const int g = (int)(255.0f * min( 1.0f, sqrt( p.y ) ));
	const int b = (int)(255.0f * min( 1.0f, sqrt( p.z ) ));
	pixels[pixelIdx] = (r << 16) + (g << 8) + b;
}