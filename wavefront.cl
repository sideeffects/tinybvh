// gpu-side path tracing (wavefront)

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#include "traverse.cl"

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

// PathState: path throughput, current extension ray, pixel index
struct PathState
{
	float4 T; // xyz = rgb, postponed pdf in w
	float4 O; // pixel index in O.w
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
	unsigned frameIdx,
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
	const unsigned x = get_global_id( 0 ), y = get_global_id( 1 );
	const unsigned id = x + y * get_global_size( 0 );
	const float u = (float)x / (float)get_global_size( 0 );
	const float v = (float)y / (float)get_global_size( 1 );
	const float4 P = rd.p0 + u * (rd.p1 - rd.p0) + v * (rd.p2 - rd.p0);
	raysOut[id].T = (float4)( 1, 1, 1, 1 /* pdf */ );
	raysOut[id].O = (float4)( rd.eye.xyz, as_float( id ) );
	raysOut[id].D = (float4)( normalize( P.xyz - rd.eye.xyz ), 1e30f );
}

// extend: trace the generated rays to find the nearest intersection point.
void kernel Extend( global struct PathState* raysIn )
{
	while (1)
	{
		// obtain task
		const int pathId = atomic_dec( &extendTasks ) - 1;
		if (pathId < 0) break;
		const float4 O4 = raysIn[pathId].O;
		const float4 D4 = raysIn[pathId].D;
		const float3 rD = (float3)( 1.0f / D4.x, 1.0f / D4.y, 1.0f / D4.z );
		raysIn[pathId].hit = traverse_cwbvh( rd.cwbvhNodes, rd.cwbvhTris, O4.xyz, D4.xyz, rD, 1e30f );
	}
}

// shade: process intersection results; this evaluates the BRDF and creates extension
// rays and shadow rays.
void kernel Shade( global float4* accumulator, global struct PathState* raysIn, global struct PathState* raysOut, global struct Potential* shadowOut )
{
	while (1)
	{
		// obntain task
		const int pathId = atomic_dec( &shadeTasks ) - 1;
		if (pathId < 0) break;
		float4 data1 = raysIn[pathId].O;
		float4 data2 = raysIn[pathId].D;
		float4 data3 = raysIn[pathId].hit;
		uint pixelIdx = as_uint( data1.w );
		accumulator[pixelIdx] += (float4)(data3.x * 0.01f );
	}
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
	unsigned pixelIdx = as_uint( O4.w );
	accumulator[pixelIdx] += T4;
}

// finalize: convert the accumulated values into final pixel values.
// NOTE: rendering result is emitted to global uint array, which needs to be copied back 
// to the host. This is not efficient. A proper scheme should use OpenGL / D3D / Vulkan 
// interop do write directly to a texture.
void kernel Finalize( global float4* accumulator, const float scale, global uint* pixels )
{
	const unsigned x = get_global_id( 0 ), y = get_global_id( 1 );
	const unsigned pixelIdx = x + y * get_global_size( 0 );
	const float4 p = accumulator[pixelIdx] * scale;
	const int r = (int)(255.0f * min( 1.0f, sqrt( p.x ) ));
	const int g = (int)(255.0f * min( 1.0f, sqrt( p.y ) ));
	const int b = (int)(255.0f * min( 1.0f, sqrt( p.z ) ));
	pixels[pixelIdx] = (r << 16) + (g << 8) + b;
}