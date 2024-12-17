// gpu-side path tracing (wavefront)

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

__global int pathCount, shadowRays;

// struct for rendering parameters - keep in sync with CPU side.
struct RenderData
{
	float4 eye, C, p0, p1, p2;
	uint frameIdx, scrWidth, scrHeight;
	uint dummy;
};

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
void kernel ResetCounters( primaryRayCount )
{
	if (get_global_id( 0 ) != 0) return;
	pathCount = primaryRayCount;
	shadowRays = 0;
}

// primary ray generation
void kernel Generate( struct RenderData rd, global PathState* raysOut  )
{
	const unsigned x = get_global_id( 0 ), y = get_global_id( 1 );
	const float u = (float)x / (float)rd.scrWidth;
	const float v = (float)y / (float)rd.scrHeight;
	const float3 P = rd.p0 + u * (rd.p1 - rd.p0) + v * (rd.p2 - rd.p0);
	raysOut[threadId].T = (float4)( 1, 1, 1, 1 /* pdf */ );
	raysOut[threadId].O = (float4)( eye, as_float( x + y * rd * scrWidth ) );
	raysOut[threadId].D = (float4)( normalize( P - eye ), 1e30f );
}

// extend: trace the generated rays to find the nearest intersection point.
void kernel Extend( global PathState* raysIn )
{
	// obtain task
	const int pathId = atomic_dec( &pathCount ) - 1;
	if (pathId < 0) return;
	const O4 = raysIn[pathId].O;
	const D4 = raysIn[pathId].D;
	const float3 rD = (float3)( 1.0f / D4.x, 1.0f / D4.y, 1.0f / D4.z, 1 );
	raysIn[pathId].hit = traverse_ailalaine( O4.xyz, D4.xyz, rD, 1e30f );
}

// shade: process intersection results; this evaluates the BRDF and creates extension
// rays and shadow rays.
void kernel Shade( global float4* accumulator, global float4* raysIn, global float4* raysOut, global float4* shadowOut )
{
}

// connect: trace shadow rays and deposit their potential contribution to the pixels
// if not occluded.
void kernel Connect( global float4* accumulator, global Potential* shadowIn )
{
	// obtain task
	const int rayId = atomic_dec( &shadowRays ) - 1;
	if (rayId < 0) return;
	const T4 = shadowIn[rayId].T;
	const O4 = shadowIn[rayId].O;
	const D4 = shadowIn[rayId].D;
	const float3 rD = (float3)( 1.0f / D4.x, 1.0f / D4.y, 1.0f / D4.z, 1 );
	bool occluded = isoccluded_ailalaine( O4.xyz, D4.xyz, rD, D4.w );
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
	const int r = (int)(255.0f * min( 1.0f, sqrtf( p.x ) ));
	const int g = (int)(255.0f * min( 1.0f, sqrtf( p.y ) ));
	const int b = (int)(255.0f * min( 1.0f, sqrtf( p.z ) ));
	pixels[pixelIdx] = (r << 16) + (g << 8) + b;
}