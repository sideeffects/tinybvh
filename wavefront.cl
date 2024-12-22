// basic gpu-side path tracing (wavefront)

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#include "traverse.cl"

#define PI			3.14159265358979323846264f
#define INVPI		0.31830988618379067153777f
#define INV2PI		0.15915494309189533576888f
#define TWOPI		6.28318530717958647692528f
#define EPSILON		0.0001f // for a 100^3 world

// struct for rendering parameters - keep in sync with CPU side.
struct RenderData
{
	// camera setup
	float4 eye, C, p0, p1, p2;
	uint frameIdx, width, height, dummy3;
	// BVH data
	global float4* cwbvhNodes;
	global float4* cwbvhTris;
	global uint* blueNoise;
};

__global volatile int extendTasks, shadeTasks, connectTasks;
__global struct RenderData rd;

// Xor32 RNG
uint WangHash( uint s ) { s = (s ^ 61) ^ (s >> 16), s *= 9, s = s ^ (s >> 4), s *= 0x27d4eb2d; return s ^ (s >> 15); }
uint RandomUInt( uint* seed ) { *seed ^= *seed << 13, * seed ^= *seed >> 17, * seed ^= *seed << 5; return *seed; }
float RandomFloat( uint* seed ) { return RandomUInt( seed ) * 2.3283064365387e-10f; }

// Blue noise interface for fixed 128x128x8 dataset.
float2 Noise( const uint x, const uint y, const uint page /* 0..7 */ )
{
	const uint ix = x & 127, iy = y & 127;
	const uint v2 = rd.blueNoise[(page << 14) + (iy << 7) + ix];
	const uint r = v2 >> 16, g = (v2 >> 8) & 255;
	return (float2)( (float)r * 0.00392f, (float)g * 0.00392f );
}

// Color conversion
float3 rgb32_to_vec3( uint c )
{
	return (float3)((float)((c >> 16) & 255), (float)((c >> 8) & 255), (float)(c & 255)) * 0.00392f;
}

// Specular reflection
float3 Reflect( const float3 D, const float3 N ) { return D - 2.0f * N * dot( N, D ); }

// DiffuseReflection: Uniform random bounce in the hemisphere
float3 DiffuseReflection( float3 N, uint* seed )
{
	float3 R;
	do
	{
		R = (float3)(RandomFloat( seed ) * 2 - 1, RandomFloat( seed ) * 2 - 1, RandomFloat( seed ) * 2 - 1);
	} while (dot( R, R ) > 1);
	return fast_normalize( dot( R, N ) > 0 ? R : -R );
}

// CosWeightedDiffReflection: Cosine-weighted random bounce in the hemisphere
float3 CosWeightedDiffReflection( const float3 N, const float r0, const float r1 )
{
	const float r = sqrt( 1 - r1 * r1 ), phi = 4 * PI * r0;
	const float3 R = (float3)( cos( phi ) * r, sin( phi ) * r, r1);
	return fast_normalize( N + R );
}

// PathState: path throughput, current extension ray, pixel index
#define PATH_LAST_SPECULAR 1
#define PATH_VIA_DIFFUSE 2
struct PathState
{
	float4 T; // xyz = rgb, postponed MIS pdf in w
	float4 O; // O.w: 24-bit pixel index, 4-bit path depth, 4-bit path flags
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
void kernel SetRenderData( int primaryRayCount,
	float4 eye, float4 p0, float4 p1, float4 p2, 
	uint frameIdx, uint width, uint height,
	global float4* cwbvhNodes, global float4* cwbvhTris,
	global uint* blueNoise
)
{
	if (get_global_id( 0 ) != 0) return;
	// set camera parameters
	rd.eye = eye;
	rd.p0 = p0, rd.p1 = p1, rd.p2 = p2;
	rd.frameIdx = frameIdx;
	rd.width = width;
	rd.height = height;
	// set BVH pointers
	rd.cwbvhNodes = cwbvhNodes;
	rd.cwbvhTris = cwbvhTris;
	rd.blueNoise = blueNoise;
	// initialize atomic counters
	extendTasks = primaryRayCount;
	shadeTasks = primaryRayCount;
	connectTasks = 0;
}

// clear accumulator
void kernel Clear( global float4* accumulator )
{
	const uint pixelIdx = get_global_id( 0 );
	accumulator[pixelIdx] = (float4)(0);
}

// primary ray generation
void kernel Generate( global struct PathState* raysOut, uint frameSeed )
{
	const uint x = get_global_id( 0 ), y = get_global_id( 1 );
	const uint id = x + y * get_global_size( 0 );
	uint seed = WangHash( id * 13131 + frameSeed );
	const float u = ((float)x + RandomFloat( &seed )) / (float)get_global_size( 0 );
	const float v = ((float)y + RandomFloat( &seed )) / (float)get_global_size( 1 );
	const float4 P = rd.p0 + u * (rd.p1 - rd.p0) + v * (rd.p2 - rd.p0);
	raysOut[id].T = (float4)(1, 1, 1, 1 );
	raysOut[id].O = (float4)(rd.eye.xyz, as_float( (id << 8) + PATH_LAST_SPECULAR ));
	raysOut[id].D = (float4)(fast_normalize( P.xyz - rd.eye.xyz ), 1e30f);
	raysOut[id].hit = (float4)(1e30f, 0, 0, as_float( 0 ));
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
		const float3 rD = native_recip( D4.xyz );
		raysIn[pathId].hit = traverse_cwbvh( rd.cwbvhNodes, rd.cwbvhTris, O4.xyz, D4.xyz, rD, 1e30f );
	}
}

// syncing counters: at this point, we need to reset the extendTasks and connectTasks counters.
void kernel UpdateCounters1()
{
	if (get_global_id( 0 ) != 0) return;
	extendTasks = 0;
}

// shade: process intersection results; this evaluates the BRDF and creates 
// extension rays and shadow rays.
void kernel Shade( global float4* accumulator,
	global struct PathState* raysIn, global struct PathState* raysOut,
	global struct Potential* shadowOut,
	global float4* verts, uint sampleIdx
)
{
	while (1)
	{
		// obtain task
		if (shadeTasks < 1) break;
		const int pathId = atomic_dec( &shadeTasks ) - 1;
		if (pathId < 0) break;
		// fetch path data
		float4 T4 = raysIn[pathId].T;		// xyz = rgb, postponed pdf in w
		float4 O4 = raysIn[pathId].O;		// pixel index in O.w
		float4 D4 = raysIn[pathId].D;		// t in D.w
		float4 hit = raysIn[pathId].hit;	// dist, u, v, prim
		// prepare for shading
		uint pathState = as_uint( O4.w );
		uint pixelIdx = pathState >> 8;
		uint depth = (pathState >> 4) & 15;
		uint seed = WangHash( as_uint( O4.w ) + rd.frameIdx * 17117 );
		float3 T = T4.xyz;
		float t = hit.x;
		// end path on sky
		if (t == 1e30f)
		{
			float3 skyColor = (float3)(0.7f, 0.7f, 1.2f);
			accumulator[pixelIdx] += (float4)(T * skyColor, 1);
			continue;
		}
		// fetch geometry at intersection point
		uint vertIdx = as_uint( hit.w ) * 3;
		float4 v0 = verts[vertIdx];
		uint mat = as_uint( v0.w ) >> 24;
		// end path on light
		float3 lightColor = (float3)(20);
		float hemiPDF = T4.w;
		float3 D = D4.xyz;
		if (mat == 1 /* light source */)
		{
			float3 potential;
			if (pathState & PATH_LAST_SPECULAR)
			{
				potential = T * lightColor;
			}
			else
			{
				float lightDistance = D4.w;
				float lightArea = 9 * 5;
				float NLdotL = fabs( D.y ); // actually: fabs( dot( D, NL ) );
				float solidAngle = min( TWOPI, lightArea * (1.0f / (lightDistance * lightDistance)) * NLdotL );
				float lightPDF = 1 / solidAngle;
				potential = T * (1 / (lightPDF + hemiPDF)) * lightColor;
			}
			accumulator[pixelIdx] += (float4)(potential, 1);
			continue;
		}
		T *= 1.0f / hemiPDF;
		float3 vert0 = v0.xyz, vert1 = verts[vertIdx + 1].xyz, vert2 = verts[vertIdx + 2].xyz;
		float3 materialColor = rgb32_to_vec3( as_uint( v0.w ) ); // material color
		float3 N = fast_normalize( cross( vert1 - vert0, vert2 - vert0 ) );
		if (dot( N, D ) > 0) N *= -1;
		float3 I = O4.xyz + t * D;
		// handle pure specular BRDF
		if (mat == 2)
		{
			uint newRayIdx = atomic_inc( &extendTasks );
			float3 R = Reflect( D, N );
			raysOut[newRayIdx].T = (float4)(T * materialColor, 1);
			raysOut[newRayIdx].O = (float4)(I + R * EPSILON, as_float( (pixelIdx << 8) + ((depth + 1) << 4) + PATH_LAST_SPECULAR ));
			raysOut[newRayIdx].D = (float4)(R, 1e30f);
			continue;
		}
		// direct illumination: next event estimation
		float r0, r1, r2, r3;
		if (depth == 0 && sampleIdx < 4)
		{
			float2 noise0 = Noise( pixelIdx % rd.height, pixelIdx / rd.height, sampleIdx * 2 );
			float2 noise1 = Noise( pixelIdx % rd.height, pixelIdx / rd.height, sampleIdx * 2 + 1 );
			r0 = noise0.x, r1 = noise0.y;
			r2 = noise0.x, r3 = noise0.y;
		}
		else
		{
			r0 = RandomFloat( &seed ), r1 = RandomFloat( &seed );
			r2 = RandomFloat( &seed ), r3 = RandomFloat( &seed );
		}
		float3 P = (float3)(r0 * 9.0f - 4.5f, 30, r1 * 5.0f - 3.5f);
		float3 L = P - I;
		float NdotL = dot( N, L );
		float3 BRDF = materialColor * INVPI; // lambert BRDF: albedo / pi
		if (NdotL > 0)
		{
			uint newShadowIdx = atomic_inc( &connectTasks );
			float dist2 = dot( L, L ), dist = sqrt( dist2 );
			L *= native_recip( dist );
			float NLdotL = fabs( L.y ); // actually, fabs( dot( L, LN ) )
			shadowOut[newShadowIdx].T = (float4)(lightColor * BRDF * T * NdotL * NLdotL * native_recip( dist2 ), 0);
			shadowOut[newShadowIdx].O = (float4)(I + L * EPSILON, as_float( pixelIdx ));
			shadowOut[newShadowIdx].D = (float4)(L, dist - 2 * EPSILON);
		}
		// indirect illumination: diffuse bounce
		if (depth < 3 && (pathState & PATH_VIA_DIFFUSE) == 0 )
		{
			uint newRayIdx = atomic_inc( &extendTasks );
			float3 R = CosWeightedDiffReflection( N, r2, r3 );
			float PDF = dot( N, R ) * INVPI;
			T *= dot( N, R ) * BRDF;
			raysOut[newRayIdx].T = (float4)(T, PDF /* for MIS, we postpone the pdf until after light sampling */ );
			raysOut[newRayIdx].O = (float4)(I + R * EPSILON, as_float( (pixelIdx << 8) + ((depth + 1) << 4) + PATH_VIA_DIFFUSE ));
			raysOut[newRayIdx].D = (float4)(R, 1e30f);
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
	while (1)
	{
		// obtain task
		if (connectTasks < 1) break;
		const int rayId = atomic_dec( &connectTasks ) - 1;
		if (rayId < 0) break;
		const float4 T4 = shadowIn[rayId].T;
		const float4 O4 = shadowIn[rayId].O;
		const float4 D4 = shadowIn[rayId].D;
		const float3 rD = native_recip( D4.xyz );
		if (!isoccluded_cwbvh( rd.cwbvhNodes, rd.cwbvhTris, O4.xyz, D4.xyz, rD, D4.w ))
		{
			uint pixelIdx = as_uint( O4.w );
			accumulator[pixelIdx] += T4;
		}
	}
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
	int3 rgb = convert_int3( min( sqrt( p.xyz ), (float3)(1.0f, 1.0f, 1.0f) ) * 255.0f );
	pixels[pixelIdx] = (rgb.x << 16) + (rgb.y << 8) + rgb.z;
}