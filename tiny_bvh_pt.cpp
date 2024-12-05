#define FENSTER_APP_IMPLEMENTATION
#define SCRWIDTH 800
#define SCRHEIGHT 600
#define TILESIZE 8
#include "external/fenster.h" // https://github.com/zserge/fenster

#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"
#include <fstream>

using namespace tinybvh;

// Application variables
static BVH bvh;
static bvhvec4* tris = 0;
static int triCount = 0, frameIdx = 0, spp = 0;
static bvhvec3 accumulator[SCRWIDTH * SCRHEIGHT];

// Setup view pyramid for a pinhole camera: 
// eye, p1 (top-left), p2 (top-right) and p3 (bottom-left)
static bvhvec3 eye( 0, 30, 0 ), p1, p2, p3;
static bvhvec3 view = normalize( bvhvec3( -1, 0, 0 ) );

// Xor32 RNG
static unsigned RandomUInt( unsigned& seed ) { seed ^= seed << 13, seed ^= seed >> 17, seed ^= seed << 5; return seed; }
static float RandomFloat( unsigned& seed ) { return RandomUInt( seed ) * 2.3283064365387e-10f; }

// Ray tracing math
bvhvec3 DiffuseReflection( const bvhvec3 N, unsigned& seed )
{
	bvhvec3 R;
	do
	{
		R = bvhvec3( RandomFloat( seed ) * 2 - 1, RandomFloat( seed ) * 2 - 1, RandomFloat( seed ) * 2 - 1 );
	} while (dot( R, R ) > 1);
	return normalize( dot( R, N ) < 0 ? R : (R * -1.0f) );
}

// Color conversion
bvhvec3 rgb32_to_vec3( const unsigned c )
{
	return bvhvec3( (float)(c >> 16), (float)((c >> 8) & 255), (float)(c & 255) ) * (1 / 255.f);
}

// Scene management - Append a file, with optional position, scale and color override, tinyfied
void AddMesh( const char* file, float scale = 1, bvhvec3 pos = {}, int c = 0, int N = 0 )
{
	std::fstream s{ file, s.binary | s.in };
	s.read( (char*)&N, 4 );
	bvhvec4* data = (bvhvec4*)malloc64( (N + triCount) * 48 );
	if (tris) memcpy( data, tris, triCount * 48 ), free64( tris );
	tris = data, s.read( (char*)tris + triCount * 48, N * 48 ), triCount += N;
	for (int* b = (int*)tris + (triCount - N) * 12, i = 0; i < N * 3; i++)
		*(bvhvec3*)b = *(bvhvec3*)b * scale + pos, b[3] = c ? c : b[3], b += 4;
}

// Application init
void Init()
{
	// load raw vertex data
	AddMesh( "./testdata/cryteksponza.bin", 1, bvhvec3( 0 ), 0xffffff );
	AddMesh( "./testdata/dragon.bin", 1.1f, bvhvec3( 29, 3.01f, 0 ), 0xffbb88 );
	AddMesh( "./testdata/lucy.bin", 1.1f, bvhvec3( -2, 4.1f, -3 ), 0xaaaaff );
	AddMesh( "./testdata/bunny.bin", 0.2f, bvhvec3( -7, 0.13f, 0 ), 0x333333 );
	AddMesh( "./testdata/legocar.bin", 0.3f, bvhvec3( -12, 0.8f, -5 ) );
	AddMesh( "./testdata/armadillo.bin", 0.3f, bvhvec3( 7, 1, 3 ), 0xff2020 );
	AddMesh( "./testdata/xyzrgb_dragon.bin", 0.5f, bvhvec3( -22, 0.95f, 0 ), 0xffffaa );
	AddMesh( "./testdata/suzanne.bin", 0.2f, bvhvec3( -18, 0.95f, -16 ), 0x90ff90 );
	AddMesh( "./testdata/head.bin", 0.5f, bvhvec3( 0, 3, 9 ) );
	// build bvh
	bvh.BuildAVX( tris, triCount );
#if defined BVH_USEAVX || defined BVH_USENEON
	bvh.Convert( BVH::WALD_32BYTE, BVH::BASIC_BVH4 );
	bvh.Convert( BVH::BASIC_BVH4, BVH::BVH4_AFRA );
#endif
	// load camera position / direction from file
	std::fstream t = std::fstream{ "camera.bin", t.binary | t.in };
	if (!t.is_open()) return;
	t.read( (char*)&eye, sizeof( eye ) );
	t.read( (char*)&view, sizeof( view ) );
	t.close();
}

// Keyboard handling
bool UpdateCamera( float delta_time_s, fenster& f )
{
	bvhvec3 right = normalize( cross( bvhvec3( 0, 1, 0 ), view ) ), up = 0.8f * cross( view, right );
	// get camera controls.
	bool moved = false;
	if (f.keys['A']) eye += right * -1.0f * delta_time_s * 10, moved = true;
	if (f.keys['D']) eye += right * delta_time_s * 10, moved = true;
	if (f.keys['W']) eye += view * delta_time_s * 10, moved = true;
	if (f.keys['S']) eye += view * -1.0f * delta_time_s * 10, moved = true;
	if (f.keys['R']) eye += up * delta_time_s * 20, moved = true;
	if (f.keys['F']) eye += up * -1.0f * delta_time_s * 20, moved = true;
	if (f.keys[20]) view = normalize( view + right * -1.0f * delta_time_s ), moved = true;
	if (f.keys[19]) view = normalize( view + right * delta_time_s ), moved = true;
	if (f.keys[17]) view = normalize( view + up * -1.0f * delta_time_s ), moved = true;
	if (f.keys[18]) view = normalize( view + up * delta_time_s ), moved = true;
	// recalculate right, up
	right = normalize( cross( bvhvec3( 0, 1, 0 ), view ) ), up = 0.8f * cross( view, right );
	bvhvec3 C = eye + 1.2f * view;
	p1 = C - right + up, p2 = C + right + up, p3 = C - right - up;
	return moved;
}

// Light transport calculation - Ambient Occlusion
bvhvec3 Trace( Ray& ray, unsigned& seed )
{
	// find primary intersection
	bvh.Intersect( ray, bvh.bvh4Alt2 ? BVH::BVH4_AFRA : BVH::WALD_32BYTE );
	if (ray.hit.t == 1e30f) return bvhvec3( 0.6f, 0.7f, 2 ); // hit nothing
	bvhvec3 I = ray.O + ray.hit.t * ray.D;
	// get normal at intersection point
	unsigned primIdx = ray.hit.prim;
	bvhvec3 v0 = tris[primIdx * 3 + 0];
	bvhvec3 v1 = tris[primIdx * 3 + 1];
	bvhvec3 v2 = tris[primIdx * 3 + 2];
	bvhvec3 N = normalize( cross( v1 - v0, v0 - v2 ) );
	// shoot AO rays
	float total = 0;
	for (int i = 0; i < 4; i++)
	{
		bvhvec3 R = DiffuseReflection( N, seed );
		Ray aoRay( I + R * 0.001f, R, 10 );
		bvh.Intersect( aoRay, bvh.bvh4Alt2 ? BVH::BVH4_AFRA : BVH::WALD_32BYTE );
		total += aoRay.hit.t;
	}
	unsigned triColor = *(unsigned*)&tris[primIdx * 3 + 0].w;
	return rgb32_to_vec3( triColor ) * bvhvec3( total / 40 );
}

// Application Tick
void Tick( float delta_time_s, fenster& f, uint32_t* buf )
{
	// handle user input and update camera
	if (frameIdx++ == 0 || UpdateCamera( delta_time_s, f ))
	{
		memset( accumulator, 0, SCRWIDTH * SCRHEIGHT * sizeof( bvhvec3 ) );
		spp = 1;
	}

	// render tiles
	const int xtiles = SCRWIDTH / TILESIZE, ytiles = SCRHEIGHT / TILESIZE;
	const int tiles = xtiles * ytiles;
	const float scale = 1.0f / spp;
#pragma omp parallel for schedule(dynamic)
	for (int tile = 0; tile < tiles; tile++)
	{
		const int tx = tile % xtiles, ty = tile / xtiles;
		unsigned seed = (tile + 17) * 171717 + frameIdx * 1023;
		for (int y = 0; y < TILESIZE; y++) for (int x = 0; x < TILESIZE; x++)
		{
			const int pixel_x = tx * TILESIZE + x;
			const int pixel_y = ty * TILESIZE + y;
			// setup primary ray
			const float u = (float)pixel_x / SCRWIDTH, v = (float)pixel_y / SCRHEIGHT;
			const bvhvec3 D = normalize( p1 + u * (p2 - p1) + v * (p3 - p1) - eye );
			Ray primaryRay( eye, D );
			// trace
			const unsigned pixelIdx = pixel_x + pixel_y * SCRWIDTH;
			accumulator[pixelIdx] += Trace( primaryRay, seed );
			const bvhvec3 E = accumulator[pixelIdx] * scale;
			// visualize, with a poor man's gamma correct
			const int r = (int)tinybvh_min( 255.0f, sqrtf( E.x ) * 255.0f );
			const int g = (int)tinybvh_min( 255.0f, sqrtf( E.y ) * 255.0f );
			const int b = (int)tinybvh_min( 255.0f, sqrtf( E.z ) * 255.0f );
			buf[pixel_x + pixel_y * SCRWIDTH] = b + (g << 8) + (r << 16);
		}
	}
	spp++;
}

// Application Shutdown
void Shutdown()
{
	// save camera position / direction to file
	std::fstream s = std::fstream{ "camera.bin", s.binary | s.out };
	s.write( (char*)&eye, sizeof( eye ) );
	s.write( (char*)&view, sizeof( view ) );
	s.close();
}