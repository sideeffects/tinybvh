#define FENSTER_APP_IMPLEMENTATION
#define SCRWIDTH 800
#define SCRHEIGHT 600
#define TILESIZE 20
#include "external/fenster.h" // https://github.com/zserge/fenster

#define GRIDSIZE 45
#define INSTCOUNT (GRIDSIZE * GRIDSIZE * GRIDSIZE)

#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"
#include <fstream>
#include <thread>

using namespace tinybvh;

BVH_Double sponza, obj, tlas;
BVH_Double* bvhList[] = { &sponza, &obj };
BLASInstanceEx inst[INSTCOUNT + 1 /* one extra for sponza */];
int frameIdx = 0, verts = 0, bverts = 0;
bvhvec4* triangles = 0, * bunny = 0;
bvhdbl3* trianglesEx = 0, * bunnyEx = 0;
static std::atomic<int> tileIdx( 0 );
static unsigned threadCount = std::thread::hardware_concurrency();

// setup view pyramid for a pinhole camera
static bvhvec3 eye( -15.24f, 21.5f, 2.54f ), p1, p2, p3;
static bvhvec3 view = tinybvh_normalize( bvhvec3( 0.826f, -0.438f, -0.356f ) );

void Init()
{
	// load raw vertex data for Crytek's Sponza
	std::fstream s{ "./testdata/cryteksponza.bin", s.binary | s.in };
	s.read( (char*)&verts, 4 );
	printf( "Loading triangle data (%i tris).\n", verts );
	verts *= 3, triangles = (bvhvec4*)malloc64( verts * sizeof( bvhvec4 ) );
	s.read( (char*)triangles, verts * 16 );
	trianglesEx = (bvhdbl3*)malloc64( verts * sizeof( bvhdbl3 ) );
	for (int i = 0; i < verts; i++) trianglesEx[i] = bvhdbl3( triangles[i] );
	sponza.Build( trianglesEx, verts / 3 );

	// load bunny
	std::fstream b{ "./testdata/bunny.bin", s.binary | s.in };
	b.read( (char*)&bverts, 4 );
	bverts *= 3, bunny = (bvhvec4*)malloc64( bverts * sizeof( bvhvec4 ) );
	b.read( (char*)bunny, verts * 16 );
	bunnyEx = (bvhdbl3*)malloc64( bverts * sizeof( bvhdbl3 ) );
	for (int i = 0; i < bverts; i++) bunnyEx[i] = bvhdbl3( bunny[i] );
	obj.Build( bunnyEx, bverts / 3 );

	// build a TLAS
	inst[0] = BLASInstanceEx( 0 /* sponza */ );
	for (int b = 1, x = 0; x < GRIDSIZE; x++) for (int y = 0; y < GRIDSIZE; y++) for (int z = 0; z < GRIDSIZE; z++, b++)
	{
		inst[b] = BLASInstanceEx( 1 /* bunny */ );
		inst[b].transform[0] = inst[b].transform[5] = inst[b].transform[10] = 0.02; // scale
		inst[b].transform[3] = (float)x * 0.2 - GRIDSIZE * 0.1;
		inst[b].transform[7] = (float)y * 0.2 - GRIDSIZE * 0.1 + 7;
		inst[b].transform[11] = (float)z * 0.2 - GRIDSIZE * 0.1 - 1;
	}
	tlas.Build( inst, 1 + INSTCOUNT, bvhList, 2 ); // just move build to Tick if instance transforms are not static.
}

bool UpdateCamera( float delta_time_s, fenster& f )
{
	bvhvec3 right = tinybvh_normalize( tinybvh_cross( bvhvec3( 0, 1, 0 ), view ) ), up = 0.8f * tinybvh_cross( view, right );
	float moved = 0, spd = 10.0f * delta_time_s;
	if (f.keys['A'] || f.keys['D']) eye += right * (f.keys['D'] ? spd : -spd), moved = 1;
	if (f.keys['W'] || f.keys['S']) eye += view * (f.keys['W'] ? spd : -spd), moved = 1;
	if (f.keys['R'] || f.keys['F']) eye += up * 2.0f * (f.keys['R'] ? spd : -spd), moved = 1;
	if (f.keys[20]) view = tinybvh_normalize( view + right * -0.1f * spd ), moved = 1;
	if (f.keys[19]) view = tinybvh_normalize( view + right * 0.1f * spd ), moved = 1;
	if (f.keys[17]) view = tinybvh_normalize( view + up * -0.1f * spd ), moved = 1;
	if (f.keys[18]) view = tinybvh_normalize( view + up * 0.1f * spd ), moved = 1;
	// recalculate right, up
	right = tinybvh_normalize( tinybvh_cross( bvhvec3( 0, 1, 0 ), view ) ), up = 0.8f * tinybvh_cross( view, right );
	bvhvec3 C = eye + 1.2f * view;
	p1 = C - right + up, p2 = C + right + up, p3 = C - right - up;
	return moved > 0;
}

void TraceWorkerThread( uint32_t* buf, int threadIdx )
{
	const int xtiles = SCRWIDTH / TILESIZE, ytiles = SCRHEIGHT / TILESIZE;
	const int tiles = xtiles * ytiles;
	int tile = threadIdx;
	while (tile < tiles)
	{
		const int tx = tile % xtiles, ty = tile / xtiles;
		unsigned seed = (tile + 17) * 171717 + frameIdx * 1023;
		const bvhvec3 L = tinybvh_normalize( bvhvec3( 1, 2, 3 ) );
		for (int y = 0; y < TILESIZE; y++) for (int x = 0; x < TILESIZE; x++)
		{
			const int pixel_x = tx * TILESIZE + x, pixel_y = ty * TILESIZE + y;
			const int pixelIdx = pixel_x + pixel_y * SCRWIDTH;
			// setup primary ray
			const float u = (float)pixel_x / SCRWIDTH, v = (float)pixel_y / SCRHEIGHT;
			const bvhvec3 D = tinybvh_normalize( p1 + u * (p2 - p1) + v * (p3 - p1) - eye );
			RayEx ray( eye, D, 1e30f );
			tlas.Intersect( ray );
			if (ray.hit.t < 10000)
			{
				uint32_t pixel_x = tx * 4 + x, pixel_y = ty * 4 + y;
				// instance and primitive index are stored in separate fields
				uint64_t primIdx = ray.hit.prim;
				uint64_t instIdx = ray.hit.inst;
				BVH_Double* blas = (BVH_Double*)tlas.blasList[inst[instIdx].blasIdx];
				bvhdbl3 v0 = blas->verts[primIdx * 3];
				bvhdbl3 v1 = blas->verts[primIdx * 3 + 1];
				bvhdbl3 v2 = blas->verts[primIdx * 3 + 2];
				bvhdbl3 N = tinybvh_normalize( tinybvh_cross( v1 - v0, v2 - v0 ) ); // TODO: Transform to world space
				int c = (int)(255.9 * fabs( tinybvh_dot( N, L ) ));
				buf[pixelIdx] = c + (c << 8) + (c << 16);
			}
		}
		tile = tileIdx++;
	}
}

void Tick( float delta_time_s, fenster& f, uint32_t* buf )
{
	// handle user input and update camera
	bool moved = UpdateCamera( delta_time_s, f ) || frameIdx++ == 0;

	// clear the screen with a debug-friendly color
	for (int i = 0; i < SCRWIDTH * SCRHEIGHT; i++) buf[i] = 0xaaaaff;

	// render tiles
	tileIdx = threadCount;
	std::vector<std::thread> threads;
	for (uint32_t i = 0; i < threadCount; i++)
		threads.emplace_back( &TraceWorkerThread, buf, i );
	for (auto& thread : threads) thread.join();
}

void Shutdown() { /* nothing here. */ }