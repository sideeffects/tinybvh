#define FENSTER_APP_IMPLEMENTATION
#define SCRWIDTH 800
#define SCRHEIGHT 600
#define TILESIZE 20
#include "external/fenster.h" // https://github.com/zserge/fenster

#define GRIDSIZE 45
#define INSTCOUNT (GRIDSIZE * GRIDSIZE * GRIDSIZE)

// #define DOUBLE_PRECISION_TEST

#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"
#include <fstream>
#include <thread>

using namespace tinybvh;

BVH sponza, obj, tlas;
BVHBase* bvhList[] = { &sponza, &obj };
BLASInstance inst[INSTCOUNT + 1 /* one extra for sponza */];
int frameIdx = 0, verts = 0, bverts = 0;
bvhvec4* triangles = 0;
bvhvec4* bunny = 0;
static std::atomic<int> tileIdx( 0 );
static unsigned threadCount = std::thread::hardware_concurrency();

// setup view pyramid for a pinhole camera
static bvhvec3 eye( -15.24f, 21.5f, 2.54f ), p1, p2, p3;
static bvhvec3 view = normalize( bvhvec3( 0.826f, -0.438f, -0.356f ) );

// double-precision test
bvhdbl3* trianglesEx = 0;
bvhdbl3* bunnyEx = 0;
BLASInstanceEx instEx[INSTCOUNT + 1];
BVH_Double bvhEx, blasEx, tlasEx;
BVH_Double* bvhExList[] = { &bvhEx, &blasEx };

void Init()
{
	uint32_t test = sizeof( tinybvh::Ray );

	// load raw vertex data for Crytek's Sponza
	std::fstream s{ "./testdata/cryteksponza.bin", s.binary | s.in };
	s.read( (char*)&verts, 4 );
	printf( "Loading triangle data (%i tris).\n", verts );
	verts *= 3, triangles = (bvhvec4*)malloc64( verts * 16 );
	s.read( (char*)triangles, verts * 16 );
	sponza.Build( triangles, verts / 3 );

	// load bunny
	std::fstream b{ "./testdata/bunny.bin", s.binary | s.in };
	b.read( (char*)&bverts, 4 );
	bverts *= 3, bunny = (bvhvec4*)malloc64( bverts * 16 );
	b.read( (char*)bunny, verts * 16 );
	obj.Build( bunny, bverts / 3 );

	// build a TLAS
	inst[0] = BLASInstance( 0 /* sponza */ );
	for (int b = 0, x = 0; x < GRIDSIZE; x++) for (int y = 0; y < GRIDSIZE; y++) for (int z = 0; z < GRIDSIZE; z++, b++)
	{
		inst[b] = BLASInstance( 1 /* bunny */ );
		inst[b].transform[0] = inst[b].transform[5] = inst[b].transform[10] = 0.02f; // scale
		inst[b].transform[3] = (float)x * 0.2f - GRIDSIZE * 0.1f;
		inst[b].transform[7] = (float)y * 0.2f - GRIDSIZE * 0.1f + 7;
		inst[b].transform[11] = (float)z * 0.2f - GRIDSIZE * 0.1f - 1;
	}
	tlas.Build( inst, 1 + INSTCOUNT, bvhList, 2 );

	// convert data to doubles
	trianglesEx = (bvhdbl3*)malloc64( verts * sizeof( bvhdbl3 ) );
	for (int i = 0; i < verts; i++) trianglesEx[i] = bvhdbl3( triangles[i] );
	bunnyEx = (bvhdbl3*)malloc64( bverts * sizeof( bvhdbl3 ) );
	for (int i = 0; i < bverts; i++) bunnyEx[i] = bvhdbl3( bunny[i] );

	// build double-precision TLAS
	bvhEx.Build( trianglesEx, verts / 3 );
	blasEx.Build( bunnyEx, bverts / 3 );
	instEx[0] = BLASInstanceEx( 0 );
	for (int b = 0, x = 0; x < GRIDSIZE; x++) for (int y = 0; y < GRIDSIZE; y++) for (int z = 0; z < GRIDSIZE; z++, b++)
	{
		instEx[b] = BLASInstanceEx( 1 /* bunny */ );
		instEx[b].transform[0] = instEx[b].transform[5] = instEx[b].transform[10] = 0.025; // scale
		instEx[b].transform[3] = (double)x * 0.3 - GRIDSIZE * 0.15;
		instEx[b].transform[7] = (double)y * 0.3 - GRIDSIZE * 0.15 + 5;
		instEx[b].transform[11] = (double)z * 0.3 - GRIDSIZE * 0.15;
	}
	tlasEx.Build( instEx, 1 + INSTCOUNT, bvhExList, 2 );
}

bool UpdateCamera( float delta_time_s, fenster& f )
{
	bvhvec3 right = normalize( cross( bvhvec3( 0, 1, 0 ), view ) ), up = 0.8f * cross( view, right );
	float moved = 0, spd = 10.0f * delta_time_s;
	if (f.keys['A'] || f.keys['D']) eye += right * (f.keys['D'] ? spd : -spd), moved = 1;
	if (f.keys['W'] || f.keys['S']) eye += view * (f.keys['W'] ? spd : -spd), moved = 1;
	if (f.keys['R'] || f.keys['F']) eye += up * 2.0f * (f.keys['R'] ? spd : -spd), moved = 1;
	if (f.keys[20]) view = normalize( view + right * -0.1f * spd ), moved = 1;
	if (f.keys[19]) view = normalize( view + right * 0.1f * spd ), moved = 1;
	if (f.keys[17]) view = normalize( view + up * -0.1f * spd ), moved = 1;
	if (f.keys[18]) view = normalize( view + up * 0.1f * spd ), moved = 1;
	// recalculate right, up
	right = normalize( cross( bvhvec3( 0, 1, 0 ), view ) ), up = 0.8f * cross( view, right );
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
		const bvhvec3 L = normalize( bvhvec3( 1, 2, 3 ) );
		for (int y = 0; y < TILESIZE; y++) for (int x = 0; x < TILESIZE; x++)
		{
			const int pixel_x = tx * TILESIZE + x, pixel_y = ty * TILESIZE + y;
			const int pixelIdx = pixel_x + pixel_y * SCRWIDTH;
			// setup primary ray
			const float u = (float)pixel_x / SCRWIDTH, v = (float)pixel_y / SCRHEIGHT;
			const bvhvec3 D = normalize( p1 + u * (p2 - p1) + v * (p3 - p1) - eye );
		#ifdef DOUBLE_PRECISION_TEST
			RayEx ray( eye, D, 1e30f );
			tlasEx.Intersect( ray );
		#else
			Ray ray( eye, D, 1e30f );
			tlas.Intersect( ray );
		#endif
			if (ray.hit.t < 10000)
			{
				uint32_t pixel_x = tx * 4 + x, pixel_y = ty * 4 + y;
			#ifdef DOUBLE_PRECISION_TEST
				uint64_t primIdx = ray.hit.prim;
				uint64_t instIdx = ray.hit.inst;
			#else
			#if TLAS_BITS == 32
				// instance and primitive index are stored in separate fields
				uint32_t primIdx = ray.hit.prim;
				uint32_t instIdx = ray.hit.inst;
			#else
				// instance and primitive index are stored together for compactness
				uint32_t primIdx = ray.hit.prim & PRIM_IDX_MASK;
				uint32_t instIdx = (uint32_t)ray.hit.prim >> INST_IDX_SHFT;
			#endif
			#endif
				BVH* blas = (BVH*)tlas.blasList[inst[instIdx].blasIdx];
				bvhvec4slice& instTris = blas->verts;
				bvhvec3 v0 = instTris[primIdx * 3];
				bvhvec3 v1 = instTris[primIdx * 3 + 1];
				bvhvec3 v2 = instTris[primIdx * 3 + 2];
				bvhvec3 N = normalize( cross( v1 - v0, v2 - v0 ) ); // TODO: Transform to world space
				int c = (int)(255.9f * fabs( dot( N, L ) ));
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

	// update TLAS
	// tlas.Build( inst, 3, bvhList, 2 );			// regular
	// tlasEx.Build( instEx, 3, bvhExList, 2 );	// double-precision

	// render tiles
	tileIdx = threadCount;
#ifdef _DEBUG
	for (uint32_t i = 0; i < threadCount; i++) TraceWorkerThread( buf, i );
#else
	std::vector<std::thread> threads;
	for (uint32_t i = 0; i < threadCount; i++)
		threads.emplace_back( &TraceWorkerThread, buf, i );
	for (auto& thread : threads) thread.join();
#endif

	// change instance transforms
	static float a[3] = { 0 };
	for (int i = 1; i < 3; i++)
	{
		inst[i].transform[7] /* y-pos */ = sinf( a[i] ) * 3.0f + 3.5f;
		instEx[i].transform[7] = sin( (double)a[i] ) * 3 + 3.5;
		a[i] += 0.1f + (0.01f * (float)i);
		if (a[i] > 6.2832f) a[i] -= 6.2832f;
	}
}

void Shutdown() { /* nothing here. */ }