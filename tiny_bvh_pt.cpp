#define FENSTER_APP_IMPLEMENTATION
#define SCRWIDTH 800
#define SCRHEIGHT 600
#include "external/fenster.h" // https://github.com/zserge/fenster

#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"
#include <fstream>

using namespace tinybvh;

// Application variables
static BVH bvh;
static bvhvec4* triangles = 0;
static int verts = 0, frameIdx = 0, spp = 0;
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
	return normalize( dot( R, N ) < 0 ? R : (R * -1.0f ) );
}

// Application init
void Init()
{
	// load raw vertex data for Crytek's Sponza
	std::fstream s{ "./testdata/cryteksponza.bin", s.binary | s.in };
	s.seekp( 0 );
	s.read( (char*)&verts, 4 );
	printf( "Loading triangle data (%i tris).\n", verts );
	verts *= 3, triangles = (bvhvec4*)malloc64( verts * 16 );
	s.read( (char*)triangles, verts * 16 );
	s.close();
	bvh.BuildAVX( triangles, verts / 3 );
	// load camera position / direction from file
	std::fstream t = std::fstream{ "camera.bin", t.binary | t.in };
	if (!t.is_open()) return;
	t.seekp( 0 );
	t.read( (char*)&eye, sizeof( eye ) );
	t.read( (char*)&view, sizeof( view ) );
	t.close();
}

// Keyboard handling
bool UpdateCamera(float delta_time_s, fenster& f)
{
	bvhvec3 right = normalize( cross( bvhvec3( 0, 1, 0 ), view ) ), up = 0.8f * cross( view, right );
	// get camera controls.
	bool moved = false;
	if (f.keys['A']) eye += right * -1.0f * delta_time_s * 100, moved = true;
	if (f.keys['D']) eye += right * delta_time_s * 100, moved = true;
	if (f.keys['W']) eye += view * delta_time_s * 100, moved = true;
	if (f.keys['S']) eye += view * -1.0f * delta_time_s * 100, moved = true;
	if (f.keys['R']) eye += up * delta_time_s * 200, moved = true;
	if (f.keys['F']) eye += up * -1.0f * delta_time_s * 200, moved = true;
	if (f.keys[20]) view = normalize( view + right * -1.0f * delta_time_s ), moved = true;
	if (f.keys[19]) view = normalize( view + right * delta_time_s ), moved = true;
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
	bvh.Intersect( ray );
	if (ray.hit.t == 1e30f) return bvhvec3( 0 ); // hit nothing
	bvhvec3 I = ray.O + ray.hit.t * ray.D;
	// get normal at intersection point
	unsigned primIdx = ray.hit.prim;
	bvhvec3 v0 = triangles[primIdx * 3 + 0];
	bvhvec3 v1 = triangles[primIdx * 3 + 1];
	bvhvec3 v2 = triangles[primIdx * 3 + 2];
	bvhvec3 N = normalize( cross( v1 - v0, v0 - v2 ) );
	// shoot AO ray
	bvhvec3 R = DiffuseReflection( N, seed );
	Ray aoRay( I + R * 0.01f, R, 100 );
	bvh.Intersect( aoRay );
	return bvhvec3( aoRay.hit.t / 100 );
}

// Application Tick
void Tick(float delta_time_s, fenster & f, uint32_t* buf)
{
	// handle user input and update camera
	if (frameIdx++ == 0 || UpdateCamera(delta_time_s, f))
	{
		memset( accumulator, 0, 800 * 600 * sizeof( bvhvec3 ) );
		spp = 1;
	}

	// render tiles
	const int xtiles = SCRWIDTH / 4, ytiles = SCRHEIGHT / 4, tiles = xtiles * ytiles;
	const float scale = 1.0f / spp;
#pragma omp parallel for schedule(dynamic)
	for( int tile = 0; tile < tiles; tile++ )
	{
		const int tx = tile % xtiles, ty = tile / xtiles;
		unsigned seed = (tile + 17) * 171717 + frameIdx * 1023;
		for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++)
		{
			const int pixel_x = tx * 4 + x;
			const int pixel_y = ty * 4 + y;
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
	s.seekp( 0 );
	s.write( (char*)&eye, sizeof( eye ) );
	s.write( (char*)&view, sizeof( view ) );
	s.close();
}