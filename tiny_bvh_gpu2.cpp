// This example shows how to build a GPU path tracer with instancing
// using a TLAS. TinyOCL is used to render on the GPU using OpenCL.

#define FENSTER_APP_IMPLEMENTATION
#define SCRWIDTH 640
#define SCRHEIGHT 400
#include "external/fenster.h" // https://github.com/zserge/fenster

// This application uses tinybvh - And this file will include the implementation.
#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"
using namespace tinybvh;

// This application uses tinyocl - And this file will include the implementation.
#define TINY_OCL_IMPLEMENTATION
#include "tiny_ocl.h"

// Other includes
#include <fstream>

// Application variables

static BVH8_CWBVH bistro, dragon;
static BVHBase* blasList[] = { &bistro, &dragon };
static BVH_GPU tlas;
static BLASInstance instance[17];
static bvhvec4* verts = 0, * dragonVerts = 0;
static int triCount = 0, dragonTriCount = 0, frameIdx = 0, spp = 0;
static Kernel* init, * clear, * rayGen, * extend, * shade;
static Kernel* updateCounters1, * updateCounters2, * traceShadows, * finalize;
static Buffer* pixels, * accumulator, * raysIn, * raysOut, * connections;
static Buffer* bistroNodes = 0, * bistroTris = 0, * bistroVerts, * dragonNodes = 0, * dragonTris = 0, * drVerts, * noise = 0;
static Buffer* tlasNodes = 0, * tlasIndices = 0, * blasInstances = 0;
static size_t computeUnits;
static uint32_t* blueNoise = new uint32_t[128 * 128 * 8];

static bvhvec3 path[] = {
	bvhvec3( 1.136f, -8, -35.723f ),
	bvhvec3( -0.321f, -8, -31.902f ),
	bvhvec3( -2.429f, -8, -27.977f ),
	bvhvec3( -6.053f, -8, -22.313f ),
	bvhvec3( -8.648f, -8, -18.847f ),
	bvhvec3( -11.814f, -8, -14.620f ),
	bvhvec3( -14.837f, -8, -10.581f ),
	bvhvec3( -17.910f, -8, -6.357f ),
	bvhvec3( -18.307f, -8, -0.169f ),
	bvhvec3( -16.152f, -8, 2.698f ),
	bvhvec3( -13.622f, -8, 4.469f ),
	bvhvec3( -8.390f, -8, 6.424f ),
	bvhvec3( -3.329f, -8, 8.316f ),
	bvhvec3( 2.168f, -8, 10.370f ),
	bvhvec3( 7.773f, -8, 12.465f ),
	bvhvec3( 15.531f, -8, 16.273f )
};

// View pyramid for a pinhole camera
struct RenderData
{
	bvhvec4 eye = bvhvec4( 0, 30, 0, 0 ), view = bvhvec4( -1, 0, 0, 0 ), C, p0, p1, p2;
	uint32_t frameIdx, dummy1, dummy2, dummy3;
} rd;

// Scene management - Append a file, with optional position, scale and color override, tinyfied
void AddMesh( const char* file, float scale = 1, bvhvec3 pos = {}, int c = 0, int N = 0 )
{
	std::fstream s{ file, s.binary | s.in }; s.read( (char*)&N, 4 );
	bvhvec4* data = (bvhvec4*)tinybvh::malloc64( (N + triCount) * 48 );
	if (verts) memcpy( data, verts, triCount * 48 ), tinybvh::free64( verts );
	verts = data, s.read( (char*)verts + triCount * 48, N * 48 ), triCount += N;
	for (int* b = (int*)verts + (triCount - N) * 12, i = 0; i < N * 3; i++)
		*(bvhvec3*)b = *(bvhvec3*)b * scale + pos, b[3] = c ? c : b[3], b += 4;
}
void AddQuad( const bvhvec3 pos, const float w, const float d, int c )
{
	bvhvec4* data = (bvhvec4*)tinybvh::malloc64( (triCount + 2) * 48 );
	if (verts) memcpy( data + 6, verts, triCount * 48 ), tinybvh::free64( verts );
	data[0] = bvhvec3( -w, 0, -d ), data[1] = bvhvec3( w, 0, -d );
	data[2] = bvhvec3( w, 0, d ), data[3] = bvhvec3( -w, 0, -d ), verts = data;
	data[4] = bvhvec3( w, 0, d ), data[5] = bvhvec3( -w, 0, d ), triCount += 2;
	for (int i = 0; i < 6; i++) data[i] = 0.5f * data[i] + pos, data[i].w = *(float*)&c;
}

// Blue noise from file
void LoadBlueNoise()
{
	std::fstream s{ "./testdata/blue_noise_128x128x8_2d.raw", s.binary | s.in };
	s.read( (char*)blueNoise, 128 * 128 * 8 * 4 );
}

// Application init
void Init()
{
	// create OpenCL kernels
	init = new Kernel( "wavefront2.cl", "SetRenderData" );
	clear = new Kernel( "wavefront2.cl", "Clear" );
	rayGen = new Kernel( "wavefront2.cl", "Generate" );
	extend = new Kernel( "wavefront2.cl", "Extend" );
	shade = new Kernel( "wavefront2.cl", "Shade" );
	updateCounters1 = new Kernel( "wavefront2.cl", "UpdateCounters1" );
	updateCounters2 = new Kernel( "wavefront2.cl", "UpdateCounters2" );
	traceShadows = new Kernel( "wavefront2.cl", "Connect" );
	finalize = new Kernel( "wavefront2.cl", "Finalize" );

	// we need the 'compute unit' or 'SM' count for wavefront rendering; ask OpenCL for it.
	clGetDeviceInfo( init->GetDeviceID(), CL_DEVICE_MAX_COMPUTE_UNITS, sizeof( size_t ), &computeUnits, NULL );

	// create OpenCL buffers for wavefront path tracing
	int N = SCRWIDTH * SCRHEIGHT;
	pixels = new Buffer( N * sizeof( uint32_t ) );
	accumulator = new Buffer( N * sizeof( bvhvec4 ) );
	raysIn = new Buffer( N * sizeof( bvhvec4 ) * 4 );
	raysOut = new Buffer( N * sizeof( bvhvec4 ) * 4 );
	connections = new Buffer( N * 3 * sizeof( bvhvec4 ) * 3 );
	accumulator = new Buffer( N * sizeof( bvhvec4 ) );
	pixels = new Buffer( N * sizeof( uint32_t ) );
	LoadBlueNoise();
	noise = new Buffer( 128 * 128 * 8 * sizeof( uint32_t ), blueNoise );
	noise->CopyToDevice();

	// load dragon mesh
	AddMesh( "./testdata/dragon.bin", 1, bvhvec3( 0 ) );
	swap( verts, dragonVerts );
	swap( triCount, dragonTriCount );
	dragon.Build( dragonVerts, dragonTriCount );

	// create dragon instances
	for (int x = 0; x < 4; x++) for (int y = 0; y < 4; y++)
	{
		instance[x + y * 4] = BLASInstance( 1 /* the dragon */ );
		BLASInstance& i = instance[x + y * 4];
		i.transform[0] = i.transform[5] = i.transform[10] = 0.25f;
		i.transform[3] = (float)x * 4 - 18;
		i.transform[7] = 14;
		i.transform[11] = (float)y * 4 - 18;
	}

	// load vertex data for static scenery
	AddQuad( bvhvec3( -22, 12, 2 ), 9, 5, 0x1ffffff ); // hard-coded light source
	AddMesh( "./testdata/bistro_ext_part1.bin", 1, bvhvec3( 0 ) );
	AddMesh( "./testdata/bistro_ext_part2.bin", 1, bvhvec3( 0 ) );

	// build bvh (here: 'compressed wide bvh', for efficient GPU rendering)
	bistro.Build( verts, triCount );
	instance[16] = BLASInstance( 0 /* static geometry */ );
	tlas.Build( instance, 17, blasList, 2 );

	// create OpenCL buffers for BVH data
	tlasNodes = new Buffer( tlas.allocatedNodes /* could change! */ * sizeof( BVH_GPU::BVHNode ), tlas.bvhNode );
	tlasIndices = new Buffer( tlas.bvh.idxCount * sizeof( uint32_t ), tlas.bvh.triIdx );
	tlasNodes->CopyToDevice();
	tlasIndices->CopyToDevice();
	blasInstances = new Buffer( 17 * sizeof( BLASInstance ), instance );
	blasInstances->CopyToDevice();
	bistroNodes = new Buffer( bistro.usedBlocks * sizeof( bvhvec4 ), bistro.bvh8Data );
	bistroTris = new Buffer( bistro.idxCount * 3 * sizeof( bvhvec4 ), bistro.bvh8Tris );
	bistroVerts = new Buffer( triCount * 3 * sizeof( bvhvec4 ), verts );
	dragonNodes = new Buffer( dragon.usedBlocks * sizeof( bvhvec4 ), dragon.bvh8Data );
	dragonTris = new Buffer( dragon.idxCount * 3 * sizeof( bvhvec4 ), dragon.bvh8Tris );
	drVerts = new Buffer( dragonTriCount * 3 * sizeof( bvhvec4 ), dragonVerts );
	dragonNodes->CopyToDevice();
	dragonTris->CopyToDevice();
	drVerts->CopyToDevice();
	bistroNodes->CopyToDevice();
	bistroTris->CopyToDevice();
	bistroVerts->CopyToDevice();

	// load camera position / direction from file
	std::fstream t = std::fstream{ "camera_gpu.bin", t.binary | t.in };
	if (!t.is_open()) return;
	t.read( (char*)&rd, sizeof( rd ) );
}

// Keyboard handling
bool UpdateCamera( float delta_time_s, fenster& f )
{
	bvhvec3 right = normalize( cross( bvhvec3( 0, 1, 0 ), rd.view ) ), up = 0.8f * cross( rd.view, right );
	// emit camera pos
	static bool pdown = false;
	if (!GetAsyncKeyState( 'P' )) pdown = false; else if (!pdown)
	{
		FILE* f = fopen( "path.txt", "a" );
		fprintf( f, "bvhvec3( %.3ff, %.3ff, %.3ff ),\n", rd.eye.x, rd.eye.y, rd.eye.z );
		printf( "point emitted.\n" );
		fclose( f );
		pdown = true;
	}
	// get camera controls
	float moved = 0, spd = 10.0f * delta_time_s;
	if (f.keys['A'] || f.keys['D']) rd.eye += right * (f.keys['D'] ? spd : -spd), moved = 1;
	if (f.keys['W'] || f.keys['S']) rd.eye += rd.view * (f.keys['W'] ? spd : -spd), moved = 1;
	if (f.keys['R'] || f.keys['F']) rd.eye += up * 2.0f * (f.keys['R'] ? spd : -spd), moved = 1;
	if (f.keys[20]) rd.view = normalize( rd.view + right * -0.1f * spd ), moved = 1;
	if (f.keys[19]) rd.view = normalize( rd.view + right * 0.1f * spd ), moved = 1;
	if (f.keys[17]) rd.view = normalize( rd.view + up * -0.1f * spd ), moved = 1;
	if (f.keys[18]) rd.view = normalize( rd.view + up * 0.1f * spd ), moved = 1;
	// recalculate right, up
	right = normalize( cross( bvhvec3( 0, 1, 0 ), rd.view ) ), up = 0.8f * cross( rd.view, right );
	bvhvec3 C = rd.eye + 1.2f * rd.view;
	rd.p0 = C - right + up, rd.p1 = C + right + up, rd.p2 = C - right - up;
	return moved > 0;
}

// Application Tick
void Tick( float delta_time_s, fenster& f, uint32_t* buf )
{
	// handle user input and update camera
	int N = SCRWIDTH * SCRHEIGHT;
	if (UpdateCamera( delta_time_s, f ) || frameIdx++ == 0)
	{
		clear->SetArguments( accumulator );
		clear->Run( N );
		spp = 1;
	}
	// wavefront step 0: render on the GPU
	init->SetArguments( N, rd.eye, rd.p0, rd.p1, rd.p2,
		frameIdx, SCRWIDTH, SCRHEIGHT,
		bistroNodes, bistroTris, bistroVerts, dragonNodes, dragonTris, drVerts,
		tlasNodes, tlasIndices, blasInstances,
		noise
	);
	init->Run( 1 ); // init atomic counters, set buffer ptrs etc.
	rayGen->SetArguments( raysOut, spp * 19191 );
	rayGen->Run2D( oclint2( SCRWIDTH, SCRHEIGHT ) );
	for (int i = 0; i < 3; i++)
	{
		swap( raysOut, raysIn );
		extend->SetArguments( raysIn );
		extend->Run( computeUnits * 64 * 16, 64 );
		updateCounters1->Run( 1 );
		shade->SetArguments( accumulator, raysIn, raysOut, connections, spp - 1 );
		shade->Run( computeUnits * 64 * 16, 64 );
		updateCounters2->Run( 1 );
	}
	traceShadows->SetArguments( accumulator, connections );
	traceShadows->Run( computeUnits * 64 * 8, 64 );
	finalize->SetArguments( accumulator, 1.0f / (float)spp++, pixels );
	finalize->Run2D( oclint2( SCRWIDTH, SCRHEIGHT ) );
	pixels->CopyFromDevice();
	memcpy( buf, pixels->GetHostPtr(), N * sizeof( uint32_t ) );
	// print frame time / rate in window title
	char title[50];
	sprintf( title, "tiny_bvh %.2f s %.2f Hz", delta_time_s, 1.0f / delta_time_s );
	fenster_update_title( &f, title );
}

// Application Shutdown
void Shutdown()
{
	// save camera position / direction to file
	std::fstream s = std::fstream{ "camera_gpu.bin", s.binary | s.out };
	s.write( (char*)&rd, sizeof( rd ) );
}