
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
using namespace std;
#include <stdio.h>
//#include <glm/glm.hpp>
//using namespace glm;
#include "cutil_math.h"
#include <curand.h>
#include <curand_kernel.h>

#define M_PI 3.1415926;
const int spp = 1;

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA error: %s: %s. \n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

// "__host__": This function called by CPU and runs on CPU
// "__device__": This function called by GPU and runs on GPU (inside one thread)
// "__global__": This is a kernel function, called by CPU and runs on GPU

struct Ray {
	float3 origin;
	float3 direction;
	// create a ray
	__device__ Ray(float3 o_, float3 d_) : origin(o_), direction(d_) {}
};

// reflection type (DIFFuse, SPECular, REFRactive)
enum Refl_t { DIFF, SPEC, REFR };

struct Sphere {

	float radius;
	float3 position, emission, color;	// color may not use in the future (emission only)
	Refl_t reflectType;	//DIFF, SPEC, REFR
	//__device__ Sphere(float rad_, float3 p_, float3 e_, float3 c_, Refl_t refl_) :
	//	radius(rad_), position(p_), emission(e_), color(c_), reflectType(refl_) {}
	__device__ float intersect(const Ray& r) const { // returns distance, 0 if nohit 

		// Ray/sphere intersection
		// Quadratic formula required to solve ax^2 + bx + c = 0 
		// Solution x = (-b +- sqrt(b*b - 4ac)) / 2a
		// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 

		float3 op = position - r.origin;
		float t, epsilon = 0.01f;
		float b = dot(op, r.direction);
		float disc = b * b - dot(op, op) + radius * radius; // discriminant
		if (disc < 0) return 0; else disc = sqrtf(disc);
		return (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0);
	}
};

__constant__ Sphere spheres[] = {
	/* cornell box
	{radius		position						emission				color					reflectType}*/
	{1e5f,		{1e5f + 1.0f, 40.8f, 81.6f},	{0.0f, 0.0f, 0.0f},		{0.75f, 0.25f, 0.25f},	DIFF},//Left 
	{1e5f,		{-1e5f + 99.0f, 40.8f, 81.6f},	{0.0f, 0.0f, 0.0f},		{0.25f, 0.25f, 0.75f},	DIFF},//Rght 
	{1e5f,		{50.0f, 40.8f, 1e5f},			{0.0f, 0.0f, 0.0f},		{0.75f, 0.75f, 0.75f},	DIFF},//Back 
	{1e5f,		{50.0f, 40.8f, -1e5f + 170.0f},	{0.0f, 0.0f, 0.0f},		{0.0f, 0.0f, 0.0f},		DIFF},//Frnt 
	{1e5f,		{50.0f, 1e5f, 81.6f},			{0.0f, 0.0f, 0.0f},		{0.75f, 0.75f, 0.75f},	DIFF},//Botm 
	{1e5f,		{50.0f, -1e5f + 81.6f, 81.6f},	{0.0f, 0.0f, 0.0f},		{0.75f, 0.75f, 0.75f},	DIFF},//Top 
	{16.5f,		{27.0f, 16.5f, 47.0f},			{0.0f ,0.0f ,0.0f },	{0.99f, 0.99f, 0.99f},	SPEC},//Mirr 
	{16.5f,		{73.0f ,16.5f, 78.0f},			{0.0f ,0.0f ,0.0f },	{0.99f, 0.99f, 0.99f},	REFR},//Glas 
	{600.0f,	{50.0f ,681.6f - 0.27f, 81.6f},	{12.0f ,12.0f ,12.0f},	{1.0f, 1.0f, 1.0f},		DIFF} //Lite 
};


// hash function to calculate new seed for each frame
// see http://www.reedbeta.com/blog/2013/01/12/quick-and-easy-gpu-random-numbers-in-d3d11/
uint WangHash(uint a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

__device__ inline bool intersect_scene(const Ray& r, float& t, int& sphere_id) {

	float tmin = 1e20;
	float tmax = -1e20;
	float d = 1e21;
	float k = 1e21;
	float q = 1e21;
	float inf = t = 1e20;

	// intersect all spheres in the scene
	float numspheres = sizeof(spheres) / sizeof(Sphere);
	for (int i = int(numspheres); i--;)  // for all spheres in scene
		// keep track of distance from origin to closest intersection point
		if ((d = spheres[i].intersect(r)) && d < t) { t = d; sphere_id = i;}

	// t is distance to closest intersection of ray with all primitives in the scene
	return t < inf;
}

// radiance function
// compute path bounces in scene and accumulate returned color from each path sgment
__device__ float3 radiance(Ray& r, curandState* randstate) { // returns ray color

	// colour mask
	float3 mask = make_float3(1.0f, 1.0f, 1.0f);
	// accumulated colour
	float3 accucolor = make_float3(0.0f, 0.0f, 0.0f);
	//float t = 100000;
	//int sphere_id = -1;
	//if (!intersect_scene(r, t, sphere_id))
	//	return make_float3(0.0f, 0.0f, 0.0f); // if miss, return black
	//else accucolor = spheres[sphere_id].color;

	// iteration up to 4 bounces (instead of recursion in CPU code)
	for (int bounces = 0; bounces < 5; bounces++) 
	{  
		// reset scene intersection function parameters
		float t = 100000; // distance to intersection 
		int sphere_id = -1;	// which sphere the ray interesect
		float3 f;  // primitive color
		float3 emit; // primitive emission color
		float3 x; // intersection point
		float3 n; // normal
		float3 nl; // oriented normal
		float3 d; // ray direction of next path segment
		Refl_t refltype;

		// intersect ray with scene
		// intersect_scene keeps track of closest intersected primitive and distance to closest intersection point
		if (!intersect_scene(r, t, sphere_id))
			return make_float3(0.0f, 0.0f, 0.0f); // if miss, return black

		// else: we've got a hit with a scene primitive
		// determine geometry type of primitive: sphere/box/triangle

		Sphere& sphere = spheres[sphere_id]; // hit object with closest intersection
		x = r.origin + r.direction * t;  // intersection point on object
		n = normalize(x - sphere.position);		// normal
		nl =dot(n, r.direction) < 0 ? n : n * -1.0f; // correctly oriented normal
		f = sphere.color;   // object color
		refltype = sphere.reflectType;
		emit = sphere.emission;  // object emission
		accucolor += (mask * emit);

		// SHADING: diffuse, specular or refractive

		// ideal diffuse reflection (see "Realistic Ray Tracing", P. Shirley)
		if (refltype == DIFF) {

			// create 2 random numbers
			float r1 = curand_uniform(randstate) * 2.0f * M_PI;
			float r2 = curand_uniform(randstate);
			float r2s = sqrtf(r2);

			// compute orthonormal coordinate frame uvw with hitpoint as origin 
			float3 w = nl;
			float3 u = normalize(cross((fabs(w.x) > 0.1f ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
			float3 v = cross(w, u);

			// compute cosine weighted random ray direction on hemisphere 
			d = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrtf(1 - r2));

			// offset origin next path segment to prevent self intersection
			x += nl * 0.03f;

			// multiply mask with colour of object
			mask *= f;
		}

		// ideal specular reflection (mirror) 
		if (refltype == SPEC) {

			// compute relfected ray direction according to Snell's law
			d = r.direction - 2.0f * n * dot(n, r.direction);

			// offset origin next path segment to prevent self intersection
			x += nl * 0.01f;

			// multiply mask with colour of object
			mask *= f;
		}

		// ideal refraction (based on smallpt code by Kevin Beason)
		if (refltype == REFR) {

			bool into = dot(n, nl) > 0; // is ray entering or leaving refractive material?
			float nc = 1.0f;  // Index of Refraction air
			float nt = 1.5f;  // Index of Refraction glass/water
			float nnt = into ? nc / nt : nt / nc;  // IOR ratio of refractive materials
			float ddn = dot(r.direction, nl);
			float cos2t = 1.0f - nnt * nnt * (1.0f - ddn * ddn);

			if (cos2t < 0.0f) // total internal reflection 
			{
				d = reflect(r.direction, n); //d = r.dir - 2.0f * n * dot(n, r.dir);
				x += nl * 0.01f;
			}
			else // cos2t > 0
			{
				// compute direction of transmission ray
				float3 tdir = normalize(r.direction * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrtf(cos2t))));

				float R0 = (nt - nc) * (nt - nc) / (nt + nc) * (nt + nc);
				float c = 1.f - (into ? -ddn : dot(tdir, n));
				float Re = R0 + (1.f - R0) * c * c * c * c * c;
				float Tr = 1 - Re; // Transmission
				float P = .25f + .5f * Re;
				float RP = Re / P;
				float TP = Tr / (1.f - P);

				// randomly choose reflection or transmission ray
				if (curand_uniform(randstate) < 0.25) // reflection ray
				{
					mask *= RP;
					d = reflect(r.direction, n);
					x += nl * 0.02f;
				}
				else // transmission ray
				{
					mask *= TP;
					d = tdir; //r = Ray(x, tdir); 
					x += nl * 0.0005f; // epsilon must be small to avoid artefacts
				}
			}
		}

		// set up origin and direction of next path segment
		r.origin = x;
		r.direction = d;
	}

	// add radiance up to a certain ray depth
	// return accumulated ray colour after all bounces are computed
	return accucolor;
}


__device__ unsigned char Color(float c)
{
	c = clamp(c, 0.0f, 1.0f);
	return int(c * 255.99) & 0xff;
}

__global__ void render(uchar4 *pos, float3* accumbuffer, int width, int height, int frameNum, int HashedFrameNum)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= width) || (j >= height)) 
		return;

	int index = j * width + i;
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	// create random number generator, see RichieSams blogspot
	curandState randState; // state of the random number generator, to prevent repetition
	curand_init(HashedFrameNum + threadId, 0, 0, &randState);

	float3 pixelColor = make_float3(0);
	float2 uv = make_float2((float)i / width, (float)j / height) - make_float2(0.5f, 0.5f);
	Ray cam(make_float3(50.0f, 52.0f, 295.6f), normalize(make_float3(0.0f, -0.042612f, -1.0f)));
	float3 cx = make_float3(width * 0.5135f / height, 0.0f, 0.0f);  // ray direction offset along X-axis 
	float3 cy = normalize(cross(cx, cam.direction)) * -0.5135f; // ray dir offset along Y-axis, .5135 is FOV angle
	float3 dir = cx * uv.x + cy * uv.y + cam.direction;
	for (int s = 0; s < spp; s++)
	{
		pixelColor += radiance(Ray(cam.origin + dir * 140.0f, normalize(dir)), &randState) / (float)spp;
	}
	
	accumbuffer[index] += pixelColor;

	float3 tempCol = accumbuffer[index] / (float)frameNum;

	// (0.0f, 1.0f) -> (0, 255)
	unsigned char r = Color(tempCol.x);
	unsigned char g = Color(tempCol.y);
	unsigned char b = Color(tempCol.z);

	pos[index].w = 0;
	pos[index].x = r;
	pos[index].y = g;
	pos[index].z = b;
}

extern "C" void launch_kernel(uchar4* pos, float3* accumbuffer, unsigned int w, unsigned int h, unsigned int frame) {

	//set blocks
	int tx = 8;
	int ty = 8;

	dim3 blocks(w / tx + 1, h / ty + 1);
	dim3 threads(tx, ty);
	render <<<blocks, threads >>> (pos, accumbuffer, w, h, frame, WangHash(frame));

	cudaThreadSynchronize();
	checkCUDAError("kernel failed!");
}

