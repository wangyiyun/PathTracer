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
#include <ctime>

__shared__ float _Seed;
#define M_PI 3.14159265358979323846;

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
	{radius	position						emission				color					reflectType}*/
	{1e5f,	{-1e5f - 100.0f, 0.0f, 0.0f},		{0.0f, 0.0f, 0.0f},		{0.75f, 0.25f, 0.25f},	DIFF},//Left 
	{1e5f,	{1e5f + 100.0f, 0.0f, 0.0f},		{0.0f, 0.0f, 0.0f},		{0.25f, 0.25f, 0.75f},	DIFF},//Rght 
	{1e5f,	{0.0f, 0.0f, -1e5f - 100.0f},		{0.0f, 0.0f, 0.0f},		{0.25f, 0.75f, 0.25f},	DIFF},//Back 
	{1e5f,	{0.0f, 0.0f, 1e5f + 500.0f},		{0.0f, 0.0f, 0.0f},		{0.0f, 0.0f, 0.0f},		DIFF},//Frnt 
	{1e5f,	{0.0f, -1e5f - 100.0f, 0.0f},		{0.0f, 0.0f, 0.0f},		{0.75f, 0.75f, 0.75f},	DIFF},//Botm 
	{1e5f,	{0.0f, 1e5f + 100.0f, 0.0f},		{0.0f, 0.0f, 0.0f},		{0.75f, 0.75f, 0.75f},	DIFF},//Top 
	{20.0f,	{-50.0f, -80.0f, 0.0f},			{0.0f ,0.0f ,0.0f },	{0.99f, 0.99f, 0.99f},	SPEC},//Mirr 
	{10.0f,	{0.0f, -90.0f, 20.0f},			{0.0f ,3.0f ,5.0f },	{0.80f, 0.70f, 0.20f},	DIFF},//light 
	{10.0f,	{-10.0f, -90.0f, -30.0f},			{0.7f ,0.0f ,0.0f },	{0.70f, 0.00f, 0.60f},	DIFF},//light 
	{30.0f,	{40.0f ,-70.0f, 0.0f},			{0.0f ,0.0f ,0.0f },	{0.99f, 0.99f, 0.99f},	REFR},//Glas 
	{50.0f,	{0.0f ,135.0f, 0.0f},			{12.0f ,12.0f ,12.0f},	{1.0f, 1.0f, 1.0f},		DIFF} //Lite 
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

__device__ float fract(float v)
{
	return v - floor(v);
}
// this randmon function is unused
__device__ float m_rand(int frameNum, float2 uv)
{
	float result = fract(sin((_Seed / (float)frameNum) * dot(uv, make_float2(12.9898f, 78.233f))) * 43758.5453f);
	_Seed += 1.0f;
	return result;
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
	{
		// keep track of distance from origin to closest intersection point
		if ((d = spheres[i].intersect(r)) && d < t) { t = d; sphere_id = i;}
	}

	// t is distance to closest intersection of ray with all primitives in the scene
	return t < inf;
}

// radiance function
// compute path bounces in scene and accumulate returned color from each path sgment
__device__ float3 radiance(Ray& r, curandState* randstate, int frameNum, float2 uv) { // returns ray color

	// color mask
	float3 mask = make_float3(1.0f, 1.0f, 1.0f);
	// accumulated color for current pixel
	float3 accucolor = make_float3(0.0f, 0.0f, 0.0f);

	//float t = 100000;
	//int sphere_id = -1;
	//if (!intersect_scene(r, t, sphere_id))
	//	return make_float3(0.0f, 0.0f, 0.0f); // if miss, return black
	//else accucolor = spheres[sphere_id].color;

	int bounces = 0;
	while(bounces < 5 || curand_uniform(randstate) < 0.5f)
	{  
		bounces++;
		// reset scene intersection function parameters
		float t = 1e20; // distance to intersection 
		int sphere_id = -1;	// which sphere the ray interesect
		float3 f;		// primitive color
		float3 emit;	// primitive emission color
		float3 hitPoint;		// intersection point
		float3 n;		// normal
		float3 nl;		// oriented normal
		float3 d;		// ray direction of next path segment
		Refl_t refltype;

		// intersect ray with scene
		// intersect_scene keeps track of closest intersected primitive and distance to closest intersection point
		if (!intersect_scene(r, t, sphere_id))
			return make_float3(0.0f, 0.0f, 0.0f); // if miss, return black

		// else: we've got a hit with a scene primitive

		Sphere& sphere = spheres[sphere_id]; // hit object with closest intersection
		hitPoint = r.origin + r.direction * t;  // intersection point on object
		n = normalize(hitPoint - sphere.position);		// normal
		nl =dot(n, r.direction) < 0 ? n : n * -1.0f; // correctly oriented normal
		f = sphere.color;   // object color
		refltype = sphere.reflectType;
		emit = sphere.emission;  // object emission
		accucolor += (mask * emit);

		// SHADING: diffuse, specular or refractive

		// ideal diffuse reflection
		if (refltype == DIFF) {

			// create 2 random numbers
			float cosTheta = curand_uniform(randstate);
			//float cosTheta = m_rand(frameNum, uv);
			float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
			float phi = 2 * 3.1415926 * curand_uniform(randstate);
			//float phi = 2 * 3.1415926 * m_rand(frameNum, uv);
			// compute orthonormal coordinate frame uvw with hitpoint as origin 
			float3 w = nl;	// normal
			float3 u = normalize(cross(w, (fabs(w.x) > (1 - 0.01) ? make_float3(0, 0, 1) : make_float3(1, 0, 0)))); //tangent
			float3 v = cross(w, u);	// binormal
			// compute cosine weighted random ray direction on hemisphere 
			d = normalize(u * cos(phi) * sinTheta + v * sin(phi) * sinTheta + w * cosTheta);

			// offset origin next path segment to prevent self intersection
			hitPoint += nl * 0.01f;

			// multiply color to the object
			mask *= f;
		}

		// ideal specular reflection
		if (refltype == SPEC) {

			// reflect
			d = r.direction - 2.0f * n * dot(n, r.direction);

			// offset origin next path segment to prevent self intersection
			hitPoint += nl * 0.01;

			// multiply color to the object
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
				hitPoint += nl * 0.01;
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
				if ( curand_uniform(randstate) < 0.25f) // reflection ray
				{
					mask *= RP;
					d = reflect(r.direction, n);
					hitPoint += nl * 0.01;
				}
				else // transmission ray
				{
					mask *= TP;
					d = tdir; //r = Ray(x, tdir); 
					hitPoint += nl * 0.01; // epsilon must be small to avoid artefacts
				}
			}
		}

		// set up origin and direction of next path segment
		r.origin = hitPoint;
		r.direction = d;
	}

	// add radiance up to a certain ray depth
	// return accumulated color after all bounces are computed
	return accucolor;
}


__device__ unsigned char Color(float c)
{
	c = clamp(c, 0.0f, 1.0f);
	return int(c * 255.99) & 0xff;
}
__device__ float3 gammaCorrect(float3 c)
{
	float3 g;
	g.x = pow(c.x, 1 / 2.2f);
	g.y = pow(c.y, 1 / 2.2f);
	g.z = pow(c.z, 1 / 2.2f);
	return g;
}


__global__ void rand_init(int max_x, int max_y, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y))
		return;
	int pixel_index = j * max_x + i;
	// Each thread gets same seed, a different sequence number, no offset
	curand_init(1997 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(uchar4 *pos, float3* accumbuffer, curandState* randSt, int width, int height, int frameNum, int HashedFrameNum)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= width) || (j >= height)) 
		return;
	
	// unique id for the pixel
	int index = j * width + i;
	// create random number generator, see RichieSams blogspot
	curandState randState; // state of the random number generator, to prevent repetition, need refresh per frame
	curand_init(HashedFrameNum + index, 0, 0, &randState);
	float3 pixelColor = make_float3(0);
	// offset inside each pixel
	float offsetX = curand_uniform(&randState);	// get random float between (0, 1)
	float offsetY = curand_uniform(&randState);
	//float offsetX = m_rand(frameNum, make_float2(i, j));	// get random float between (0, 1)
	//float offsetY = m_rand(frameNum, make_float2(i, j));
	//if(index == 0 && frameNum < 100) printf("%f, %f\n", offsetX, offsetY);
	// uv(-0.5, 0.5)
	float2 uv = make_float2((i + offsetX) / width, (j + offsetY) / height) - make_float2(0.5f, 0.5f);
	Ray cam(make_float3(0.0f,0.0f,250.0f), normalize(make_float3(0.0f, 0.0f, -1.0f)));
	float3 screen = make_float3(uv.x * width, -uv.y * height, -500);
	float3 dir = normalize(screen - cam.origin);

	pixelColor = radiance(Ray(cam.origin, dir), &randState, frameNum, uv);
	if (frameNum == 0) accumbuffer[index] = make_float3(0.0);
	accumbuffer[index] += pixelColor;

	float3 tempCol = accumbuffer[index]/(float)frameNum;
	tempCol = gammaCorrect(tempCol);

	// (0.0f, 1.0f) -> (0, 255)
	unsigned char r = Color(tempCol.x);
	unsigned char g = Color(tempCol.y);
	unsigned char b = Color(tempCol.z);
	//debug
	//unsigned char r = Color(dir.x);
	//unsigned char g = Color(dir.y);
	//unsigned char b = Color(dir.z);

	pos[index].w = 0;
	pos[index].x = r;
	pos[index].y = g;
	pos[index].z = b;
}

extern "C" void launch_kernel(uchar4* pos, float3* accumbuffer, curandState* randState, unsigned int w, unsigned int h, unsigned int frame) {

	//set thread number
	int tx = 32;
	int ty = 32;

	dim3 blocks(w / tx + 1, h / ty + 1);
	dim3 threads(tx, ty);
	render <<<blocks, threads >>> (pos, accumbuffer, randState, w, h, frame, WangHash(frame));

	cudaThreadSynchronize();
	checkCUDAError("kernel failed!");
}

