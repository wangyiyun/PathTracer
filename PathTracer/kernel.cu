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

// "__host__": This function called by CPU and runs on CPU
// "__device__": This function called by GPU and runs on GPU (inside one thread)
// "__global__": This is a kernel function, called by CPU and runs on GPU

// reflection type (DIFFuse, SPECular, REFRactive)
enum Refl_t { DIFF, SPEC, REFR };
// geometry type
enum Geom_t { SPHERE, CONE };

struct Ray {
	float3 origin;
	float3 direction;
	// create a ray
	__device__ Ray(float3 o_, float3 d_) : origin(o_), direction(d_) {}
};

struct Hit
{
	float hitDist;		//hitDistance
	float3 normal;
	float3 oriNormal;	// oriented normal (for rafraction)
	float3 nextDir;		// direction for next segment
	float3 color;
	float3 emission;
	Refl_t reflectType;
	Geom_t geomtryType;
	int geomID;
	__device__ void Init() {
		hitDist = 1e20;
		normal = make_float3(0.0f);
		oriNormal = make_float3(0.0f);
		nextDir = make_float3(0.0f);
		color = make_float3(0.0f);
		emission = make_float3(0.0f);
		reflectType = DIFF;
		geomtryType = SPHERE;
		geomID = -1;
	}
};

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
	{1e5f,	{-1e5f - 100.0f, 0.0f, 0.0f},	{0.0f, 0.0f, 0.0f},		{0.75f, 0.25f, 0.25f},	DIFF},//Left 
	{1e5f,	{1e5f + 100.0f, 0.0f, 0.0f},	{0.0f, 0.0f, 0.0f},		{0.25f, 0.25f, 0.75f},	DIFF},//Rght 
	{1e5f,	{0.0f, 0.0f, -1e5f - 100.0f},	{0.0f, 0.0f, 0.0f},		{0.25f, 0.75f, 0.25f},	DIFF},//Back 
	{1e5f,	{0.0f, 0.0f, 1e5f + 500.0f},	{0.0f, 0.0f, 0.0f},		{0.0f, 0.0f, 0.0f},		DIFF},//Frnt 
	{1e5f,	{0.0f, -1e5f - 100.0f, 0.0f},	{0.0f, 0.0f, 0.0f},		{0.75f, 0.75f, 0.75f},	DIFF},//Botm 
	{1e5f,	{0.0f, 1e5f + 100.0f, 0.0f},	{0.0f, 0.0f, 0.0f},		{0.75f, 0.75f, 0.75f},	DIFF},//Top 
	//{20.0f,	{-50.0f, -80.0f, 0.0f},			{0.0f ,0.0f ,0.0f },	{0.99f, 0.99f, 0.99f},	SPEC},//Mirr 
	{10.0f,	{0.0f, -90.0f, 20.0f},			{0.0f ,3.0f ,5.0f },	{0.80f, 0.70f, 0.20f},	DIFF},//light 
	{10.0f,	{-10.0f, -90.0f, -30.0f},		{0.7f ,0.0f ,0.0f },	{0.70f, 0.00f, 0.60f},	DIFF},//light 
	{30.0f,	{40.0f ,-70.0f, 0.0f},			{0.0f ,0.0f ,0.0f },	{0.99f, 0.99f, 0.99f},	REFR},//Glas 
	{50.0f,	{0.0f ,135.0f, 0.0f},			{12.0f ,12.0f ,12.0f},	{1.0f, 1.0f, 1.0f},		DIFF} //Lite 
};

//https://www.shadertoy.com/view/MtcXWr
struct Cone {
	float3 tip, axis;
	float cosA, height;
	float3 emission, color;
	Refl_t reflectType;	//DIFF, SPEC, REFR
	__device__ float intersect(const Ray& r) const { // returns distance, 0 if nohit  

		float3 co = r.origin - tip;
		float3 fixAxis = normalize(axis);
		float a = dot(r.direction, fixAxis) * dot(r.direction, fixAxis) - cosA * cosA;
		float b = 2. * (dot(r.direction, fixAxis) * dot(co, fixAxis) - dot(r.direction, co) * cosA * cosA);
		float c = dot(co, fixAxis) * dot(co, fixAxis) - dot(co, co) * cosA * cosA;

		float det = b * b - 4. * a * c;
		if (det < 0.) return 0;

		det = sqrt(det);
		float t1 = (-b - det) / (2. * a);
		float t2 = (-b + det) / (2. * a);

		// This is a bit messy; there ought to be a more elegant solution.
		float t = t1;
		if (t < 0. || t2 > 0. && t2 < t) t = t2;
		if (t < 0.) return 0;

		float3 cp = r.origin + t * r.direction - tip;
		float h = dot(cp, fixAxis);
		if (h < 0. || h > height) return 0;
		return t;
	}
};

__constant__ Cone cones[] = {
	/*
	tip							axis					cosA	height	emission				color					reflectType*/
	{{-50.0f, -20.0f, 0.0f},	{0.0f, -1.0f, 0.0f},	0.95f,	80.0f,	{0.0f ,0.0f ,0.0f },	{0.99f, 0.99f, 0.99f},	DIFF}
};

__device__ inline bool intersect_scene(const Ray& ray, Hit& bestHit)
{
	float d = 1e20;
	float INF = 1e20;

	// intersect all spheres in the scene
	float spheresNum = sizeof(spheres) / sizeof(Sphere);
	for (int i = 0; i < spheresNum; i++)  // for all spheres in scene
	{
		// keep track of distance from origin to closest intersection point
		if ((d = spheres[i].intersect(ray)) && d < bestHit.hitDist && d > 0)
		{ 
			bestHit.hitDist = d;
			bestHit.geomtryType = SPHERE;
			bestHit.geomID = i;
		}
	}

	// intersect all cones in the scene
	float conesNum = sizeof(cones) / sizeof(Cone);
	for (int i = 0; i < conesNum; i++)  // for all cones in scene
	{
		// keep track of distance from origin to closest intersection point
		if ((d = cones[i].intersect(ray)) && d < bestHit.hitDist && d > 0)
		{
			bestHit.hitDist = d;
			bestHit.geomtryType = CONE;
			bestHit.geomID = i;
		}
	}

	// t is distance to closest intersection of ray with all primitives in the scene
	if (bestHit.hitDist < INF)
	{
		float3 hitPostion = ray.origin + ray.direction * bestHit.hitDist;
		switch (bestHit.geomtryType)
		{
		case SPHERE:
			bestHit.normal = normalize(hitPostion - spheres[bestHit.geomID].position);
			bestHit.oriNormal = dot(bestHit.normal, ray.direction) < 0 ? bestHit.normal : bestHit.normal * -1;
			bestHit.color = spheres[bestHit.geomID].color;
			bestHit.emission = spheres[bestHit.geomID].emission;
			bestHit.reflectType = spheres[bestHit.geomID].reflectType;
			break;
		case CONE:
			float3 cp = hitPostion - cones[bestHit.geomID].tip;
			bestHit.normal = normalize(cp * dot(cones[bestHit.geomID].axis, cp) / dot(cp, cp) - cones[bestHit.geomID].axis);
			bestHit.oriNormal = dot(bestHit.normal, ray.direction) < 0 ? bestHit.normal : bestHit.normal * -1;
			bestHit.color = cones[bestHit.geomID].color;
			bestHit.emission = cones[bestHit.geomID].emission;
			bestHit.reflectType = cones[bestHit.geomID].reflectType;
			break;
		default:
			break;
		}
		return true;
	}
	else return false;
}

// radiance function
// compute path bounces in scene and accumulate returned color from each path sgment
__device__ float3 radiance(Ray& ray, curandState* randstate, int frameNum) { // returns ray color

	Hit bestHit;
	// color mask
	float3 colorMask = make_float3(1.0f, 1.0f, 1.0f);
	// accumulated color for current pixel
	float3 accuColor = make_float3(0.0f, 0.0f, 0.0f);

	//// hit debug
	//bestHit.Init();
	//if (!intersect_scene(ray, bestHit))
	//	return make_float3(0.0f, 0.0f, 0.0f); // if miss, return black
	//else
	//{
	//	return (bestHit.normal + make_float3(0.5f)) / 2.0f;
	//	//return bestHit.emission;
	//}
	//// hit debug end

	int bounces = 0;
	while(bounces < 5 || curand_uniform(randstate) < 0.5f)
	{  
		bounces++;
		bestHit.Init();
		// intersect ray with scene
		if (!intersect_scene(ray, bestHit))
			return make_float3(0.0f, 0.0f, 0.0f); // if miss, return black

		// else: we've got a hit with a scene primitive
		accuColor += (colorMask * bestHit.emission);
		float3 hitPosition = ray.origin + ray.direction * bestHit.hitDist;

		// SHADING: diffuse, specular or refractive

		// ideal diffuse reflection
		if (bestHit.reflectType == DIFF) {

			// create 2 random numbers
			// create 2 random numbers
			float r1 = 2 * 3.1415926 * curand_uniform(randstate);
			float r2 = curand_uniform(randstate);
			float r2s = sqrtf(r2);

			// compute orthonormal coordinate frame uvw with hitpoint as origin 
			float3 w = bestHit.oriNormal;
			float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
			float3 v = cross(w, u);

			// compute cosine weighted random ray direction on hemisphere 
			bestHit.nextDir = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrtf(1 - r2));

			// offset origin next path segment to prevent self intersection
			hitPosition += bestHit.oriNormal * 0.03;

			// multiply mask with colour of object
			colorMask *= bestHit.color;
		}

		// ideal specular reflection
		if (bestHit.reflectType == SPEC) {

			// reflect
			bestHit.nextDir = ray.direction - 2.0f * bestHit.normal * dot(bestHit.normal, ray.direction);

			// offset origin next path segment to prevent self intersection
			hitPosition += bestHit.oriNormal * 0.01;

			// multiply color to the object
			colorMask *= bestHit.color;
		}

		// ideal refraction (based on smallpt code by Kevin Beason)
		if (bestHit.reflectType == REFR) {

			bool into = dot(bestHit.normal, bestHit.oriNormal) > 0; // is ray entering or leaving refractive material?
			float nc = 1.0f;  // Index of Refraction air
			float nt = 1.5f;  // Index of Refraction glass/water
			float nnt = into ? nc / nt : nt / nc;  // IOR ratio of refractive materials
			float ddn = dot(ray.direction, bestHit.oriNormal);
			float cos2t = 1.0f - nnt * nnt * (1.f - ddn * ddn);

			if (cos2t < 0.0f) // total internal reflection 
			{
				bestHit.nextDir = reflect(ray.direction, bestHit.normal); //d = r.dir - 2.0f * n * dot(n, r.dir);
				hitPosition += bestHit.oriNormal * 0.01f;
			}
			else // cos2t > 0
			{
				// compute direction of transmission ray
				float3 tdir = normalize(ray.direction * nnt - bestHit.normal * ((into ? 1 : -1) * (ddn * nnt + sqrtf(cos2t))));

				float R0 = (nt - nc) * (nt - nc) / (nt + nc) * (nt + nc);
				float c = 1.f - (into ? -ddn : dot(tdir, bestHit.normal));
				float Re = R0 + (1.f - R0) * c * c * c * c * c;
				float Tr = 1 - Re; // Transmission
				float P = .25f + .5f * Re;
				float RP = Re / P;
				float TP = Tr / (1.f - P);

				// randomly choose reflection or transmission ray
				if (curand_uniform(randstate) < 0.25) // reflection ray
				{
					colorMask *= RP;
					bestHit.nextDir = reflect(ray.direction, bestHit.normal);
					hitPosition += bestHit.oriNormal * 0.02f;
				}
				else // transmission ray
				{
					colorMask *= TP;
					bestHit.nextDir = tdir; //r = Ray(x, tdir); 
					hitPosition += bestHit.oriNormal * 0.0005f; // epsilon must be small to avoid artefacts
				}
			}
		}

		// set up origin and direction of next path segment
		ray.origin = hitPosition;
		ray.direction = bestHit.nextDir;
	}

	// add radiance up to a certain ray depth
	// return accumulated color after all bounces are computed
	return accuColor;
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

	pixelColor = radiance(Ray(cam.origin, dir), &randState, frameNum);
	if (frameNum == 0) accumbuffer[index] = make_float3(0.0);	//init
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
	int tx = 16;
	int ty = 16;

	dim3 blocks(w / tx + 1, h / ty + 1);
	dim3 threads(tx, ty);
	render <<<blocks, threads >>> (pos, accumbuffer, randState, w, h, frame, WangHash(frame));

	cudaThreadSynchronize();
	checkCUDAError("kernel failed!");
}

