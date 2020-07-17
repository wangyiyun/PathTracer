#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
using namespace std;
#include <stdio.h>
#include "cutil_math.h"
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>
#include <unordered_map>

__device__ const double PI = 3.141592653589793238463;
#define OBJ_INFO_COUNT 7
#define room_width 300.0f
#define room_height 300.0f
#define room_depth 1200.0f
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
// "__constant__": This data won't and can't be modified

// Changing variables
__constant__ float3 cam_right = { 200.0f, 150.0f, 1100.0f };
__constant__ float3 cam_left = { 100.0f, 150.0f, 1100.0f };
#define USING_WAVE 0	// from 0 to 10

// reflection type (DIFFuse, SPECular, REFRactive)
enum Refl_t { DIFF, SPEC, REFR };
// geometry type
enum Geom_t { SPHERE, CONE, TRIANGLE};

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
	Refl_t reflectType;
	Geom_t geomtryType;
	int geomID;
	float3 color;
	float3 emission;
	__device__ void Init() {
		hitDist = 1e20;
		normal = make_float3(0.0f);
		oriNormal = make_float3(0.0f);
		nextDir = make_float3(0.0f);
		reflectType = DIFF;
		geomtryType = SPHERE;
		geomID = -1;
		color = make_float3(0.0f);
		emission = make_float3(0.0f);
	}
};

struct Sphere {

	float radius;
	float3 position;
	int matName;
	float temperature;
	Refl_t reflectType;	//DIFF, SPEC, REFR
	__device__ float intersect(const Ray& ray) const { // returns distance, 0 if nohit 

		// Ray/sphere intersection
		// Quadratic formula required to solve ax^2 + bx + c = 0 
		// Solution x = (-b +- sqrt(b*b - 4ac)) / 2a
		// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 

		float3 op = position - ray.origin;
		float t, epsilon = 0.01f;
		float b = dot(op, ray.direction);
		float disc = b * b - dot(op, op) + radius * radius; // discriminant
		if (disc < 0) return 0; else disc = sqrtf(disc);
		return (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0);
	}
};

struct Cone {
	float theta;
	float3 tip, axis;
	int matName;
	float temperature;
	Refl_t reflectType;	//DIFF, SPEC, REFR
	__device__ float intersect(const Ray& ray) const { // returns distance, 0 if nohit  

		float3 co = ray.origin - tip; float cos2t = cos(theta*PI/180.0f); cos2t *= cos2t;
		float t, dotDV = dot(ray.direction, axis), dotCOV = dot(co, axis);
		float a = dotDV * dotDV - cos2t, b = 2.0f * (dotDV * dotCOV - dot(ray.direction, co) * cos2t),
			c = dotCOV * dotCOV - dot(co, co) * cos2t, delta = b * b - 4 * a * c;
		if (delta <= 0.0f) return 0; else delta = sqrt(delta);
		t = (-b + delta) / 2.0f / a > 0.01f ? (-b + delta) / 2.0f / a : max((-b - delta) / 2.0f / a, 0.0f);
		float3 hit = ray.origin + t * ray.direction;
		if (dot(hit - tip, axis) <= 0.0f) return 0;
		return t;
	}
};

__device__ float TriangleIntersect(const Ray& ray, float3 vert0, float3 vert1, float3 vert2, float &u, float &v)
{
	// find vectors for two edges sharing vert0
	float3 edge1 = vert1 - vert0;
	float3 edge2 = vert2 - vert0;
	float t;
	// begin calculating determinant - also used to calculate U parameter
	float3 pvec = cross(ray.direction, edge2);
	// if determinant is near zero, ray lies in plane of triangle
	float det = dot(edge1, pvec);
	// use backface culling
	if (det < 0.01f)
		return 0;
	float inv_det = 1.0f / det;
	// calculate distance from vert0 to ray origin
	float3 tvec = ray.origin - vert0;
	// calculate U parameter and test bounds
	u = dot(tvec, pvec) * inv_det;
	if (u < 0.0 || u > 1.0f)
		return 0;
	// prepare to test V parameter
	float3 qvec = cross(tvec, edge1);
	// calculate V parameter and test bounds
	v = dot(ray.direction, qvec) * inv_det;
	if (v < 0.0 || u + v > 1.0f)
		return 0;
	// calculate t, ray intersects triangle
	t = dot(edge2, qvec) * inv_det;
	return t;
}

__device__ inline bool intersect_scene(const Ray& ray, Hit& bestHit, 
	int vertsNum, float3* scene_verts, int objsNum, int* scene_objs_info,
	float2* scene_uvs, float3* scene_normals,
	int texNum, int* tex_wh, float3* tex_data, float* colorList, float* emiList)
{
	float d = 1e20;
	float INF = 1e20;

	// intersect all triangles in the scene
	int currentObj = 0;	// current object max vert = scene_objs_info[currentObj]
	int facesNum = vertsNum / 3;
	for (int i = 0; i < facesNum; i++)
	{
		float3 v0 = scene_verts[i * 3];
		float3 v1 = scene_verts[i * 3 + 1];
		float3 v2 = scene_verts[i * 3 + 2];
		int currentVert = i * 3 + 2;
		// u, v, 1-u-v; 
		float u = 0; 
		float v = 0;
		// which object?
		if (currentObj + 1 < objsNum)
		{
			// move to next obj
			if (currentVert >= scene_objs_info[(currentObj + 1) * OBJ_INFO_COUNT]) currentObj++;
		}
		if ((d = TriangleIntersect(ray, v0, v1, v2, u, v)) && d < bestHit.hitDist && d > 0)
		{
			float2 uv0 = scene_uvs[i * 3];
			float2 uv1 = scene_uvs[i * 3 + 1];
			float2 uv2 = scene_uvs[i * 3 + 2];
			float w = 1 - u - v;
			float2 uv = w*uv0 + u*uv1 + v*uv2;
			//bestHit.color = make_float3(uv.x, uv.y, 0);
			bestHit.reflectType = DIFF;
			if (scene_objs_info[currentObj * OBJ_INFO_COUNT + 4] == 1) bestHit.reflectType = SPEC;
			else if (scene_objs_info[currentObj * OBJ_INFO_COUNT + 4] == 2) bestHit.reflectType = REFR;
			// do not have a normal texture
			if (scene_objs_info[currentObj * OBJ_INFO_COUNT + 2] == -1)
			{
				// smooth shading
				float3 n0 = scene_normals[i * 3];
				float3 n1 = scene_normals[i * 3 + 1];
				float3 n2 = scene_normals[i * 3 + 2];

				bestHit.normal = normalize(w * n0 + u * n1 + v * n2);
			}
			else
			{
				// find normal tex in all textures
				int texIndex = scene_objs_info[currentObj * OBJ_INFO_COUNT + 2];
				int texWidth = tex_wh[texIndex * 2];
				int texHeight = tex_wh[texIndex * 2 + 1];
				int offset = 0;	// get pixel offset in tex_data
				for (int t = 0; t < texIndex; t++)
				{
					offset += tex_wh[t * 2] * tex_wh[t * 2 + 1];
				}
				// map current uv(float2) to index in tex_data[]
				int u_index = uv.x * texWidth;
				int v_index = uv.y * texHeight;
				// map the color in tex_data[offset + u_index*texWidth + v_index] to normal
				bestHit.normal = normalize(tex_data[offset + v_index * texWidth + u_index] * 2.0f - 1.0f);
			}
			// which color source?
			// 0: value, 1: tex
			if (scene_objs_info[currentObj * OBJ_INFO_COUNT + 5] == 0)
			{
				bestHit.color = make_float3(colorList[currentObj * 3],
					colorList[currentObj * 3 + 1],
					colorList[currentObj * 3 + 2]);
			}
			else if (scene_objs_info[currentObj * OBJ_INFO_COUNT + 5] == 1 && scene_objs_info[currentObj * OBJ_INFO_COUNT + 3] != -1)
			{
				// find tex in all textures
				int texIndex = scene_objs_info[currentObj * OBJ_INFO_COUNT + 3];
				int texWidth = tex_wh[texIndex * 2];
				int texHeight = tex_wh[texIndex * 2 + 1];
				int offset = 0;	// get pixel offset in tex_data
				for (int t = 0; t < texIndex; t++)
				{
					offset += tex_wh[t * 2] * tex_wh[t * 2 + 1];
				}
				// map current uv(float2) to index in tex_data[]
				int u_index = uv.x * texWidth;
				int v_index = uv.y * texHeight;
				// map the color in tex_data[offset + v_index * texWidth + u_index] to emissivity
				bestHit.color = tex_data[offset + v_index * texWidth + u_index];
			}
			// which emi source?
			// 0: value, 1: tex
			if (scene_objs_info[currentObj * OBJ_INFO_COUNT + 6] == 0)
			{
				bestHit.emission = make_float3(emiList[currentObj * 3],
					emiList[currentObj * 3 + 1],
					emiList[currentObj * 3 + 2])*15.0f;
			}
			else if (scene_objs_info[currentObj * OBJ_INFO_COUNT + 6] == 1 && scene_objs_info[currentObj * OBJ_INFO_COUNT + 3] != -1)
			{
				// find tex in all textures
				int texIndex = scene_objs_info[currentObj * OBJ_INFO_COUNT + 3];
				int texWidth = tex_wh[texIndex * 2];
				int texHeight = tex_wh[texIndex * 2 + 1];
				int offset = 0;	// get pixel offset in tex_data
				for (int t = 0; t < texIndex; t++)
				{
					offset += tex_wh[t * 2] * tex_wh[t * 2 + 1];
				}
				// map current uv(float2) to index in tex_data[]
				int u_index = uv.x * texWidth;
				int v_index = uv.y * texHeight;
				// map the color in tex_data[offset + v_index * texWidth + u_index] to emissivity
				bestHit.emission = tex_data[offset + v_index * texWidth + u_index];
			}
			
			bestHit.hitDist = d;
			bestHit.geomtryType = TRIANGLE;
			bestHit.oriNormal = dot(bestHit.normal, ray.direction) < 0.0f ? bestHit.normal : bestHit.normal * -1.0f;
		}
	}
	
	// t is distance to closest intersection of ray with all primitives in the scene
	if (bestHit.hitDist < INF)
	{
		return true;
	}
	else return false;
}

// radiance function
// compute path bounces in scene and accumulate returned color from each path sgment
__device__ float3 radiance(Ray& ray, curandState* randstate, int frameNum, int index, 
	int vertsNum, float3* scene_verts, int objsNum, int* scene_objs_info,
	float2* scene_uvs, float3* scene_normals,
	int texNum, int* tex_wh, float3* tex_data,
	int type, float* colorList, float* emiList) {

	Hit bestHit;
	// accumulated color for current pixel
	float3 colorMask = make_float3(1.0f);
	// accumulated color for current pixel
	float3 accuIntensity = make_float3(0.0f);

	//// hit debug
	//bestHit.Init();
	//if (!intersect_scene(ray, bestHit, vertsNum, scene_verts, objsNum, scene_objs_info, scene_uvs, scene_normals,
	//	texNum, tex_wh, tex_data, colorList))
	//	return make_float3(0.0f); // if miss, return black
	//else
	//{
	//	return bestHit.color;
	//}
	//// hit debug end

	int bounces = 0;
	while(bounces < 5 || curand_uniform(randstate) < 0.5f)
	{  
		if (bounces >= 10) break;
		bounces++;
		bestHit.Init();
		float emi = 0.0f;
		float rt = 0.0f;
		// intersect ray with scene
		if (!intersect_scene(ray, bestHit, vertsNum, scene_verts, objsNum, scene_objs_info, scene_uvs, scene_normals,
							texNum, tex_wh, tex_data, colorList, emiList))
		{
			// sky color
			break; // if miss STOP looping, will influnce the output of recuData since already return 
		}

		// else: we've got a hit with a scene primitive
		accuIntensity += colorMask * bestHit.emission;

		float3 hitPosition = ray.origin + ray.direction * bestHit.hitDist;

		// ideal diffuse reflection
		if (bestHit.reflectType == DIFF)
		{
			// create 2 random numbers
			float r1 = 2 * PI * curand_uniform(randstate);
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

			colorMask *= bestHit.color;
		}
		// ideal specular reflection
		if (bestHit.reflectType == SPEC)
		{

			// reflect
			bestHit.nextDir = ray.direction - 2.0f * bestHit.normal * dot(bestHit.normal, ray.direction);

			// offset origin next path segment to prevent self intersection
			hitPosition += bestHit.oriNormal * 0.01;

			// multiply color to the object
			colorMask *= bestHit.color;
		}

		// ideal refraction (based on smallpt code by Kevin Beason)
		if (bestHit.reflectType == REFR)
		{
			float nc = 1.0, ng = 1.5; //Refraction index
			bool inside = dot(bestHit.normal, bestHit.oriNormal) < 0;
			//Snells Law
			float eta = inside ? ng / nc : nc / ng, R0 = pow((nc - ng) / (nc + ng), 2), c = abs(dot(ray.direction, bestHit.oriNormal));
			float k = 1.0 - eta * eta * (1.0 - c * c);
			if (k < 0)
				bestHit.nextDir = reflect(ray.direction, bestHit.oriNormal);
			else {
				//Shilick's approximation of Fresnel's equation
				float Re = R0 + (1 - R0) * pow(1 - c, 5);
				if (curand_uniform(randstate) < Re)
					bestHit.nextDir = reflect(ray.direction, bestHit.oriNormal);
				else {
					bestHit.nextDir = (eta * ray.direction - (eta * dot(bestHit.oriNormal, ray.direction) + sqrt(k)) * bestHit.oriNormal);
				}
			}
			// multiply color to the object
			colorMask *= bestHit.color;
		}
		// set up origin and direction of next path segment
		ray.origin = hitPosition;
		ray.direction = bestHit.nextDir;
	}

	return accuIntensity;
}

__device__ float3 gammaCorrect(float3 c)
{
	float3 g;
	g.x = pow(c.x, 1 / 2.2f);
	g.y = pow(c.y, 1 / 2.2f);
	g.z = pow(c.z, 1 / 2.2f);
	return g;
}

__global__ void render(float3 *result, float3* accumbuffer, curandState* randSt, 
	int width, int height, int frameNum, int HashedFrameNum, bool camAtRight, 
	int vertsNum, float3* scene_verts, int objsNum, int* scene_objs_info,
	float2* scene_uvs, float3* scene_normals,
	int texNum, int* tex_wh, float3* tex_data,
	int type, float* colorList, float* emiList)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= width) || (j >= height)) 
		return;
	// unique id for the pixel
	int index = j * width + i;
	if (frameNum == 0)	//init
	{
		accumbuffer[index] = make_float3(0.0);
	}
	else
	{
		// create random number generator, see RichieSams blogspot
		curandState randState; // state of the random number generator, to prevent repetition, need refresh per frame
		curand_init(HashedFrameNum + index, 0, 0, &randState);
		float3 pixelColor = make_float3(0);
		// offset inside each pixel
		float offsetX = curand_uniform(&randState);	// get random float between (0, 1)
		float offsetY = curand_uniform(&randState);
		// uv(-0.5, 0.5)
		float2 uv = make_float2((i + offsetX) / width, (j + offsetY) / height) - make_float2(0.5f, 0.5f);
		float3 camPos;
		if (camAtRight) camPos = cam_right;
		else camPos = cam_left;
		Ray cam(camPos, normalize(make_float3(0.0f, 0.0f, -1.0f)));
		float3 screen = make_float3(uv.x * width + room_width / 2.0f, -uv.y * height + room_width / 2.0f, 1100.0f - (width / 2.0f) * 1.73205080757f);
		// screen x offset
		if (camAtRight)
		{
			screen += make_float3(50.0f, 0.0f, 0.0f);
		} 
		else
		{
			screen -= make_float3(50.0f, 0.0f, 0.0f);
		}
		float3 dir = normalize(screen - cam.origin);
		//result[index] = make_float3(dir.x);
		float3 intensity = radiance(Ray(cam.origin, dir), &randState, frameNum, index, 
			vertsNum, scene_verts, objsNum, scene_objs_info, scene_uvs, scene_normals,
			texNum, tex_wh, tex_data,
			type, colorList, emiList);
		pixelColor = intensity;

		accumbuffer[index] += pixelColor;
	}
	float3 tempCol = accumbuffer[index] / (float)frameNum;
	//tempCol = gammaCorrect(tempCol);

	result[index] = tempCol;
}

extern "C" void launch_kernel(float3* result, float3* accumbuffer, curandState* randState, 
	unsigned int w, unsigned int h, unsigned int frame, 
	bool camAtRight, 
	int vertsNum, float3* scene_verts, 
	int objsNum, int* scene_objs_info,
	float2* scene_uvs, float3* scene_normals,
	int texNum, int* tex_wh, float3* tex_data,
	int type, float* colorList, float* emiList) {

	//set thread number
	int tx = 16;
	int ty = 16;

	dim3 blocks(w / tx + 1, h / ty + 1);
	dim3 threads(tx, ty);
	
	render <<<blocks, threads >>> (result, accumbuffer, randState, w, h, frame, WangHash(frame), 
		camAtRight,
		vertsNum, scene_verts, objsNum, scene_objs_info, scene_uvs, scene_normals,
		texNum, tex_wh, tex_data,
		type, colorList, emiList);

	cudaThreadSynchronize();
	checkCUDAError("kernel failed!");
}

