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

__device__ const double c = 299792458, k = 138064852e-31, PI = 3.141592653589793238463;

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

// mat name
#define mat_human 0
#define mat_marble 1
#define mat_paint 2
#define mat_glass 3
#define mat_rubber 4
#define mat_brass 5
#define mat_road 6
#define mat_al 7
#define mat_al2o3 8
#define mat_brick 9

__constant__ float wave[11] = {
	7.8576538e+02,
	8.1770000e+02,
	8.6250000e+02,
	9.1025000e+02,
	9.4255000e+02,
	9.7750000e+02,
	1.0277500e+03,
	1.0780000e+03,
	1.1255000e+03,
	1.1860000e+03,
	1.2766667e+03
};


// emiLib[waveNum][matName]
__constant__ float emiLib[11][10] = {
	/*
	human,			marble,			paint,			glass,			rubber,			brass,			road,			al,				al2o3,			brick*/
	9.9000000e-01,	9.5834758e-01,	8.7470001e-01,	5.0455443e-01,	9.2789246e-01,	1.2250251e-01,	9.6426578e-01,	5.5701898e-01,	4.1617280e-02,	9.7773773e-01,
	9.9000000e-01,	9.5462609e-01,	8.8365367e-01,	2.8523451e-01,	9.2827028e-01,	1.1789014e-01,	9.7194589e-01,	5.4616836e-01,	4.1602933e-02,	9.7348785e-01,
	9.9000000e-01,	9.5099592e-01,	9.6279529e-01,	3.8887318e-01,	9.2640468e-01,	1.2078545e-01,	9.6430868e-01,	5.2990503e-01,	4.0821044e-02,	9.6252597e-01,
	9.9000000e-01,	9.5741246e-01,	8.6909910e-01,	4.2252257e-01,	9.2027605e-01,	1.2892990e-01,	9.4494491e-01,	5.1621436e-01,	4.8036999e-02,	9.4693874e-01,
	9.9000000e-01,	9.6385735e-01,	8.5889954e-01,	4.4505789e-01,	9.2317386e-01,	1.3452107e-01,	9.5513005e-01,	5.0484414e-01,	1.4619579e-01,	9.3275042e-01,
	9.9000000e-01,	9.6087765e-01,	9.3344199e-01,	4.7704424e-01,	8.9968776e-01,	1.4311263e-01,	9.5631467e-01,	4.9568769e-01,	2.6974721e-01,	9.1201603e-01,
	9.9000000e-01,	9.5962251e-01,	9.4205163e-01,	5.6399482e-01,	8.6774658e-01,	1.4932587e-01,	9.5258259e-01,	4.7984848e-01,	4.2480553e-01,	8.7901868e-01,
	9.9000000e-01,	9.5305901e-01,	9.4627694e-01,	3.2859562e-01,	8.8061124e-01,	1.4229701e-01,	9.1783893e-01,	4.6578646e-01,	4.7823023e-01,	8.5128884e-01,
	9.9000000e-01,	9.5385122e-01,	9.5199753e-01,	4.2369253e-02,	8.9911606e-01,	1.3455656e-01,	9.1771733e-01,	4.5454008e-01,	5.1389488e-01,	9.0261137e-01,
	9.9000000e-01,	9.5852822e-01,	9.5649050e-01,	2.7487807e-02,	9.1817783e-01,	1.2604779e-01,	9.1884949e-01,	4.3838823e-01,	5.4462383e-01,	9.3754130e-01,
	9.9000000e-01,	9.5240096e-01,	9.5069231e-01,	8.9005827e-02,	9.3104627e-01,	1.1098321e-01,	9.5362853e-01,	4.1783501e-01,	5.6727138e-01,	9.7270040e-01
};

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
	int matName;
	float temperature;
	float emissivity;
	Refl_t reflectType;
	Geom_t geomtryType;
	int geomID;
	float3 emission;
	float3 color;	// just for texture debug
	__device__ void Init() {
		hitDist = 1e20;
		normal = make_float3(0.0f);
		oriNormal = make_float3(0.0f);
		nextDir = make_float3(0.0f);
		matName = -1;
		temperature = 0.0f;
		emissivity = 0.0f;
		reflectType = DIFF;
		geomtryType = SPHERE;
		geomID = -1;
		color = make_float3(0.0f);
	}
};

struct Sphere {

	float radius;
	float3 position;
	float3 emission;
	float3 color;
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
	float3 emission;
	float3 color;
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

#define room_width 300.0f
#define room_height 300.0f
#define room_depth 1200.0f
__constant__ Sphere spheres[] = {
	/* cornell box
	{radius	position														emission				color			reflectType*/
	{1e5f,	{-1e5f, 0.0f, 0.0f},											{0.0f ,0.0f, 0.0f},		{0.75f ,0.25f, 0.25f},	DIFF},// left wall
	{1e5f,	{1e5f + room_width, 0.0f, 0.0f},								{0.0f ,0.0f, 0.0f},		{0.25f ,0.25f, 0.75f},	DIFF},// right wall
	{1e5f,	{0.0f, 0.0f, -1e5f},											{0.0f ,0.0f, 0.0f},		{0.75f ,0.75f, 0.75f},	DIFF},// back wall
	{1e5f,	{0.0f, 0.0f, 1e5f + room_depth},								{0.0f ,0.0f, 0.0f},		{0.0f ,0.0f, 0.0f},		DIFF},// front wall
	{1e5f,	{0.0f, -1e5f, 0.0f},											{0.0f ,0.0f, 0.0f},		{0.75f ,0.75f, 0.75f},	DIFF},// floor
	{1e5f,	{0.0f, 1e5f + room_height, 0.0f},								{0.0f ,0.0f, 0.0f},		{0.75f ,0.75f, 0.75f},	DIFF},// ceiling  
	{50.0f,	{200.0f ,50.0f, 700.0f},										{0.0f ,0.0f, 0.0f},		{1.0f ,1.0f, 1.0f},		DIFF},// sphere 
	{600.0f,{room_width / 2 ,room_height + 600.0f - 2.0f, room_depth / 2},	{12.0f ,12.0f, 12.0f},	{1.0f ,1.0f, 1.0f},	DIFF} // lamp 
};

__constant__ Cone cones[] = {
	/*
	theta	tip							axis					emission			color					reflectType*/
	{15,	{100.0f, 80.0f, 500.0f},	{0.0f, -1.0f, 0.0f},	{0.0f, 0.0f, 0.0f}, {0.75f ,0.75f, 0.25f},	DIFF}
};

__device__ inline bool intersect_scene(const Ray& ray, Hit& bestHit, 
	int vertsNum, float3* scene_verts, int objsNum, int* scene_objs_info,
	float2* scene_uvs, float3* scene_normals,
	int texNum, int* tex_wh, float3* tex_data)
{
	float d = 1e20;
	float INF = 1e20;

	//// intersect all spheres in the scene
	//float spheresNum = sizeof(spheres) / sizeof(Sphere);
	//for (int i = 0; i < spheresNum; i++)  // for all spheres in scene
	//{
	//	// keep track of distance from origin to closest intersection point
	//	if ((d = spheres[i].intersect(ray)) && d < bestHit.hitDist && d > 0.0f)
	//	{ 
	//		bestHit.hitDist = d;
	//		bestHit.geomtryType = SPHERE;
	//		bestHit.geomID = i;

	//	}
	//}

	//// intersect all cones in the scene
	//float conesNum = sizeof(cones) / sizeof(Cone);
	//for (int i = 0; i < conesNum; i++)  // for all cones in scene
	//{
	//	// keep track of distance from origin to closest intersection point
	//	if ((d = cones[i].intersect(ray)) && d < bestHit.hitDist && d > 0.0f)
	//	{
	//		bestHit.hitDist = d;
	//		bestHit.geomtryType = CONE;
	//		bestHit.geomID = i;
	//	}
	//}

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
			if (currentVert >= scene_objs_info[(currentObj + 1) * 5]) currentObj++;
		}
		if ((d = TriangleIntersect(ray, v0, v1, v2, u, v)) && d < bestHit.hitDist && d > 0)
		{
			float2 uv0 = scene_uvs[i * 3];
			float2 uv1 = scene_uvs[i * 3 + 1];
			float2 uv2 = scene_uvs[i * 3 + 2];
			float w = 1 - u - v;
			float2 uv = w*uv0 + u*uv1 + v*uv2;
			//bestHit.color = make_float3(uv.x, uv.y, 0);
			// [objVertsNum, matNum, normalTexNum, ambientTexNum]
			// do not have a normal texture
			if (scene_objs_info[currentObj * 5 + 2] == -1)
			{
				bestHit.normal = normalize(scene_normals[i]);
			}
			else
			{
				// find normal tex in all textures
				int texIndex = scene_objs_info[currentObj * 5 + 2];
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
				bestHit.normal = normalize(tex_data[offset + u_index * texWidth + v_index] * 2.0f - 1.0f);
			}
			// do not have an ambient texture
			if (scene_objs_info[currentObj * 5 + 3] == -1)
			{
				bestHit.color = make_float3(0.25f);
			}
			else
			{
				// find normal tex in all textures
				int texIndex = scene_objs_info[currentObj * 5 + 3];
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
				// map the color in tex_data[offset + u_index*texWidth + v_index] to emissivity
				bestHit.color = tex_data[offset + u_index * texWidth + v_index];
			}
			
			bestHit.hitDist = d;
			bestHit.geomtryType = TRIANGLE;
			bestHit.oriNormal = dot(bestHit.normal, ray.direction) < 0.0f ? bestHit.normal : bestHit.normal * -1.0f;
			bestHit.reflectType = DIFF;
			bestHit.emission = make_float3(0.0f);
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
			bestHit.oriNormal = dot(bestHit.normal, ray.direction) < 0.0f ? bestHit.normal : bestHit.normal * -1.0f;
			bestHit.emission = spheres[bestHit.geomID].emission;
			bestHit.color = spheres[bestHit.geomID].color;
			bestHit.reflectType = spheres[bestHit.geomID].reflectType;
			break;
		case CONE:
			float3 cp = hitPostion - cones[bestHit.geomID].tip;
			bestHit.normal = normalize(cp * dot(cones[bestHit.geomID].axis, cp) / dot(cp, cp) - cones[bestHit.geomID].axis);
			bestHit.oriNormal = dot(bestHit.normal, ray.direction) < 0.0f ? bestHit.normal : bestHit.normal * -1.0f;
			bestHit.emission = cones[bestHit.geomID].emission;
			bestHit.color = cones[bestHit.geomID].color;
			bestHit.reflectType = cones[bestHit.geomID].reflectType;
			break;
		case TRIANGLE:
			
			break;
		default:
			break;
		}
		return true;
	}
	else return false;
}

struct RecursionData
{
	float3 color;
	__device__ void add(float3 c)
	{
		color = c;
	}
	__device__ void init()
	{
		color = make_float3(0);
	}
};

// radiance function
// compute path bounces in scene and accumulate returned color from each path sgment
__device__ float3 radiance(Ray& ray, curandState* randstate, int frameNum, int index, 
	int vertsNum, float3* scene_verts, int objsNum, int* scene_objs_info,
	float2* scene_uvs, float3* scene_normals,
	int texNum, int* tex_wh, float3* tex_data,
	int type) {

	Hit bestHit;
	float3 colorMask = make_float3(1.0f);
	// accumulated color for current pixel
	float3 accuIntensity = make_float3(0.0f);
	RecursionData recuData[10];

	// hit debug
	bestHit.Init();
	if (!intersect_scene(ray, bestHit, vertsNum, scene_verts, objsNum, scene_objs_info, scene_uvs, scene_normals,
		texNum, tex_wh, tex_data))
		return make_float3(0.0f); // if miss, return black
	else
	{
		return bestHit.color;
	}
	// hit debug end

	int bounces = 0;
	while(bounces < 5 || curand_uniform(randstate) < 0.5f)
	{  
		if (bounces >= 10) break;
		bounces++;
		bestHit.Init();
		recuData[bounces].init();
		float emi = 0.0f;
		float rt = 0.0f;
		// intersect ray with scene
		if (!intersect_scene(ray, bestHit, vertsNum, scene_verts, objsNum, scene_objs_info, scene_uvs, scene_normals,
							texNum, tex_wh, tex_data))
		{
			break; // if miss STOP looping, will influnce the output of recuData since already return 
		}

		// else: we've got a hit with a scene primitive
		recuData[bounces].add(bestHit.color);
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
	int type)
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
			type);
		pixelColor = intensity;

		accumbuffer[index] += pixelColor;
	}
	float3 tempCol = accumbuffer[index] / (float)frameNum;
	//tempCol = gammaCorrect(tempCol);

	result[index] = tempCol;
}

extern "C" void launch_kernel(float3* result, float3* accumbuffer, curandState* randState, 
	unsigned int w, unsigned int h, unsigned int frame, 
	bool camAtRight, int waveNum, 
	int vertsNum, float3* scene_verts, 
	int objsNum, int* scene_objs_info,
	float2* scene_uvs, float3* scene_normals,
	int texNum, int* tex_wh, float3* tex_data,
	int type) {

	//set thread number
	int tx = 32;
	int ty = 32;

	dim3 blocks(w / tx + 1, h / ty + 1);
	dim3 threads(tx, ty);
	
	render <<<blocks, threads >>> (result, accumbuffer, randState, w, h, frame, WangHash(frame), camAtRight,
		vertsNum, scene_verts, objsNum, scene_objs_info, scene_uvs, scene_normals,
		texNum, tex_wh, tex_data,
		type);

	cudaThreadSynchronize();
	checkCUDAError("kernel failed!");
}

