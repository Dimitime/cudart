/* Raytracer using Cuda 5.5 and OpenGL 4.4 and GLSL 4.4
 *
 * Made by Dimitar Dinev
 *
 */

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#include <time.h>

#include <GL/glew.h>
#include <GL/glut.h>
/*
#include "gl\glew.h"
#include "gl\glut.h"*/
#include "glm/glm.hpp"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720
#define VIEWX WINDOW_WIDTH/2
#define VIEWY WINDOW_HEIGHT/2
#define FOCAL_LENGTH 1000

#define EPSILON 1e-4f

#define NUMBER_SPHERES 7
#define NUMBER_PLANES 1
#define NUMBER_LIGHTS 2

#define REFLECTION_DEPTH 5

//Error macro
#define GPU_CHECKERROR( err ) (gpuCheckError( err, __FILE__, __LINE__ ))
static void gpuCheckError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
               file, line );
        exit( EXIT_FAILURE );
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////Class definitions. I wanted to put these in seperate files, but the nvidia linker didn't like that. Thanks, Cuda.//////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class Ray {
public:
	__device__ __host__ __forceinline__ Ray() {
		origin = glm::vec3(0.0, 0.0, 0.0);
		direction = glm::vec3(1.0, 1.0, 1.0);
	}

	__device__ __host__ __forceinline__ Ray(glm::vec3 o, glm::vec3 d) {
		origin = o;
		direction = d;
	}
	__device__ __host__ __forceinline__ ~Ray() {}


	glm::vec3 origin;    
    glm::vec3 direction;
};

class Sphere {
public:
	__device__ __host__ Sphere() {
	
		center = glm::vec3(0.0,0.0,0.0);
		radius = 1.0;
		ambient_color = glm::vec3(1.0f, 0.1f, 0.5f);
		diffuse_color = glm::vec3(1.0f, 0.1f, 0.5f);
		specular_color = glm::vec3(1.0f, 0.1f, 0.5f);
		phong = 1.0f;
		reflectance = 0.5f;
	}

	__device__ __host__ Sphere(glm::vec3 c, float r, float ref, float a,glm::vec3 ka, glm::vec3 kd, glm::vec3 ks){
		center = c;
		radius = r;
		ambient_color = ka;
		diffuse_color = kd;
		specular_color = ks;
		phong = a;
		reflectance = ref;
	}

	__device__ __host__ ~Sphere(){}

	__device__ __host__ __forceinline__ bool intersection(Ray ray, glm::vec3 &point, glm::vec3 &normal, float &tt){
		glm::vec3 ec = ray.origin-center;
		glm::vec3 d = ray.direction;
		
		float disc = glm::dot(d,ec)*glm::dot(d,ec) - glm::dot(d,d) * (glm::dot(ec,ec) - radius*radius);
		float t = 0.0f;
		
		//If the discriminant is negative, the ray and sphere do not intersect
		if (disc < 0)
			return false;
			
		//if discriminant is zero, the ray intersects at exactly 1 point
		else if (disc == 0)
			t = glm::dot(-d,ec)/glm::dot(d,d);
		
		//If discriminant is positive, 2 solutions. We find the one with the smallest t (first point of intersection)
		else {
			float t1 = ( glm::dot(-d,ec)+sqrt(disc) )/glm::dot(d,d);
			float t2 = ( glm::dot(-d,ec)-sqrt(disc) )/glm::dot(d,d);
			
			//both possibilities are out of range
			if ( (t1 < 0)&&(t2 < 0) )
				return false;
			//t1 is in range, but t2 is not. we take t1.
			else if ( (t2 < 0) )
				t=t1;
			//otherwise, we take the smaller root
			else
				t=t2;
		}

		if (t < 0)
		    return false;
		
		//The point of intersection	
		point = ray.origin + t*ray.direction;
		
		//The normal of intersection
		normal = (point-center)/radius;

		tt=t;
		return true;
	}
//private:
	glm::vec3 center;
    float radius;
	float phong;
	float reflectance;

    glm::vec3 ambient_color;
    glm::vec3 diffuse_color;
    glm::vec3 specular_color;
};

class Plane {
public:
	__device__ __host__ Plane() {
	
		normal = glm::vec3(0.0f, 1.0f,0.0f);
		point = glm::vec3(0.0f,-1.0f,0.0f);
		ambient_color = glm::vec3(0.5f,	0.5f, 0.5f);
		diffuse_color = glm::vec3(0.5f, 0.5f, 0.5f);
		specular_color = glm::vec3(0.5f, 0.5f, 0.5f);
		phong = 1.0f;
		reflectance = 0.8f;
	}

	__device__ __host__ Plane(glm::vec3 n, glm::vec3 p, float a, float r, glm::vec3 ka, glm::vec3 kd, glm::vec3 ks){
		normal = glm::normalize(n);
		point = p;
		ambient_color = ka;
		diffuse_color = kd;
		specular_color = ks;
		phong = a;
		reflectance = r;
	}

	__device__ __host__ ~Plane(){}

	__device__ __host__ __forceinline__ bool intersection(Ray ray, glm::vec3 &ppoint, glm::vec3 &pnormal, float &tt){

		glm::vec3 pe = glm::normalize(point - ray.origin);
		float dn = glm::dot(ray.direction, normal);


		//ray is parallel to the plane
		if (dn == 0)
			return false;

		float t = (glm::dot(pe, normal) )/dn;
		
		//std::cout << "How about here? t= " << t <<std::endl;
		//if t<0, point is behind the camera
		if (t < 0)
		    return false;
		//std::cout << "How about here?" << std::endl;

		//The point of intersection
		ppoint = ray.origin + t*ray.direction;
		
		//The normal of intersection
		pnormal = normal;
		tt=t;
		return true;
	}

//private:
	glm::vec3 normal;
    glm::vec3 point;
	float phong;
	float reflectance;

    glm::vec3 ambient_color;
    glm::vec3 diffuse_color;
    glm::vec3 specular_color;
};

class Camera {
public:
	__device__ __host__ Camera() {
		position = glm::vec3(0.0f, 0.0f, 1.0f);
		up = glm::vec3(0.0f, 1.0f, 0.0f);
		right = glm::vec3(1.0f, 0.0f, 0.0f);
		back = glm::vec3(0.0f, 0.0f, 1.0f);
		tfovx = tan((float)M_PI/8);
		tfovy = tan((float)M_PI/8);
	}

	__device__ __host__ Camera(glm::vec3 p, glm::vec3 upv, glm::vec3 target, float fovx, float fovy) {
		position = p;
		up = glm::normalize(upv);

		back = glm::normalize( p-target );
		right = glm::normalize( glm::cross(upv,back) );
		up = glm::normalize( glm::cross(back,right) );

	
		tfovx = tan(fovx/2);
		tfovy = tan(fovy/2);
	}
	__device__ __host__ ~Camera() {}

	__device__ __host__ void generateRay(int i, int j, float viewx, float viewy, Ray &outRay) {

		float u = -viewx + (2*viewx) * ( (float)i + 0.5f ) / WINDOW_WIDTH;
		float v = -viewy + (2*viewy) * ( (float)j + 0.5f ) / WINDOW_HEIGHT;

		outRay.origin = position;
		glm::vec3 temp(-(float)FOCAL_LENGTH*back + u*right + v*up);
		//std::cout << "Temp: (" << temp.x << "," << temp.y << "," <<temp.z << ")" << std::endl;
		outRay.direction = glm::normalize(temp);//position + u*right + v*up;//back + u*right + v*up;
	}

	glm::vec3 position;
    glm::vec3 up;
	glm::vec3 right;
	glm::vec3 back;
	float tfovx;
	float tfovy;
};

class PointLight {
public:
	__device__ __host__ PointLight() {
		position = glm::vec3(0.0f, 1.0f, 0.0f);
		ka = glm::vec3(0.5, 0.5, 0.5);
		kd = glm::vec3(0.5, 0.5, 0.5);
		ks = glm::vec3(1.0, 1.0, 1.0);
	}

	__device__ __host__ PointLight(glm::vec3 p, glm::vec3 nka, glm::vec3 nkd, glm::vec3 nks) {
		position = p;
		ka = nka;
		kd = nkd;
		ks = nks;
		
	}
	__device__ __host__ ~PointLight() {}

	glm::vec3 position;
    glm::vec3 ka;
	glm::vec3 kd;
	glm::vec3 ks;
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///Global variables, mainly arrays. ////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
GLuint programID;
GLuint VertexArrayID;
GLuint h_vbo;
GLuint h_v_vbo;
int sphere_id = -1;

struct cudaGraphicsResource* d_vbo;

Sphere h_spheres[NUMBER_SPHERES];
Plane h_planes[NUMBER_PLANES];
PointLight h_lights[NUMBER_LIGHTS];
Sphere* d_spheres;
Plane* d_planes;
PointLight* d_lights;

Camera camera( glm::vec3(0.0f, 1.0f, 1.0f),
	           glm::vec3(0.0f, 1.0f, 0.0f),
			   glm::vec3(0.0f, 1.0f, 0.0f),
			   (float)M_PI_4,
			   (float)M_PI_4);

//An array used for testing. REMOVE
glm::vec3 *g_vertex_buffer_data;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///Functions called by the kernel./ ////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Calculates the shading on a surface using blinn-phong shading
__device__ __host__ glm::vec3 bphong(glm::vec3 int_point, glm::vec3 normal, glm::vec3 ka, glm::vec3 kd, glm::vec3 ks, float phong, PointLight* lights, Ray ray) {
		glm::vec3 n = glm::normalize(normal);			
		glm::vec3 v = glm::normalize(-ray.direction);
			
		glm::vec3 fcolor(0.0f,0.0f,0.0f);
			
		for (int i=0; i<NUMBER_LIGHTS; i++) {
			glm::vec3 Ia = lights[i].ka;
			glm::vec3 Id = lights[i].kd;
			glm::vec3 Is = lights[i].ks;

			glm::vec3 l = glm::normalize( lights[i].position - int_point );
				
			//Diffuse component
			float cosd = glm::dot(l,n);
			if (cosd < 0)
				cosd = 0;

			//Calculate the reflection vector
			glm::vec3 r = glm::reflect(-l, n);
			//glm::vec3 h = glm::normalize(l+v);

			float coss = glm::dot(r,v);
			if (coss < 0)
				coss = 0;					
			float p = glm::pow(coss,phong);
			 
			//Add the diffuse component = brdf * Is
			fcolor.x += Ia.x*ka.x +  Id.x*glm::clamp(kd.x*cosd, 0.0f, 1.0f) +  Is.x*glm::clamp(ks.x*p, 0.0f, 1.0f);
			fcolor.y += Ia.x*ka.y +  Id.y*glm::clamp(kd.y*cosd, 0.0f, 1.0f) + Is.y*glm::clamp(ks.y*p, 0.0f, 1.0f);
			fcolor.z += Ia.x*ka.z +  Id.z*glm::clamp(kd.z*cosd, 0.0f, 1.0f) + Is.z*glm::clamp(ks.z*p, 0.0f, 1.0f);
		}
			
		return fcolor;
}

//Calculates the first intersection of Ray ray
__device__ __host__ void getFirstIntersection(Sphere* spheres, Plane* planes, Ray ray, bool &is_sphere, int &sk, bool &is_plane, int &pk, glm::vec3 &opoint, glm::vec3 &onormal) {
	
	//variables used to depth-test
	glm::vec3 closest_spoint;
	glm::vec3 closest_snormal;
	glm::vec3 closest_ppoint;
	glm::vec3 closest_pnormal;
	bool int_sphere = false;
	bool int_plane = false;
	int closest_sk = -1;
	int closest_pk = -1;
	float ts = 1000.0f;
	float tp = 1000.0f;

	//placeholder variables that will be overwritten by the intersection functions
	float temp1 = 0.0f;
	float temp2 = 0.0f;	
	glm::vec3 point;
	glm::vec3 normal;	
	
	//Loop through the spheres
	for (int k=0; k<NUMBER_SPHERES; k++) {
		if ( spheres[k].intersection(ray, point, normal, temp1) ) {
			if ( (temp1 < ts) ) {
				ts = temp1;
				closest_spoint = point;
				closest_snormal = normal;
				int_sphere = true;
				closest_sk = k;
			}
		}
	}

	//Loop through the planes
	for (int k=0; k<NUMBER_PLANES; k++) {
		if (planes[k].intersection(ray, point, normal,temp2) ) {
			if (temp2 < tp) {
				tp = temp2;
				closest_ppoint = point;
				closest_pnormal = normal;
				int_plane = true;
				closest_pk = k;
			}
		}
	}

	//Compare the t values and display the closest object
	if ( (int_sphere && !int_plane) || ( int_sphere && int_plane && ts <= tp ) ) {
		opoint = closest_spoint;
		onormal = closest_snormal;
		sk = closest_sk;
		is_sphere =  true;
		is_plane =  false;
	}
	else if ( (int_plane && !int_sphere) || ( int_sphere && int_plane && tp < ts ) ) {
		opoint = closest_ppoint;
		onormal = closest_pnormal;
		pk = closest_pk;
		is_sphere =  false;
		is_plane =  true;
	}
	else {
		is_sphere = false;
		is_plane = false;
	}
}

//Calculate the reflected color recursively
__device__ __host__ glm::vec3 reflectedColor(Sphere* spheres, Plane* planes, PointLight* lights, Ray ray, glm::vec3 point, glm::vec3 normal, int level)
{
	if (level < REFLECTION_DEPTH) {
		bool is_rsphere = false;
		bool is_rplane = false;
		int sk = -1;
		int pk = -1;
		glm::vec3 rpoint;
		glm::vec3 rnormal;
		glm::vec3 refcolor(0.0f,0.0f,0.0f);
			//Generate a reflection ray
			glm::vec3 r = glm::normalize(glm::reflect(ray.direction, normal));
			Ray reflectRay(point+r*EPSILON, r);
			getFirstIntersection(spheres, planes, reflectRay, is_rsphere, sk, is_rplane, pk, rpoint, rnormal);

			if (is_rsphere && !is_rplane)
				refcolor = spheres[sk].reflectance * bphong(point, normal, spheres[sk].ambient_color, spheres[sk].diffuse_color, spheres[sk].specular_color, spheres[sk].phong, lights, reflectRay)
				+ spheres[sk].specular_color*reflectedColor(spheres, planes, lights, reflectRay, rpoint, rnormal, level+1);
			else if (is_rplane && !is_rsphere)
				refcolor = planes[pk].reflectance * bphong(point, normal, planes[pk].ambient_color, planes[pk].diffuse_color, planes[pk].specular_color, planes[pk].phong, lights, reflectRay)
				+ planes[pk].specular_color*reflectedColor(spheres, planes, lights, reflectRay, point, normal, level+1);
			return refcolor;
	}
	else
		return glm::vec3(0.0f,0.0f,0.0f);
}

//The main ray tracing kernel
__global__ void rayTrace_kernel(glm::vec3* vbuffer, Sphere* spheres, Plane* planes, PointLight* lights, Camera camera) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int index =j*WINDOW_WIDTH+i;

	if (i>=(WINDOW_WIDTH) || j>=(WINDOW_HEIGHT))
		return;

	//The screen x,y coordinates mapped from window-space to (-1,1)(-1,1)
	float x = (float)i/(WINDOW_WIDTH/2.0) - 1.0;
	float y = (float)j/(WINDOW_HEIGHT/2.0) - 1.0;
	vbuffer[index] = glm::vec3( x, y, 0.0);

	//Loop through the surfaces to find intersections and calculate colors
	glm::vec3 outColor(0.0,0.0,0.0);
	Ray ray;
	camera.generateRay(i,j,VIEWX,VIEWY, ray);

	bool is_sphere = false;
	bool is_plane = false;
	int sk = -1;
	int pk = -1;

	glm::vec3 point;
	glm::vec3 normal;

	//Get the first intersection for direct lighting
	getFirstIntersection(spheres, planes, ray, is_sphere, sk, is_plane, pk, point, normal );
	
	if (is_sphere && !is_plane)	
		outColor = (1.0f -spheres[sk].reflectance) * bphong(point, normal, spheres[sk].ambient_color, spheres[sk].diffuse_color, spheres[sk].specular_color, spheres[sk].phong, lights, ray);
	else if (is_plane && !is_sphere)
		outColor = (1.0f -planes[pk].reflectance)* bphong(point, normal, planes[pk].ambient_color, planes[pk].diffuse_color, planes[pk].specular_color, planes[pk].phong, lights, ray);
	

	//Loop through the light sources and make a shadow for each one
	for (int k=0; k<NUMBER_LIGHTS; k++) {
		glm::vec3 l = glm::normalize( lights[k].position - point );

		bool isShadow = false;
		bool is_ssphere = false;
		bool is_splane = false;
		Ray shadowRay(point+l*EPSILON, l);
		
		glm::vec3 tnormal;
		glm::vec3 tpoint;
		//Get the first intersection for the shadow ray
		getFirstIntersection(spheres, planes, shadowRay, is_ssphere, sk, is_splane, pk, tpoint, tnormal);
		if (is_ssphere || is_splane)
			isShadow = true;
		if (isShadow)
			outColor *= .2f;
	}
/*	bool is_rsphere = false;
	bool is_rplane = false;

	glm::vec3 rpoint;
	glm::vec3 refcolor(0.0f,0.0f,0.0f);
	if (is_sphere || is_plane) {
		//Generate a reflection ray
		glm::vec3 r = glm::normalize(glm::reflect(ray.direction, normal));
		Ray reflectRay(point+r*EPSILON, r);
		getFirstIntersection(spheres, planes, reflectRay, is_rsphere, sk, is_rplane, pk, point, normal);

		if (is_rsphere && !is_rplane)
			refcolor = spheres[sk].reflectance * bphong(rpoint, normal, spheres[sk].ambient_color, spheres[sk].diffuse_color, spheres[sk].specular_color, spheres[sk].phong, lights, reflectRay);
		else if (is_rplane && !is_rsphere)
			refcolor = planes[pk].reflectance * bphong(rpoint, normal, planes[pk].ambient_color, planes[pk].diffuse_color, planes[pk].specular_color, planes[pk].phong, lights, reflectRay);
	}
*/	glm::vec3 refcolor = glm::vec3(0.0f, 0.0f, 0.0f);

	if (is_sphere || is_plane)
		refcolor = reflectedColor(spheres, planes, lights, ray, point, normal, 0);

	outColor += refcolor;

	vbuffer[index+WINDOW_WIDTH*WINDOW_HEIGHT] = outColor;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///Glut callback functions///////// ////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void disp(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//Register the VBO to the GPU
	glm::vec3* dptr;
	GPU_CHECKERROR( cudaGraphicsGLRegisterBuffer(&d_vbo, h_vbo, cudaGraphicsRegisterFlagsNone) );
	GPU_CHECKERROR( cudaGraphicsMapResources(1,&d_vbo,0) );
	size_t num_bytes;
	GPU_CHECKERROR( cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, d_vbo) );
	
    // configure kernel parameters
	dim3 block_size;
    block_size.x = 32;
    block_size.y = 32;
    dim3 grid_size;
    grid_size.x = (unsigned int)ceil (WINDOW_WIDTH / (float)block_size.x);
    grid_size.y = (unsigned int)ceil (WINDOW_HEIGHT / (float)block_size.y);
	
	//since the spheres parameters can change, we need to re-copy the arry every time
    GPU_CHECKERROR( cudaMemcpy ( (void *) d_spheres,
                  (void *) h_spheres,
                   NUMBER_SPHERES * sizeof (Sphere),
                   cudaMemcpyHostToDevice) );

	rayTrace_kernel<<<grid_size,block_size>>>(dptr, d_spheres, d_planes, d_lights, camera);

	GPU_CHECKERROR( cudaGetLastError() );

	//unmap and unregsiter the resources from the GPU
	GPU_CHECKERROR( cudaGraphicsUnmapResources(1, &d_vbo, 0) );
	GPU_CHECKERROR( cudaGraphicsUnregisterResource(d_vbo) );

	//register the vbo with opengl
	glBindBuffer(GL_ARRAY_BUFFER, h_vbo);
	glUseProgram(programID);
	glEnableVertexAttribArray(0);

	//Vertices are indices [0...N-1] in the vbo
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);	
	glEnableVertexAttribArray(1);
	//Colors are indices [N...2N-1] in the vbo
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*) ( WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(glm::vec3)) );

	//Display the vbo
	glDrawArrays(GL_POINTS, 0, WINDOW_HEIGHT*WINDOW_WIDTH);

	//unbind everything
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(0);
	glUseProgram(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glFlush();
	glutSwapBuffers();
}

static void idle( void )
{
	int timems = glutGet(GLUT_ELAPSED_TIME);
	if (timems % 100 == 0)
		glutPostRedisplay( );
}

void skeyboard( int key, int x, int y) {
	switch(key)
    {
    case GLUT_KEY_UP:
		if (sphere_id != -1)
			h_spheres[sphere_id].center.y += 0.1f;
		break;
    case GLUT_KEY_DOWN:
		if (sphere_id != -1)
			h_spheres[sphere_id].center.y -= 0.1f;
		break;
	case GLUT_KEY_RIGHT:
		if (sphere_id != -1)
			h_spheres[sphere_id].center.x += 0.1f;
		break;
	case GLUT_KEY_LEFT:
		if (sphere_id != -1)
			h_spheres[sphere_id].center.x -= 0.1f;
		break;
    }
}

void keyboard( unsigned char key, int x, int y )
{
    switch(key)
    {
    case 27:
        exit(0);
        break;
    case 'w':
		if (sphere_id != -1)
			h_spheres[sphere_id].center.z -= 0.1f;
		break;
    case 's':
		if (sphere_id != -1)
			h_spheres[sphere_id].center.z += 0.1f;
		break;
    }
}

void mouse( int button, int state, int x, int y) {
	Ray ray;
	camera.generateRay(x,WINDOW_HEIGHT-y,VIEWX,VIEWY, ray);
	int k=0;
	bool intersected = false;
	do{
		glm::vec3 point;
		glm::vec3 normal;
		float t = 0.0;
		if (h_spheres[k].intersection(ray, point, normal,t) ) {
			//std::cout << "intersected with sphere" << k << std::endl;
			intersected = true;
		}
		else
			k++;
	} while ( k<NUMBER_SPHERES && !intersected);
	if (!intersected)
		sphere_id = -1;
	else
		sphere_id = k;
	//std::cout << "Clicked!" << x << ", " << y << " on sphere# " << sphere_id << std::endl;
}

void create_scene() {
	h_spheres[0].radius = .2f;
	h_spheres[0].center = glm::vec3(0.0f, 0.4f, -5.0f);
	h_spheres[0].ambient_color = glm::vec3(0.135f, 0.2225f, 0.1575f);
	h_spheres[0].diffuse_color = glm::vec3(0.54f, 0.89f, 0.63f);
	h_spheres[0].specular_color = glm::vec3(0.316228f, 0.316228f, 0.316228f);
	h_spheres[0].phong = 76.8f;

	h_spheres[1].radius = .4f;
	h_spheres[1].center = glm::vec3(0.5f, 0.5f, -3.0f);
	h_spheres[1].ambient_color = glm::vec3(0.1745f, 0.01175f, 0.01175f);
	h_spheres[1].diffuse_color = glm::vec3(0.61424f, 0.04136f, 0.04136f);
	h_spheres[1].specular_color = glm::vec3(0.727811f, 0.626959f, 0.626959f);
	h_spheres[1].phong = 76.8f;

	float x = 0.2f;
	float y = 1.0f;
	float z = -3.0f;
	for (int i=0; i<5; i++) {
		x += 0.1f;
		h_spheres[2+i].center = glm::vec3(x, y, z);
		h_spheres[2+i].radius = .05f;
	}
	h_lights[0].position = glm::vec3(-20.0f, 10.5f, 20.0f);
	h_lights[1].position = glm::vec3(1.0f, 10.5f, 20.0f);
}

//Loading shelders
GLuint LoadShaders(const char * vertex_file_path, const char * fragment_file_path){
 
    // Create the shaders
    GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);
 
    // Read the Vertex Shader code from the file
    std::string VertexShaderCode;
    std::ifstream VertexShaderStream(vertex_file_path, std::ios::in);
    if(VertexShaderStream.is_open())
    {
        std::string Line = "";
        while(std::getline(VertexShaderStream, Line))
            VertexShaderCode += "\n" + Line;
        VertexShaderStream.close();
    }
 
    // Read the Fragment Shader code from the file
    std::string FragmentShaderCode;
    std::ifstream FragmentShaderStream(fragment_file_path, std::ios::in);
    if(FragmentShaderStream.is_open()){
        std::string Line = "";
        while(std::getline(FragmentShaderStream, Line))
            FragmentShaderCode += "\n" + Line;
        FragmentShaderStream.close();
    }
 
    GLint Result = GL_FALSE;
    int InfoLogLength;
 
    // Compile Vertex Shader
    printf("Compiling shader : %s\n", vertex_file_path);
    char const * VertexSourcePointer = VertexShaderCode.c_str();
    glShaderSource(VertexShaderID, 1, &VertexSourcePointer , NULL);
    glCompileShader(VertexShaderID);
 
    // Check Vertex Shader
    glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    std::vector<char> VertexShaderErrorMessage(InfoLogLength);
    glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
    fprintf(stdout, "%s\n", &VertexShaderErrorMessage[0]);
 
    // Compile Fragment Shader
    printf("Compiling shader : %s\n", fragment_file_path);
    char const * FragmentSourcePointer = FragmentShaderCode.c_str();
    glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer , NULL);
    glCompileShader(FragmentShaderID);
 
    // Check Fragment Shader
    glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    std::vector<char> FragmentShaderErrorMessage(InfoLogLength);
    glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
    fprintf(stdout, "%s\n", &FragmentShaderErrorMessage[0]);
 
    // Link the program
    fprintf(stdout, "Linking program\n");
    GLuint ProgramID = glCreateProgram();
    glAttachShader(ProgramID, VertexShaderID);
    glAttachShader(ProgramID, FragmentShaderID);
    glLinkProgram(ProgramID);
 
    // Check the program
    glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
    glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    std::vector<char> ProgramErrorMessage( glm::max(InfoLogLength, int(1)) );
    glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
    fprintf(stdout, "%s\n", &ProgramErrorMessage[0]);
 
    glDeleteShader(VertexShaderID);
    glDeleteShader(FragmentShaderID);
 
    return ProgramID;
}

void init() {
	GPU_CHECKERROR( cudaGLSetGLDevice(0) );

	//Increase the stack size so that we can use our recursive functions
	size_t myStackSize = 1024*10*REFLECTION_DEPTH;
	GPU_CHECKERROR( cudaDeviceSetLimit (cudaLimitStackSize, myStackSize) );
	
	glClearColor(0.0, 0.0, 0.0, 0.0);

	//Load the shaders
	programID = LoadShaders( "SimpleVertexShader.vertexshader", "SimpleFragmentShader.fragmentshader" );

	//Create vertex buffer object
	glGenBuffers(1, &h_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, h_vbo);

	//Initialize VBO
	unsigned int size = 2 * WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(glm::vec3);

	//std::cout << "glm size: " << sizeof(glm::vec3) << " float3 size: " << sizeof(float3) << std::endl;

	//Serial implementation
	
	time_t start_time;
	time (&start_time);
	
	g_vertex_buffer_data = (glm::vec3*)malloc(size);
/*	for (int i=0; i<WINDOW_WIDTH; i++) {
		for (int j=0; j<WINDOW_HEIGHT; j++) {
			unsigned int index = j*WINDOW_WIDTH+i;

			float x = (float)i/(WINDOW_WIDTH/2.0f) - 1.0f;
			float y = (float)j/(WINDOW_HEIGHT/2.0f) - 1.0f;
			g_vertex_buffer_data[index] = glm::vec3( x, y, 0.0);

			glm::vec3 outColor(0.0,0.0,0.0);

			Ray ray;
			camera.generateRay(i,j,VIEWX,VIEWY, ray);

			bool is_sphere = false;
			bool is_plane = false;
			int sk = -1;
			int pk = -1;

			glm::vec3 point;
			glm::vec3 normal;

			//Get the first intersection for direct lighting
			getFirstIntersection(h_spheres, h_planes, ray, is_sphere, sk, is_plane, pk, point, normal );
	
			if (is_sphere && !is_plane)	
				outColor = (1.0f -h_spheres[sk].reflectance) * bphong(point, normal, h_spheres[sk].ambient_color, h_spheres[sk].diffuse_color, h_spheres[sk].specular_color, h_spheres[sk].phong, h_lights, ray);
			else if (is_plane && !is_sphere)
				outColor = (1.0f -h_planes[pk].reflectance)* bphong(point, normal, h_planes[pk].ambient_color, h_planes[pk].diffuse_color, h_planes[pk].specular_color, h_planes[pk].phong, h_lights, ray);

			//Loop through the light sources and make a shadow for each one
			for (int k=0; k<NUMBER_LIGHTS; k++) {
				glm::vec3 l = glm::normalize( h_lights[k].position - point );

				bool isShadow = false;
				bool is_ssphere = false;
				bool is_splane = false;
				Ray shadowRay(point+l*EPSILON, l);
		
				glm::vec3 tnormal;
				glm::vec3 tpoint;
				//Get the first intersection for the shadow ray
				getFirstIntersection(h_spheres, h_planes, shadowRay, is_ssphere, sk, is_splane, pk, tpoint, tnormal);
				if (is_ssphere || is_splane)
					isShadow = true;
				if (isShadow)
					outColor *= .2f;
			}
			
			glm::vec3 refcolor = glm::vec3(0.0f, 0.0f, 0.0f);

			if (is_sphere || is_plane)
				refcolor = reflectedColor(h_spheres, h_planes, h_lights, ray, point, normal, 0);

			outColor += refcolor;
			g_vertex_buffer_data[index+WINDOW_WIDTH*WINDOW_HEIGHT] = outColor;

		}
	}
	time_t end_time;
	time (&end_time);	
	double s = difftime(end_time,start_time);
	std::cout << "Time for serial: " <<  s << std::endl;	
*/

	//Initialize the vbo to 0, as it will be computed by the GPU
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//Create the VAO	
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);	

	//Allocate the sphere arrays
	GPU_CHECKERROR( cudaMalloc ( (void **) &d_spheres, NUMBER_SPHERES * sizeof (Sphere) ) );

	//Allocate + copy the planes. Since we cannot change the plane, it only has to be copied once.
	GPU_CHECKERROR( cudaMalloc ( (void **) &d_planes, NUMBER_PLANES * sizeof (Plane) ) );
    GPU_CHECKERROR( cudaMemcpy ((void *) d_planes,
                  (void *) h_planes,
                   NUMBER_PLANES * sizeof (Plane),
                   cudaMemcpyHostToDevice)) ;

	//Allocate + copy the lights. Since we cannot change the lights, it only has to be copied once.
	GPU_CHECKERROR( cudaMalloc ( (void **) &d_lights, NUMBER_LIGHTS * sizeof (PointLight) ) );
    GPU_CHECKERROR( cudaMemcpy ( (void *) d_lights,
                  (void *) h_lights,
                   NUMBER_LIGHTS * sizeof (PointLight),
                   cudaMemcpyHostToDevice) );
}

int main (int argc, char** argv)
{
	// init glut:
	glutInit (&argc, argv);	
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutInitWindowPosition(0,0);
	glutCreateWindow("Cuda-based Ray Tracer");	
	//glutFullScreen();
    //glutGameModeString("1280x720:16@60"); glutEnterGameMode();
	//printf("OpenGL Version:%s\n",glGetString(GL_VERSION));
	//printf("GLSL Version  :%s\n",glGetString(GL_SHADING_LANGUAGE_VERSION));

	glutDisplayFunc(disp);
    glutIdleFunc( idle );
    glutKeyboardFunc( keyboard );
	glutSpecialFunc( skeyboard );
    glutMouseFunc( mouse ); 

	//glewExperimental=TRUE;
	GLenum err=glewInit();
	if(err!=GLEW_OK)
		printf("glewInit failed, aborting.\n");
    if (!glewIsSupported("GL_VERSION_2_0 ")) {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
		exit(0);
    }
	
	// init opengl:
	create_scene();
	init();

	// enter tha main loop and process events:
	glutMainLoop();

	GPU_CHECKERROR( cudaGraphicsUnregisterResource(d_vbo) );
	GPU_CHECKERROR( cudaFree(d_spheres) );
	GPU_CHECKERROR( cudaFree(d_planes) );
	GPU_CHECKERROR( cudaFree(d_lights) );
	return 0;
}
