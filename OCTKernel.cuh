/**********************************************************************
Filename   : OCTKernel.cuh 
Author     : Scott P. Mattison
Edited     : June 6, 2016

This file defines all the necessary parameters for utilzing the GPU for
OCT processing. The main functions for performing operations on the
GPU are defined.
**********************************************************************/
#ifndef _KERNELHEADER
#define _KERNELHEADER

#include "cuda_runtime.h"				//Required for GPU
#include "device_launch_parameters.h"	//Required for GPU
#define DLLEXPORT __declspec (dllexport)

#include <stdio.h>		//Standard input and output
#include <cufft.h>		//Required to use FFT on GPU
//#include <cuComplex.h>	//Only required to use cuda complex library

//#include <iostream> //only for debugging
//#include <time.h> //only for debugging
#include <stdlib.h> //required for standard library functions (malloc)
#include <math.h> //required for math functions
#include <stdint.h> //required for unsigned16

typedef uint16_t U16;
typedef float2 Complex;
#endif

/**********************TARGET SCOPED FUNCTIONS************************/

	/******************************************************************
	Complex Number Operations

	Below are a few simple target scoped functions for working with 
	Complex	numbers. Note that there is a cuComplex libary that may be 
	utilized in	place of these functions; however, as this program was 
	written to work in concert with another language, data needs to be 
	able to seamlessly communcate across languages; therefore, 
	non-native datatypes have been avoided as much as possible. There
	is no speed or cost benefit from utilizing this approach.
	******************************************************************/

/*
comMult

Simple function for multiplying two complex numbers
(A + Bj) * (C + Dj) = (AC - BD) + (AD + BC)j
*/
__device__ Complex comMult(Complex a, Complex b);

/*
comByFloatMult

Simple function for multiplying a complex number by a real number
(A+Bj) * (C + 0j) = AC + BCj
*/
__device__ Complex comByFloatMult(Complex a, float b);

/*
comAbs

returns the magnitude of a complex number
MAG(A+Bj) = (A^2 + B^2)^.5
*/
__device__ float comAbs(Complex input);


/********************GLOBAL GPU FUNCTIONS*****************************/

/*
padDispersion

This function performs dispersion compensation and windowing from
preloaded arrays. DC subtration is not performed.

The result is added to a predefined zero padded array. Note that this
means that either the FFT will require an separate target for processed
data or the array will have to be manually reset between runs.
There are other solutions to this problem; however, memory is currently
not the limiting factor of this GPU and this method will work better on
older GPUs.
*/
__global__ void padDispersion(U16 *src, //array from the digitizer, 'raw' interferogram
	Complex *toProcess, //array destination for further processing
	Complex *dispArray, //pre-loaded arrray of dispersion elements
	float *window, //preloaded window function
	int rawL,  //length of the raw interferogram
	int fftL,  //target length of the FFT
	int nAlines); //number of Alines being batch processed


/*
cropAbs

Despite the name, this is not a 6 pack due to working in a field.
Each aline is cropped to a region of interest and the magnitude
of the FFT is returned.

This function is only called when vibrometry is not going to be
performed
*/
__global__ void cropAbs(
	Complex *input, //Input data, typically result of the FFT
	float *output, //Destination Array
	int fftL, //Length of the Fast Fourier Transform
	int cropStart, //Start Index of the region of interest
	int cropRange, //Length of the region of interest
	int nAlines); //Number of A-lines in the batch process


/*
simpleCrop

As the name implies, this function simply crops each A-line within the
batch to a region of interest.
*/
__global__ void simpleCrop(
	Complex *input, //Input data, result of the FFT
	Complex *cropped, //Destination array
	int fftL, //Length of Fourier Transform (or current A-line Length)
	int cropStart, //Start index of the region of interest
	int cropRange, //Length of the region of interest
	int nAlines); //Number of A-lines in the batch process

