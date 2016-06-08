#include "OCTKernel.cuh"
#define EXTERN extern "C"
#include <chrono>
#define CLOCK std::chrono::high_resolution_clock
using namespace std;
using namespace std::chrono;

/*****************************Setup Functions*************************/


/*
zeroPad

Sets an entire array to zero. In theory this is unnecessary if you are
starting from an empty array. (Worth the 2ms 1x for my peace of mind)
*/
EXTERN int zeroPad(Complex *in, int length);

/*
gpuProcessSetup

Runs once to allocated necessary memory and prepare necessary
predefined datasets
*/
extern "C" int gpuProcessSetup(
	float *disp, //Raw dispersion phase angles
	float *win, //Desired window function
	int fftSize, //Length of the FFT
	int nAlines, //number of alines
	int rAlineSize, //length of interferogram
	int cropS, //start location for the region of interest
	int cropEnd, //end point for region of interest
	bool retComplex); //flag for output type

/*
createDispersion

Allows for the generation of the complex dispersion data from
raw phase angles
*/
EXTERN int createDispersion(
	float *disp, //Raw dispersion data
	Complex *dispComp, //Calculated dispersion data
	int length); //length of interferogram


/*************************Process Functions***************************/

/*
gpuProcessComplex

This function takes input from the digitizer and outputs the region of
interest from each a-line as an array of complex numbers
*/
EXTERN int gpuProcessComplex(
	U16 *raw, //Raw Interferogram
	Complex *output //Cropped complex output
	);

/*
gpuProcessMagnitude

This function takes input from the digitizer and outputs the magnitude
of the region of interest from each a-line as an array of floating
point numbers
*/
EXTERN int gpuProcessMagnitude(U16 * raw, float *output);

/**********************Maintainance Functions*************************/

/*
gpuUpdateAllParameters

Allows the user to result current dispersion, window, array sizes, and
return type. Only way to change any array length or number of Alines
*/
EXTERN int gpuUpdateAllParameters(
	float *disp, //New raw dispersion data
	float *win, //New window function
	int rAlineSize, //new RawAlineSize
	int fftS, //New FFT length
	int nAlines, //new Number of Alines
	int cropS,  //new Crop start location
	int cropEnd, //new crop End location
	bool rtrnComplex);

/*
gpuGetDW

Debugging function. Allows probing of data currently in GPU
*/
EXTERN int gpuGetDW(
	Complex *disp, //Array for holding output dispersion
	float *win);


/*
gpuUpdateWindow

replaces the current window with a new one.

Length cannot change.
*/
extern "C" int gpuUpdateWindow(float *window);


/*
gpuUpdateDispersion

Allows the user to load a new raw disperison file, calculates the
new dispersion and replaces the current dispersion in memory

Length cannot change
*/
extern "C" int gpuUpdateDispersion(float *disp);

/*
gpuFlipOutputType

If the data output is currently complex, it becomes float
and vice-versa
*/
EXTERN int gpuFlipOutputType();

/*
gpuGetError

Simply returns the error state of GPU
*/
EXTERN int gpuGetError();

/*
endProgram

A function for ending the program. Currently simply calls clearGPU
*/
EXTERN int gpuEndProgram();

/*
endProgram

Deallocates memory and destroys the current FFT plan. Must be run prior
to ending software*/
EXTERN int gpuClear();

/*
gpuProcessSetupInterped

Runs once to allocated necessary memory and prepare necessary
predefined datasets. This function was written for using floating
point data coming from the FPGA
*/
extern "C" int gpuProcessSetupInterped(
	float *disp, //Raw dispersion phase angles
	float *win, //Desired window function
	int fftSize, //Length of the FFT
	int nAlines, //number of alines
	int rAlineSize, //length of interferogram
	int cropS, //start location for the region of interest
	int cropEnd, //end point for region of interest
	bool retComplex); //flag for output type

/*************************Process Functions***************************/
/*

processComplexInterped

This function takes input from the digitizer and outputs the magnitude
of the region of interest from each a-line as an array of floating
point numbers. This function assumes you are using interpolated data
coming from the FPGA
*/

EXTERN int gpuProcessComplexInterped(
	float *raw,
	Complex *output);

/*

processMagnitudeInterped

This function takes input from the digitizer and outputs the magnitude
of the region of interest from each a-line as an array of floating
point numbers. This function assumes you are using interpolated data
coming from the FPGA
*/

EXTERN int gpuProcessMagnitudeInterped(
	float * raw,
	float *output);

EXTERN int clearInterped();

EXTERN int gpuUpdateAllParametersInterped(
	float *disp, //New raw dispersion data
	float *win, //New window function
	int rAlineSize, //new RawAlineSize
	int fftS, //New FFT length
	int nAlines, //new Number of Alines
	int cropS,  //new Crop start location
	int cropEnd, //new crop End location
	bool rtrnComplex);