
#define GPUFUNCSDLL extern "C" __declspec(dllexport)
#include <stdlib.h>
#include <cstdlib>
#include <stdio.h>
#include <string.h>
#include <iostream>

using namespace std;


/*
setupGPU

called to setup parameters of the GPU, only call once prior to scan 
start. If anychanges are made to relevant scan parameters, these 
parameters this must be reflected on the GPU by clearingCurrent settings
and calling this (or simply calling updateAllParameters).
*/
GPUFUNCSDLL int setupGPU(
	float *disp, 
	float *win, 
	int rAlineSize, 
	int fftS, 
	int nAlines,
	int cropS, 
	int cropEnd, 
	bool outputComplex);

/*
processMagnitude

Processes an entire dataset as once. Maximum Size is A 16383 Aline x
4096 point FFT.
*/
GPUFUNCSDLL int processMagnitude(
	U16 *rawSpectrogram, 
	float *processedData);

/*
processComplex

Processes an entire dataset as once. Maximum Size is A 16383 Aline x
4096 point FFT.
*/
GPUFUNCSDLL int processComplex(
	U16 *rawSpectrogram,
	Complex *processedData);

GPUFUNCSDLL int processComplexInterped(float *rawSpectrogram, float *processedData);
GPUFUNCSDLL int processMagnitudeInterped(float *rawSpectrogram, float *processedData);

/*
pintError

Returns the current Error state of the device. 0 is no error
*/
GPUFUNCSDLL int pingError();

/*
getDW

Returns the dispersion and window function for debugging
*/
GPUFUNCSDLL int getDW(Complex *disp, float *win);

/*
updateAllParameters

Allows the user to update current GPU settings. 
This function is SLOW and SHOULD NOT BE USED REGULARLY
*/
GPUFUNCSDLL int updateAllParameters(
	float *disp, 
	float *win, 
	int rAlineSize,
	int fftS, 
	int nAlines, 
	int cropS, 
	int cropEnd, 
	bool outputCom);

/*
updateWindow

Allows the user to send a new window Function to the GPU.
LENGTH OF WINDOW CANNOT CHANGE
*/
GPUFUNCSDLL int updateWindow(float *win);

/*
updateDispersion

Allows the user to send a new dispersion function to the GPU

LENGTH OF THE DISPERSION DATA CANNOT CHANGE
*/
GPUFUNCSDLL int updateDispersion(float *disp);


/*
endProgram

Calls the necessary functions to free memory on the GPU
*/
GPUFUNCSDLL int endProgram();

/*
clearGPUSettings

Exactly the same as endProgram, but I wanted a separate 
function call for the end of the program
*/
GPUFUNCSDLL int clearGPUSettings();

/********************************DEPRECATED****************************/

/*
processComplexAsChunks

processes a datasetas chunks, only use if Size is greater than maximum
process as chunks requires chunks to be equal sizes, and requires the
number of alines in setup to be set to chunk size
*/
GPUFUNCSDLL int processComplexAsChunks(
	U16 *rawSpectrogram, 
	Complex *processedData,
	int rawSize, 
	int cropSize, 
	int nAlines, 
	int alinesPerChunk);

/*
processMagnitudeAsChunks

processes a datasetas chunks, only use if Size is greater than maximum
process as chunks requires chunks to be equal sizes, and requires the
number of alines in setup to be set to chunk size
*/
GPUFUNCSDLL int processMagnitudeAsChunks(
	U16 *rawSpectrogram,
	float *processedData,
	int rawSize, 
	int cropSize, 
	int nAlines,
	int alinesPerChunk);

/*
funTimes

A pointless function that allows me to ensure the DLL is working
*/
GPUFUNCSDLL void funTimes(float *a, float *b);
/*
funTimes

A pointless function that allows me to ensure the DLL is working
*/
GPUFUNCSDLL int test(int a, int b);

EXTERN int clearInterped();

GPUFUNCSDLL void setDigitizer(int in);