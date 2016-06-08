/**********************************************************************
Filename   : OCTKernel.cu
Author     : Scott P. Mattison
Edited     : June 6, 2016

This file includes the main functions for performing operations on the
GPU. Note that FFT is not defined here as the cuFFT library is simply
called from the host.
**********************************************************************/
#include "OCTKernel.cuh"


/**********************TARGET SCOPED FUNCTIONS************************/

/**********************************************************************
Complex Number Operations

Below are a few simple target scoped functions for working with Complex
numbers. Note that there is a cuComplex libary that may be utilized in
place of these functions; however, as this program was written to work
in concert with another language, data needs to be able to seamlessly
communcate across languages; therefore, non-native datatypes have been 
avoided as much as possible. There is no speed or cost benefit from 
utilizing this approach.
**********************************************************************/

/*
comMult

Simple function for multiplying two complex numbers
(A + Bj) * (C + Dj) = (AC - BD) + (AD + BC)j
*/
__device__ Complex comMult(Complex a, Complex b){
	Complex output; //holder for output data
	output.x = a.x *b.x - a.y * b.y; //Real part of the new number (j^2 = -1 hence the minus
	output.y = a.x * b.y + a.y * b.x; //imaginary part of the new number
	return output; //return the result
} //end comMult

/*
comByFloatMult

Simple function for multiplying a complex number by a real number
(A+Bj) * (C + 0j) = AC + BCj
*/
__device__ Complex comByFloatMult(Complex a, float b){
	Complex output;          //Holder for output data
	output.x = a.x * b;      //set the real part
	output.y = a.y * b;      //set the imaginary part
	return output;           //return the output
} //end comByFloatMult

/*
comAbs

returns the magnitude of a complex number
MAG(A+Bj) = (A^2 + B^2)^.5
*/
__device__ float comAbs(Complex input){
	//returns the square root of the sum of the squares
	return sqrt((input.x * input.x) + (input.y * input.y)); 
} //end comAbs

	
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
__global__ void padDispersion(
	U16 *src, //array from the digitizer, 'raw' interferogram
	Complex *toProcess, //array destination for further processing
	Complex *dispArray, //pre-loaded arrray of dispersion elements
	float *window, //preloaded window function
	int rawL,  //length of the raw interferogram
	int fftL,  //target length of the FFT
	int nAlines) //number of Alines being batch processed
{
	//calculate the index of the current thread
	int thrId = blockIdx.x * blockDim.x + threadIdx.x;

	//ensure the current thread is valid for the target range
	if (thrId < (nAlines*rawL)){

		//determine the location the data should be written
		int outId = int(thrId / rawL)*fftL + (thrId%rawL);

		//determine which location to read for dispersion and windowing
		int rId = thrId % rawL;

		//convert the input signal from U16 to raw.
		//temp used to reduce number of global memory reads and writes
		float temp = (float)src[thrId] / (float)65535;

		//A placeholder is used to reduce number of global memory reads
		//and writes. Dispersion compensation is performed.
		Complex output = comByFloatMult(dispArray[rId], temp);

		//Windowing is performed.
		output = comByFloatMult(output, window[rId]);

		//result is written to an predefined, zero-paddedarray
		toProcess[outId] = output;


		//The following code section is deprecated following the switch
		//from cuFloatComplex to float2
		//toProcess[outId] = make_cuFloatComplex(output.x, output.y);
	} //end if
} //end padDispersion

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
__global__ void padDispersionInterped(
	float *src, //array the FPGA, interpolated interferogram
	Complex *toProcess, //array destination for further processing
	Complex *dispArray, //pre-loaded arrray of dispersion elements
	float *window, //preloaded window function
	int rawL,  //length of the raw interferogram
	int fftL,  //target length of the FFT
	int nAlines)
{
	//calculate the index of the current thread
	int thrId = blockIdx.x * blockDim.x + threadIdx.x;

	//ensure the current thread is valid for the target range
	if (thrId < (nAlines*rawL)){

		//determine the location the data should be written
		int outId = int(thrId / rawL)*fftL + (thrId%rawL);

		//determine which location to read for dispersion and windowing
		int rId = thrId % rawL;

		//convert the input signal from U16 to raw.
		//temp used to reduce number of global memory reads and writes
		float temp = src[thrId];

		//A placeholder is used to reduce number of global memory reads
		//and writes. Dispersion compensation is performed.
		Complex output = comByFloatMult(dispArray[rId], temp);

		//Windowing is performed.
		output = comByFloatMult(output, window[rId]);

		//result is written to an predefined, zero-paddedarray
		toProcess[outId] = output;

	}
		//The following code section is deprecated following the switch
		//from cuFloatComplex to float2
		//toProcess[outId] = make_cuFloatComplex(output.x, output.y);
} //End PadDispersionInterped
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
	int nAlines) //Number of A-lines in the batch process
{
	//Calculate current thread index
	int thrId = threadIdx.x + blockIdx.x*blockDim.x;

	//Ensure thread index is valid
	if (thrId < (cropRange*nAlines)){
		//determine the index to read from the input dataset
		int cropId = int(thrId / cropRange) * fftL + cropStart + (thrId%cropRange);

		//obtain the correct datapoint
		Complex temp = input[cropId];

		//write the magnitude of the Complex datapoint to the output array
		output[thrId] = comAbs(temp);
	} //end if

} //end cropAbs


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
	int nAlines) //Number of A-lines in the batch process
{
	//Calculate current thread index.
	int thrId = threadIdx.x + blockIdx.x*blockDim.x;

	//Ensure thread index is within the valid range
	if (thrId < (cropRange*nAlines)){
		//determine the index to read from the input dataset
		int cropId = int(thrId / cropRange) * fftL + cropStart + (thrId%cropRange);

		//obtain the correct input datapoint
		Complex temp = input[cropId];
		//write the magnitude of the Complex datapoint to the output array
		cropped[thrId] = temp;
	}// end if
} //end simpleCrop