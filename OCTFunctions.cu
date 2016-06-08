#include "OCTFunctions.cuh"
#include "Header.h"

#include <iostream>
using namespace std;
using namespace std::chrono;
/******************************Shared Variables***********************/
cufftHandle fftPlan; //instuctions for the GPU to setup FFT
int fftLength; //Desired length of the FFT
int numAlines; //Desired number of A-lines
int rawSize;   //Desired initial interferogram length
int cropRange; //Desired length of cropped Data
int cropStart; //Desired starting location of the cropped dataset


/*
This method is currently not being utilized due to (A) my inability to 
come up with a decent algorith to prevent overwrite issues and (B) the
current lack of necessity (which removes any motivation to overcome A).

In theory pinned memory is faster (and it is) However, I have issues 
with the code code writing to memory faster than I can remove it. I need 
to learn how to swap pointers to and from pinned memory, and I don't
actually know if that is possible to do.

//Only outputC or outputF is utilized, not both. (Pinned memory)
Complex *outputC; //Complex data to return to the Python Program
float *outputF;   //Floating point data to return to the python Program
U16 *input;		  //U16 Array for input from the python program
*/

//Device holders for constant values
Complex *d_disp; //Device Holder for dispersion
float *d_win;	 //Device holder for the window function

//Device holders for processing data
U16 *d_src;		//Device holder for the initial input data
Complex *d_fftIn; //Device holder for data ready for FFT processing
Complex *d_fftOut; //Device holder for FFT processed data

//Only d_cropF or c_cropC is utilized, not both
float *d_cropF; //Device holder for cropped magnitude of OCT
Complex *d_cropC; //Device holder for croppled Complex OCT data 

bool returnComplex;
int blockSize; //Size of GPU block for computation
int gridSize; //Size of GPU grid for computation



/******************************Setup Functions************************/

/*
createDispersion

Allows for the generation of the complex dispersion data from
raw phase angles
*/
EXTERN int createDispersion(
	float *disp, //Raw dispersion data
	Complex *dispComp, //Calculated dispersion data
	int length) //length of interferogram
{
	float temp; //Holder
	for (int i = 0; i < length; ++i){
		temp = disp[i]; //Reads current disperison
		dispComp[i].x = sinf(temp); //calculate 'real' disperison
		dispComp[i].y = cosf(temp); //calculate 'imaginary' disperison
	}
	return 0; //no errors (in theory)
} //End createDispersion

/*
zeroPad

Sets an entire array to zero. In theory this is unnecessary. I am
a bit overcautious
*/
EXTERN int zeroPad(Complex *in, int length){
	//For the length of the array, set all array values to zero
	for (int i = 0; i < length; ++i){
		in[i].x = 0;
		in[i].y = 0;
	}
	return 0;
} //End zeroPad

/*
gpuProcessSetup

Runs once to allocated necessary memory and prepare necessary
predefined datasets*/
extern "C" int gpuProcessSetup(
	float *disp, //Raw dispersion phase angles
	float *win, //Desired window function
	int fftSize, //Length of the FFT
	int nAlines, //number of alines
	int rAlineSize, //length of interferogram
	int cropS, //start location for the region of interest
	int cropEnd, //end point for region of interest
	bool retComplex) //flag for output type
{

	numAlines = nAlines; //set number of Alines
	rawSize = rAlineSize; //set length of raw interferogram
	cropStart = cropS; //set start location of crop
	cropRange = cropEnd - cropS; //Set length of crop
	fftLength = fftSize; //set FFTlength

	//set the output type as complex(true) or float(false)
	returnComplex = retComplex;

	//ensure crop does not try to address out of bounds
	if ((cropRange + cropStart) > fftLength){
		cropRange = fftLength - cropStart;
	}

	//Allocate an array of zeroes for zero padding of interferogram
	Complex *zeroes;
	zeroes = (Complex *)malloc(sizeof(Complex)*fftLength*numAlines);

	//Fill zero array with zeros 
	//Ensures no left over data (worth the 2 ms for a 1x process)
	zeroPad(zeroes, numAlines * fftLength);

	//Create an array for saving dispersion
	Complex *dispersion = (Complex*)malloc(rawSize*sizeof(Complex));

	//Generate the sines and cosines of the raw dispersion data
	//Saves calculation time later
	createDispersion(disp, dispersion, rawSize);

	//Allocate memory on the GPU for the source interferogram
	cudaMalloc(&d_src, rawSize*numAlines*sizeof(U16));
	//Ensure memory allocation was successful
	if (cudaGetLastError() != cudaSuccess)
		return -9; //cudaMalloc Error Code

	//Allocate memory on GPU for the disperison compensated
	//and zero padded interferogram, prior to FFT
	cudaMalloc(&d_fftIn, fftLength*nAlines*sizeof(Complex));
	//Ensure memory allocation was successful
	if (cudaGetLastError() != cudaSuccess)
		return -9; //cudaMalloc Error Code

	//Allocate Memory on GPU for output of FFT
	cudaMalloc(&d_fftOut, fftLength * nAlines*sizeof(Complex));
	//Ensure memory allocation was successful
	if (cudaGetLastError() != cudaSuccess)
		return -9; //cudaMalloc Error Code

	//If the program expects to receive complex data
	if (returnComplex){
		//Allocate memory for the cropped dataset
		cudaMalloc(&d_cropC, cropRange * nAlines * sizeof(Complex));
		//Ensure memory allocation was successful
		if (cudaGetLastError() != cudaSuccess)
			return -9; //cudaMalloc Error Code

		/*
		(NOT IN USE) Allocate host pinned memory for data transfer from device
		
		cudaMallocHost(&outputC,
			cropRange * nAlines * sizeof(Complex));


		//Ensure memory allocation was successful
		if (cudaGetLastError() != cudaSuccess)
			return -9; //cudaMalloc Error Code*/
	}
	//If the program expects to receive magnitude data (floating point)
	else {
		
		//Allocate memory for the cropped dataset
		cudaMalloc(&d_cropF, cropRange * nAlines * sizeof(float));
		//Ensure memory allocation was successful
		if (cudaGetLastError() != cudaSuccess)
			return -9; //cudaMalloc Error Code

		/* NOT CURRENTLY IN USE
		//Allocate host pinned memory for data transfer from device
		cudaMallocHost(&outputF, cropRange* nAlines * sizeof(float));
		//Ensure memory allocation was successful
		if (cudaGetLastError() != cudaSuccess)
			return -9; //cudaMalloc Error Code
		*/
	}

	/* NOT CURRENTLY IN USE
	//Allocate host pinned memory for data transfer to device
	cudaMallocHost(&input, rawSize * nAlines * sizeof(U16));

	*/
	//Allocate device memory for dispersion data
	cudaMalloc(&d_disp, rawSize*sizeof(float));
	//Ensure memory allocation was successful
	if (cudaGetLastError() != cudaSuccess)
		return -9; //cudaMalloc Error Code

	//Allocate device memory for the window function
	cudaMalloc(&d_win, rawSize*sizeof(float));
	//Ensure memory allocation was successful
	if (cudaGetLastError() != cudaSuccess)
		return -9; //cudaMalloc Error Code

	//Transfer dispersion data to device
	cudaMemcpy(d_disp, dispersion, rawSize*sizeof(float),
		cudaMemcpyHostToDevice);
	//Ensure data transfer was successful
	if (cudaGetLastError() != cudaSuccess)
		return -1; //Memcpy to device Error Code

	//Transfer zeroes for zero padding to device
	cudaMemcpy(d_fftIn, zeroes, sizeof(Complex)*fftLength*numAlines,
		cudaMemcpyHostToDevice);
	//Ensure data transfer was successful
	if (cudaGetLastError() != cudaSuccess)
		return -1; //Memcpy to device Error Code



	//Copy window function to device
	cudaMemcpy(d_win, win, rawSize*sizeof(float),
		cudaMemcpyHostToDevice);
	//Ensure data transfer was successful
	if (cudaGetLastError() != cudaSuccess)
		return -1; //Memcpy to device Error Code

	//Generate Batch FFT plan and ensure success
	if (cufftPlan1d(&fftPlan, fftLength,
		CUFFT_C2C, nAlines) != CUFFT_SUCCESS)
		return -10; //cudaMalloc Error Code

	//Deallocate useless memory
	free(dispersion);
	free(zeroes);

	return 0;
} //End gpuProcessSetup



/*************************Process Functions***************************/

/*
processComplex

This function takes input from the digitizer and outputs the region of
interest from each a-line as an array of complex numbers
*/
EXTERN int gpuProcessComplex(
	U16 *raw, //Raw interferogram
	Complex *output)//Cropped complex output
{
	int errorCode = 0; //initializes error code for debugging
	//std::copy(&raw[0], &raw[rawSize * numAlines - 1], &input[0]);
	// Attempt to copy raw interferogram to GPU memory
	cudaMemcpy(d_src, raw, rawSize * numAlines * sizeof(U16),
		       cudaMemcpyHostToDevice);

	//Ensure memory copy success
	if (cudaGetLastError() != cudaSuccess)
		errorCode = -1; //MemcpytoDevice failure error code

	//Attempt to perform dispersion compensation
	padDispersion <<<65535, 1024>>>(d_src, d_fftIn, d_disp, d_win,
		                            rawSize, fftLength, numAlines);

	//ensure disperison compensation success
	if (cudaGetLastError() != cudaSuccess)
		errorCode = -4; //padDispersion Error Code

	//Attempt to perform FFT and ensure success
	if (cufftExecC2C(fftPlan, (cufftComplex *)d_fftIn, 
		             (cufftComplex *)d_fftOut, CUFFT_FORWARD) 
					 != CUFFT_SUCCESS)
		errorCode = -3; //FFT Error Code

	//Attempt to crop the data
	simpleCrop <<<32768, 1024>>>(d_fftOut, d_cropC, fftLength, 
		                         cropStart, cropRange, numAlines);

	//Ensure crop was successful
	if (cudaGetLastError() != cudaSuccess)
		errorCode = -5; //crop Error Code

	//Attempt to copy data from device to host
	cudaMemcpy(output, d_cropC, cropRange*numAlines*sizeof(Complex),
		       cudaMemcpyDeviceToHost);
	/* NOT IN USE
	//Move from pinned memory to main memory
	//std::copy(&output[0], &outputC[cropRange*numAlines - 1], &output[0]);
	//Endure Memcpy success
	*/
	if (cudaGetLastError() != cudaSuccess)
		errorCode = -2; //MemcpytoHost error code

	return errorCode;
}

/*
processComplex

This function takes input from the digitizer and outputs the magnitude
of the region of interest from each a-line as an array of floating
point numbers
*/
EXTERN int gpuProcessMagnitude(U16 * raw, float *output){
	int errorCode = 0; //initializes error code for debugging
	//Move raw data to pinned memory
	//std::copy(&raw[0], &raw[rawSize * numAlines - 1], &input[0]);
	// Attempt to copy raw interferogram to GPU memory
	cudaMemcpy(d_src, raw, rawSize * numAlines * sizeof(U16), 
		       cudaMemcpyHostToDevice);

	//Ensure memory copy success
	if (cudaGetLastError() != cudaSuccess)
		errorCode = -1; //MemcpytoDevice failure error code

	//Attempt to perform dispersion compensation
	padDispersion <<<65535, 1024>>>(d_src, d_fftIn, d_disp, d_win,
		                            rawSize, fftLength, numAlines);
	
	//ensure disperison compensation success
	if (cudaGetLastError() != cudaSuccess)
		errorCode = -4; //padDispersion Error Code

	//Attempt to perform FFT and ensure success
	if (cufftExecC2C(fftPlan, (cufftComplex *)d_fftIn, 
		(cufftComplex *)d_fftOut, CUFFT_FORWARD) != CUFFT_SUCCESS)
		errorCode = -3; //FFT Error Code

	//Attempt to crop the data and take the magnitude
	cropAbs <<<32768, 1024>>>(d_fftOut, d_cropF, fftLength, cropStart,
		                      cropRange, numAlines);

	//Ensure crop was successful
	if (cudaGetLastError() != cudaSuccess)
		errorCode = -5; //crop Error Code

	//Attempt to copy data from device to host	
	cudaMemcpy(output, d_cropF, cropRange*numAlines*sizeof(float),
		       cudaMemcpyDeviceToHost);
	//std::copy(&outputF[0], &outputF[cropRange*numAlines - 1], &output[0]);
	//Endure Memcpy success
	if (cudaGetLastError() != cudaSuccess)
		errorCode = -2; //MemcpytoHost error code

	return errorCode;
}

/*
endProgram

Deallocates memory and destroys the current FFT plan. Must be run prior
to ending software*/
EXTERN int gpuClear(){
	cufftDestroy(fftPlan); //Destroy the CUDA plan for running FFT
	cudaFree(d_disp); //Deallocate memory
	cudaFree(d_win);//Deallocate memory
	cudaFree(d_src);//Deallocate memory
	cudaFree(d_fftIn);//Deallocate memory
	cudaFree(d_fftOut);//Deallocate memory
	if (returnComplex){ //If returning complex data
		cudaFree(d_cropC); //Free allocated device memory
		//cudaFreeHost(outputC); //Free allocated host memory
	}
	else{
		cudaFree(d_cropF);//If returning complex data
		cudaFree(d_cropF);//Free allocated device memory
		//cudaFreeHost(outputF);//Free allocated host memory
	}
	//cudaFreeHost(input);
	return 0;
}

/*
endProgram

A function for ending the program. Currently simply calls gpuClear
*/
EXTERN int gpuEndProgram(){
	gpuClear();
	return 0;
}


/*
gpuGetError

Simply returns the error state of GPU
*/
EXTERN int gpuGetError(){
	return cudaGetLastError();
}

/*
gpuFlipOutputType

If the data output is currently complex, it becomes float
and vice-versa
*/
EXTERN int gpuFlipOutputType(){
	if (returnComplex){
		returnComplex = !returnComplex; //negate returnComplex

		//Dellocate Memory
		cudaFree(d_cropC);
		//cudaFreeHost(outputC);

		
		//Allocate Memory
		cudaMalloc(&d_cropF, numAlines * cropRange * sizeof(float));
		/*
		cudaMallocHost(&outputF,
			           numAlines * cropRange * sizeof(float));
					   */
	}
	else{
		returnComplex = !returnComplex; //negate returnComplex

		//DeallocateMemory
		cudaFree(d_cropF);
		//cudaFreeHost(outputF);

		//Allocate memory
		cudaMalloc(&d_cropC, numAlines * cropRange * sizeof(Complex));
		/*cudaMallocHost(&outputC, 
			           numAlines * cropRange * sizeof(Complex));*/
	}
	return 0;
}

/*
gpuUpdateDispersion

Allows the user to load a new raw disperison file, calculates the
new dispersion and replaces the current dispersion in memory

Length cannot change
*/
extern "C" int gpuUpdateDispersion(float *disp){
	Complex *sdisp = (Complex*)malloc(sizeof(Complex) * rawSize);

	createDispersion(disp, sdisp, rawSize);
	cudaMemcpy(d_disp, sdisp, rawSize*sizeof(Complex),
		       cudaMemcpyHostToDevice);

	return cudaGetLastError();
}

/*
gpuUpdateWindow

replaces the current window with a new one.

Length cannot change.
*/
extern "C" int gpuUpdateWindow(float *window){
	cudaMemcpy(d_win, window, rawSize*sizeof(float),
		       cudaMemcpyHostToDevice);
	return cudaGetLastError();
}

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
	bool rtrnComplex)//flag to outputing complex(true) or float(false)
{
	//Clears the current parameters
	gpuClear(); 

	//Sets up the GPU with new parameters
	gpuProcessSetup(disp, win, rAlineSize, fftS, nAlines, cropS, cropEnd, rtrnComplex);
	
	//Returns current gpu status
	return cudaGetLastError();
}

/*
gpuGetDW

Debugging function. Allows probing of data currently in GPU
*/
EXTERN int gpuGetDW(
	Complex *disp, //Array for holding output dispersion
	float *win) //Array for holding output window function
{
	cudaMemcpy(disp, d_disp, rawSize * sizeof(Complex), cudaMemcpyDeviceToHost);
	cudaMemcpy(win, d_win, rawSize * sizeof(float), cudaMemcpyDeviceToHost);
	return cudaGetLastError();
}

/*
//FOR DEBUGGING PURPOSES ONLY
int main(){
	int fftL = 4096; //Desired length of the FFT
	int numA = 16383; //Desired number of A-lines
	int rawS = 2048;   //Desired initial interferogram length
	int cropR = 1024; //Desired length of cropped Data
	int cropS = 0;

	float * disp = (float *)malloc(rawS * sizeof(float));
	U16 * raw = (U16 *)malloc(rawS * sizeof(U16) * numA);
	Complex *output = (Complex *)malloc(cropR * numA * sizeof(Complex));
	float *win = (float *)malloc(rawS * sizeof(float));

	for (int i = 0; i < rawS; i++){
		disp[i] = 3.14159;
		win[i] = 10;
	}

	for (int i = 0; i < rawS * numA; i++){
		raw[i] = i % numA;
	}

	gpuProcessSetup(disp, win, fftL, numA, rawS, cropS, cropS + cropR, true);
	CLOCK::time_point t1 = CLOCK::now();
	gpuProcessComplex(raw, output);
	CLOCK::time_point t2 = CLOCK::now();

	auto duration2 = duration_cast<microseconds>(t2 - t1).count();
	cout << duration2 << " microseconds" << endl;

	
}*/