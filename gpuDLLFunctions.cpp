#include "OCTFunctions.cuh"
#include "gpuDLLFunctions.h"


int digitizer = 0; // 0 = Alazar, 1 = FPGA
/*
setupGPU

called to setup parameters of the GPU, only call once prior to scan start. If anychanges are made
to scan parameters, these parameters must be updated
*/
GPUFUNCSDLL int setupGPU(float *disp, float *win, int rAlineSize, int fftS, int nAlines, int cropS, int cropEnd, bool outputComplex){
	if (digitizer == 0){
		return (gpuProcessSetup(disp, win, fftS, nAlines, rAlineSize, cropS, cropEnd, outputComplex));
	}
	else{
		return (gpuProcessSetupInterped(disp, win, fftS, nAlines, rAlineSize, cropS, cropEnd, outputComplex));
	}
}

/*
processMagnitude

Processes an entire dataset as once. Maximum Size is A 16383 Aline x
4096 point FFT.
*/
GPUFUNCSDLL int processMagnitude(U16 *rawSpectrogram, float *processedData){
	return (gpuProcessMagnitude(rawSpectrogram, processedData));
}
/*
processComplex

Processes an entire dataset as once. Maximum Size is A 16383 Aline x
4096 point FFT.
*/
GPUFUNCSDLL int processMagnitudeInterped(float *rawSpectrogram, float *processedData){
	return (gpuProcessMagnitudeInterped(rawSpectrogram, processedData));
}

GPUFUNCSDLL void setDigitizer(int in){
	digitizer = in;
}
/*
processComplexInterped

Processes an entire dataset as once. Maximum Size is A 16383 Aline x
4096 point FFT.
*/
GPUFUNCSDLL int processComplexInterped(float *rawSpectrogram, float *outputReal, float *outputImaginary){
	return (gpuProcessComplexInterped(rawSpectrogram, outputReal, outputImaginary));
}

/*
processComplex

Processes an entire dataset as once. Maximum Size is A 16383 Aline x
4096 point FFT.
*/
GPUFUNCSDLL int processComplex(U16 *rawSpectrogram, float *outReal, float *outImaginary){
	return (gpuProcessComplex(rawSpectrogram, outReal, outImaginary));
}

/*
processMagnitudeAsChunks

SOON TO BE DEPRECATED

processes a datasetas chunks, only use if Size is greater than maximum
process as chunks requires chunks to be equal sizes, and requires the
number of alines in setup to be set to chunk size
*/
GPUFUNCSDLL int processMagnitudeAsChunks(U16 *rawSpectrogram, float *processedData, int rawSize, int cropSize, int nAlines, int alinesPerChunk){
	int repetitions = nAlines / alinesPerChunk;
	int leftOvers = nAlines % alinesPerChunk; //This should always be zero
	int error = 0;
	if (leftOvers != 0)
		error = 5;
	if (repetitions == 1 && leftOvers == 0){
		error = gpuProcessMagnitude(rawSpectrogram, processedData);
	}
	else{
		float *processed = (float *)malloc(cropSize * alinesPerChunk * sizeof(float));
		U16 *toProcess = (U16*)malloc(rawSize * alinesPerChunk * sizeof(U16));
		for (int i = 0; i < repetitions; ++i){
			memcpy(toProcess, rawSpectrogram + (i*alinesPerChunk*rawSize), rawSize * alinesPerChunk * sizeof(U16));
			error = gpuProcessMagnitude(toProcess, processed);

			memcpy(processedData + (i *cropSize * alinesPerChunk), processed, cropSize * alinesPerChunk * sizeof(float));
		}
		free(processed);
		free(toProcess);
	}
	return error;
}

/*
processComplexAsChunks

SOON TO BE DEPRECATED

processes a datasetas chunks, only use if Size is greater than maximum
process as chunks requires chunks to be equal sizes, and requires the
number of alines in setup to be set to chunk size

GPUFUNCSDLL int processComplexAsChunks(U16 *rawSpectrogram, Complex *processedData, int rawSize, int cropSize, int nAlines, int alinesPerChunk){
	int repetitions = nAlines / alinesPerChunk;
	int leftOvers = nAlines % alinesPerChunk; //This should always be zero
	int error = 0;
	if (leftOvers != 0)
		error = 5;
	if (repetitions == 1 && leftOvers == 0){
		error = gpuProcessComplex(rawSpectrogram, processedData);
	}
	else{
		Complex *processed = (Complex *)malloc(cropSize * alinesPerChunk * sizeof(float));
		U16 *toProcess = (U16*)malloc(rawSize * alinesPerChunk * sizeof(U16));
		for (int i = 0; i < repetitions; ++i){
			memcpy(toProcess, rawSpectrogram + (i*alinesPerChunk*rawSize), rawSize * alinesPerChunk * sizeof(U16));
			error = gpuProcessComplex(toProcess, processed);

			memcpy(processedData + (i *cropSize * alinesPerChunk), processed, cropSize * alinesPerChunk * sizeof(float));
		}
		free(processed);
		free(toProcess);
	}
	return error;
}*/

/*
clearGPUSettings()

Clears out allocated GPU memory so that it may be reallocated.
*/
GPUFUNCSDLL int clearGPUSettings(){
	gpuClear();
	if (digitizer != 0){
		clearInterped();
	}
	return 0;
}

GPUFUNCSDLL int endProgram(){
	gpuEndProgram();
	if (digitizer != 0){
		clearInterped();
	}
	return 0;
}

GPUFUNCSDLL int test(int a, int b){
	return a + b;
}

GPUFUNCSDLL void funTimes(float *a, float *b){
	float temp = a[0];
	a[0] = b[0];
	b[0] = temp;
}

GPUFUNCSDLL int updateDispersion(float *disp){
	return gpuUpdateDispersion(disp);
}

GPUFUNCSDLL int updateWindow(float *win){
	return gpuUpdateWindow(win);
}

GPUFUNCSDLL int updateAllParameters(float *disp, float *win, int rAlineSize, int fftS, int nAlines, int cropS, int cropEnd, bool outputCom){
	if (digitizer != 0){
		return gpuUpdateAllParametersInterped(disp, win, rAlineSize, fftS, nAlines, cropS, cropEnd, outputCom);
	}
	return gpuUpdateAllParameters(disp, win, rAlineSize, fftS, nAlines, cropS, cropEnd, outputCom);
}

GPUFUNCSDLL int getDW(Complex *disp, float *win){
	return gpuGetDW(disp, win);
}

GPUFUNCSDLL int pingError(){
	return gpuGetError();
}


//This is not the function you're looking for...

//move along, move along

extern "C" int main(){
	int nAlines = 16383;


	int rAlineSize = 2048;
	int fftSize = 4096;
	int cropLength = 1024;

	U16 *raw;
	float *croppedR;
	float *croppedI;
	raw = (U16 *)malloc(rAlineSize*nAlines*sizeof(U16));
	croppedR = (float *)malloc(cropLength *nAlines*sizeof(float));
	croppedI = (float *)malloc(cropLength * nAlines * sizeof(float));
	croppedR[nAlines * cropLength - 1] = 100;
	croppedR[0] = -10;
	cout << "Size = " << nAlines * cropLength << endl;
	cout << croppedR[nAlines * cropLength - 1] << ' ' << croppedI[0] <<  endl;
	for (int i = 0; i<rAlineSize*nAlines; ++i){
		raw[i] = 5;//sin(2 * 3.14159*float(i) / 360);
	}
	cout << raw[10] << endl;
	//printf("%d\n", raw[nAlines*rAlineSize - 1]);
	float *dispReal = (float *)malloc(rAlineSize * sizeof(float));
	//disp = (Complex *)malloc(validAlineSize * sizeof(Complex));
	float *win;
	win = (float *)malloc(rAlineSize * sizeof(float));


	for (int i = 0; i < rAlineSize; ++i){

		dispReal[i] = 3.1415;
		win[i] = 10;
		
	}


	int status = setupGPU(dispReal, win, rAlineSize, fftSize, nAlines, 0, cropLength, true);
	cout << "Status: " << status << endl;

	//printf("Status : %d\n", status);

	//status = processAsWhole(raw, cropped);
	//status = processAsChunks(raw, cropped, rAlineSize, cropLength, nAlines, chunkSize);
	//high_resolution_clock::time_point t4 = high_resolution_clock::now();

	
	//printf("Status: %d\n", status);

	//printf("%f %f %f\n", cropped[0], cropped[1], cropped[(nAlines - 1)*cropLength]);
	status = pingError();
	cout << status << endl;
	high_resolution_clock::time_point t3 = high_resolution_clock::now();
	int error = gpuProcessComplex(raw, croppedR, croppedI);
	high_resolution_clock::time_point t4 = high_resolution_clock::now();
	auto duration2 = duration_cast<microseconds>(t4 - t3).count();

	cout << "Process : " << duration2 << " microseconds" << endl << "Error: " << error << endl;
	//	auto duration2 = duration_cast<microseconds>(t4 - t3).count();
	//cout << "Process : " << duration2 << " microseconds" << endl;
	//cout << cropped[0] << " " << cropped[(numAlines - 1) * cropRange] << " " << cropped[(numAlines*cropRange - 1)] << endl;
	
	endProgram();
	free(dispReal);

	free(win);

	free(raw);
	free(croppedR);
	free(croppedI);
	return 0;
}