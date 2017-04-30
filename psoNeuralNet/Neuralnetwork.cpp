/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "NeuralNetwork.h"
#include "Particle.h"
#include "Random.h"
#include <stdexcept>
#include <math.h>
#include <limits>
#include <cstring>
#include <cmath>
#include <time.h>
#include <iostream>
using namespace std;

NeuralNetwork::NeuralNetwork(int nInput, int nHidden, int nOutput) {
    //rnd = new Random(16); // for particle initialization. 16 just gives nice demo
    numInput = nInput;
    numHidden = nHidden;
    numOutput = nOutput;
    inputs = new double[numInput];
    ihWeights = MakeMatrix(numInput, numHidden);
    hBiases = new double[numHidden];
    hOutputs = new double[numHidden];
    hoWeights = MakeMatrix(numHidden, numOutput);
    oBiases = new double[numOutput];
    outputs = new double[numOutput];
}; // ctor

NeuralNetwork::~NeuralNetwork() {
    delete [] inputs;
    for (int i = 0; i < numInput; i++) {
        delete [] ihWeights[i];
    }
    delete [] ihWeights;
    delete [] hBiases;
    delete [] hOutputs;
    for (int i = 0; i < numHidden; i++) {
        delete [] hoWeights[i];
    }
    delete [] hoWeights;
    delete [] oBiases;
    delete [] outputs;
}

double** NeuralNetwork::MakeMatrix(int rows, int cols) // helper for ctor
{
    double** result = new double *[rows];
    for (int r = 0; r < rows; ++r)
        result[r] = new double[cols];
    return result;
};

//public override string ToString() // yikes
//{
//  string s = "";
//  s += "===============================\n";
//  s += "numInput = " + numInput + " numHidden = " + numHidden + " numOutput = " + numOutput + "\n\n";

//  s += "inputs: \n";
//  for (int i = 0; i < inputs.Length; ++i)
//    s += inputs[i].ToString("F2") + " ";
//  s += "\n\n";

//  s += "ihWeights: \n";
//  for (int i = 0; i < ihWeights.Length; ++i)
//  {
//    for (int j = 0; j < ihWeights[i].Length; ++j)
//    {
//      s += ihWeights[i][j].ToString("F4") + " ";
//    }
//    s += "\n";
//  }
//  s += "\n";

//  s += "hBiases: \n";
//  for (int i = 0; i < hBiases.Length; ++i)
//    s += hBiases[i].ToString("F4") + " ";
//  s += "\n\n";

//  s += "hOutputs: \n";
//  for (int i = 0; i < hOutputs.Length; ++i)
//    s += hOutputs[i].ToString("F4") + " ";
//  s += "\n\n";

//  s += "hoWeights: \n";
//  for (int i = 0; i < hoWeights.Length; ++i)
//  {
//    for (int j = 0; j < hoWeights[i].Length; ++j)
//    {
//      s += hoWeights[i][j].ToString("F4") + " ";
//    }
//    s += "\n";
//  }
//  s += "\n";

//  s += "oBiases: \n";
//  for (int i = 0; i < oBiases.Length; ++i)
//    s += oBiases[i].ToString("F4") + " ";
//  s += "\n\n";

//  s += "outputs: \n";
//  for (int i = 0; i < outputs.Length; ++i)
//    s += outputs[i].ToString("F2") + " ";
//  s += "\n\n";

//  s += "===============================\n";
//  return s;
//}

// ----------------------------------------------------------------------------------------

void NeuralNetwork::SetWeights(double* weights, int len) {
    // copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases
    int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
    if (len != numWeights)
        throw std::invalid_argument("Bad weights array length: ");

    int k = 0; // points into weights param

    for (int i = 0; i < numInput; ++i)
        for (int j = 0; j < numHidden; ++j)
            ihWeights[i][j] = weights[k++];
    for (int i = 0; i < numHidden; ++i)
        hBiases[i] = weights[k++];
    for (int i = 0; i < numHidden; ++i)
        for (int j = 0; j < numOutput; ++j)
            hoWeights[i][j] = weights[k++];
    for (int i = 0; i < numOutput; ++i)
        oBiases[i] = weights[k++];
};

double* NeuralNetwork::GetWeights() {
    // returns the current set of wweights, presumably after training
    int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
    double* result = new double[numWeights];
    int k = 0;
    for (int i = 0; i < numInput; ++i)
        for (int j = 0; j < numHidden; ++j)
            result[k++] = ihWeights[i][j];
    for (int i = 0; i < numHidden; ++i)
        result[k++] = hBiases[i];
    for (int i = 0; i < numHidden; ++i)
        for (int j = 0; j < numOutput; ++j)
            result[k++] = hoWeights[i][j];
    for (int i = 0; i < numOutput; ++i)
        result[k++] = oBiases[i];
    return result;
}

// ----------------------------------------------------------------------------------------

double* NeuralNetwork::ComputeOutputs(double* xValues, int len) {
    if (len != numInput)
        throw new std::invalid_argument("Bad xValues array length");

    double* hSums = new double[numHidden]; // hidden nodes sums scratch array
    double* oSums = new double[numOutput]; // output nodes sums

    for (int i = 0; i < len; ++i) // copy x-values to inputs
        inputs[i] = xValues[i];

    for (int j = 0; j < numHidden; ++j) // compute i-h sum of weights * inputs
        for (int i = 0; i < numInput; ++i)
            hSums[j] += inputs[i] * ihWeights[i][j]; // note +=

    for (int i = 0; i < numHidden; ++i) // add biases to input-to-hidden sums
        hSums[i] += hBiases[i];

    for (int i = 0; i < numHidden; ++i) // apply activation
        hOutputs[i] = HyperTanFunction(hSums[i]); // hard-coded

    for (int j = 0; j < numOutput; ++j) // compute h-o sum of weights * hOutputs
        for (int i = 0; i < numHidden; ++i)
            oSums[j] += hOutputs[i] * hoWeights[i][j];

    for (int i = 0; i < numOutput; ++i) // add biases to input-to-hidden sums
        oSums[i] += oBiases[i];

    double* softOut = Softmax(oSums, numOutput); // softmax activation does all outputs at once for efficiency
    memcpy(outputs, softOut, numOutput * sizeof(double));

    double* retResult = new double[numOutput]; // could define a GetOutputs method instead
    memcpy(retResult, outputs, numOutput * sizeof(double));
    return retResult;
} // ComputeOutputs

double NeuralNetwork::HyperTanFunction(double x) {
    if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
    else if (x > 20.0) return 1.0;
    else return std::tanh(x);
}

double* NeuralNetwork::Softmax(double *oSums, int len) {
    // does all output nodes at once so scale doesn't have to be re-computed each time
    // determine max output sum
    double max = oSums[0];
    for (int i = 0; i < len; ++i)
        if (oSums[i] > max) max = oSums[i];

    // determine scaling factor -- sum of exp(each val - max)
    double scale = 0.0;
    for (int i = 0; i < len; ++i)
        scale += std::exp(oSums[i] - max);

    double * result = new double[len];
    for (int i = 0; i < len; ++i)
        result[i] = std::exp(oSums[i] - max) / scale;

    return result; // now scaled so that xi sum to 1.0
}

// ----------------------------------------------------------------------------------------

double* NeuralNetwork::Train(double** trainData, int sampleCount, 
        int numParticles, int maxEpochs, double exitError, double probDeath) {
    // PSO version training. best weights stored into NN and returned
    // particle position == NN weights
   
    Random *rnd = new Random(time(NULL)); // 16 just gives nice demo

    int numWeights = (numInput * numHidden) + (numHidden * numOutput) +
            numHidden + numOutput;

    // use PSO to seek best weights
    int epoch = 0;
    double minX = -10.0; // for each weight. assumes data has been normalized about 0
    double maxX = 10.0;
    double w = 0.729; // inertia weight
    double c1 = 1.49445; // cognitive/local weight
    double c2 = 1.49445; // social/global weight
    double r1, r2; // cognitive and social randomizations

    Particle** swarm = new Particle*[numParticles];
    // best solution found by any particle in the swarm. implicit initialization to all 0.0
    double* bestGlobalPosition = new double[numWeights];
    double bestGlobalError = std::numeric_limits<double>::max();//double.MaxValue; // smaller values better

    //double minV = -0.01 * maxX;  // velocities
    //double maxV = 0.01 * maxX;

    // swarm initialization
    // initialize each Particle in the swarm with random positions and velocities
    for (int i = 0; i < numParticles; ++i) {
        double* randomPosition = new double[numWeights];
        
        for (int j = 0; j < numWeights; ++j) {
            //double lo = minX;
            //double hi = maxX;
            //randomPosition[j] = (hi - lo) * rnd.NextDouble() + lo;
            randomPosition[j] = (maxX - minX) * rnd->NextDouble() + minX;
            
        }

        // randomPosition is a set of weights; sent to NN
        //double error = MeanCrossEntropy(trainData, randomPosition);
        double error = MeanSquaredError(trainData, sampleCount, randomPosition, numWeights);
        double* randomVelocity = new double[numWeights];

        for (int j = 0; j < numWeights; ++j) {
            //double lo = -1.0 * Math.Abs(maxX - minX);
            //double hi = Math.Abs(maxX - minX);
            //randomVelocity[j] = (hi - lo) * rnd.NextDouble() + lo;
            double lo = 0.1 * minX;
            double hi = 0.1 * maxX;
            randomVelocity[j] = (hi - lo) * rnd->NextDouble() + lo;

        }
        
        
        swarm[i] = new Particle(randomPosition, error, randomVelocity, randomPosition, error, numWeights); // last two are best-position and best-error

        // does current Particle have global best position/solution?
        if (swarm[i]->error < bestGlobalError) {
            bestGlobalError = swarm[i]->error;
            memcpy(bestGlobalPosition, swarm[i]->position, numWeights * sizeof(double));
        }
    }
    // initialization



    cout<<"Entering main PSO weight estimation processing loop" << endl;

    // main PSO algorithm

    int* sequence = new int[numParticles]; // process particles in random order
    for (int i = 0; i < numParticles; ++i)
        sequence[i] = i;

    while (epoch < maxEpochs) {
        if (bestGlobalError < exitError) break; // early exit (MSE error)

        double* newVelocity = new double[numWeights]; // step 1
        double* newPosition = new double[numWeights]; // step 2
        double newError; // step 3

        Shuffle(sequence, numParticles, rnd); // move particles in random sequence

        for (int pi = 0; pi < numParticles; ++pi) // each Particle (index)
        {
            int i = sequence[pi];
            Particle *currP = swarm[i]; // for coding convenience

            // 1. compute new velocity
            for (int j = 0; j < numWeights; ++j) // each x value of the velocity
            {
                r1 = rnd->NextDouble();
                r2 = rnd->NextDouble();

                // velocity depends on old velocity, best position of parrticle, and 
                // best position of any particle
                newVelocity[j] = (w * currP->velocity[j]) +
                        (c1 * r1 * (currP->bestPosition[j] - currP->position[j])) +
                        (c2 * r2 * (bestGlobalPosition[j] - currP->position[j]));
            }

            memcpy(currP->velocity, newVelocity, numWeights * sizeof(double) );

            // 2. use new velocity to compute new position
            for (int j = 0; j < numWeights; ++j) {
                newPosition[j] = currP->position[j] + newVelocity[j]; // compute new position
                if (newPosition[j] < minX) // keep in range
                    newPosition[j] = minX;
                else if (newPosition[j] > maxX)
                    newPosition[j] = maxX;
            }

            memcpy(currP->position, newPosition, numWeights * sizeof(double));

            // 2b. optional: apply weight decay (large weights tend to overfit) 

            // 3. use new position to compute new error
            //newError = MeanCrossEntropy(trainData, newPosition); // makes next check a bit cleaner
            newError = MeanSquaredError(trainData, sampleCount, newPosition, numWeights);
            currP->error = newError;

            if (newError < currP->bestError) // new particle best?
            {
                memcpy(currP->bestPosition, newPosition, numWeights * sizeof(double));
                currP->bestError = newError;
            }

            if (newError < bestGlobalError) // new global best?
            {
                memcpy(bestGlobalPosition, newPosition, numWeights * sizeof(double));
                bestGlobalError = newError;
            }

            // 4. optional: does curr particle die?
            double die = rnd->NextDouble();
            if (die < probDeath) {
                // new position, leave velocity, update error
                for (int j = 0; j < numWeights; ++j)
                    currP->position[j] = (maxX - minX) * rnd->NextDouble() + minX;
                currP->error = MeanSquaredError(trainData, sampleCount, currP->position, numWeights);
                memcpy(currP->bestPosition, currP->position, numWeights * sizeof(double));
                currP->bestError = currP->error;

                if (currP->error < bestGlobalError) // global best by chance?
                {
                    bestGlobalError = currP->error;
                    memcpy(bestGlobalPosition, currP->position, numWeights * sizeof(double));
                }
            }

        } // each Particle

        cout << epoch << ": error "<< bestGlobalError <<endl;
        ++epoch;

    } // while

    SetWeights(bestGlobalPosition, numWeights); // best position is a set of weights
    double* retResult = new double[numWeights];
    memcpy(retResult, bestGlobalPosition,  numWeights * sizeof(double));
    return retResult;

} // Train

void NeuralNetwork::Shuffle(int* sequence, int len, Random *rnd) {
    for (int i = 0; i < len; ++i) {
        int r = rnd->Next(i, len);
        int tmp = sequence[r];
        sequence[r] = sequence[i];
        sequence[i] = tmp;
    }
}

double NeuralNetwork::MeanSquaredError(double** trainData, int sampleCount, double* weights, int numWeights) {
    // assumes that centroids and widths have been set!
    SetWeights(weights, numWeights); // copy the weights to evaluate in

    double* xValues = new double[numInput]; // inputs
    double* tValues = new double[numOutput]; // targets
    double sumSquaredError = 0.0;
    for (int i = 0; i < sampleCount; ++i) // walk through each training data item
    {
        // following assumes data has all x-values first, followed by y-values!
        memcpy(xValues,  trainData[i],  numInput * sizeof(double)); // extract inputs
        memcpy(tValues,  trainData[i] + numInput,  numOutput * sizeof(double)); // extract targets
        double* yValues = ComputeOutputs(xValues, numInput); // compute the outputs using centroids, widths, weights, bias values
        for (int j = 0; j < numOutput; ++j)
            sumSquaredError += ((yValues[j] - tValues[j]) * (yValues[j] - tValues[j]));
    }
    return sumSquaredError / sampleCount;
}

//private double MeanCrossEntropy(double[][] trainData, double[] weights) 
//{
//  // (average) Cross Entropy for a given particle's position/weights
//  // how good (cross entropy) are weights? CrossEntropy is error so smaller values are better
//  SetWeights(weights); // load the weights and biases to examine into the NN

//  double sce = 0.0; // sum of cross entropies of all data items
//  double[] xValues = new double[numInput]; // inputs
//  double[] tValues = new double[numOutput]; // targets

//  // walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
//  for (int i = 0; i < trainData.Length; ++i)  
//  {
//    Array.Copy(trainData[i], xValues, numInput); // extract inputs
//    Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // extract targets

//    double[] yValues = ComputeOutputs(xValues); // run the inputs through the neural network
//    // assumes outputs are 'softmaxed' -- all between 0 and 1, and sum to 1

//    // CE = -Sum( t * log(y) )
//    // see http://dame.dsf.unina.it/documents/softmax_entropy_VONEURAL-SPE-NA-0004-Rel1.0.pdf 
//    // for an explanation of why cross tropy sometimes given as CE = -Sum( t * log(t/y) ), as in 
//    // "On the Pairing of the Softmax Activation and Cross-Entropy Penalty Functions and
//    // the Derivation of the Softmax Activation Function", Dunne & Campbell.
//    double currSum = 0.0;
//    for (int j = 0; j < yValues.Length; ++j)
//    {
//      currSum += tValues[j] * Math.Log(yValues[j]); // diff between targets and y
//    }
//    sce += currSum; // accumulate
//  }

//  return -sce / trainData.Length;
//} // MeanCrossEntropy



// ----------------------------------------------------------------------------------------

double NeuralNetwork::Accuracy(double** testData, int sampleCount) {
    // percentage correct using winner-takes all
    int numCorrect = 0;
    int numWrong = 0;
    double* xValues = new double[numInput]; // inputs
    double* tValues = new double[numOutput]; // targets
    double* yValues; // computed Y

    for (int i = 0; i < sampleCount; ++i) {
        memcpy( xValues, testData[i], numInput * sizeof(double)); // parse test data into x-values and t-values
        memcpy(tValues, testData[i] + numInput, numOutput *sizeof(double));
        yValues = ComputeOutputs(xValues, numInput);
        int maxIndex = MaxIndex(yValues, numOutput); // which cell in yValues has largest value?

        if (tValues[maxIndex] == 1.0) // ugly. consider AreEqual(double x, double y)
            ++numCorrect;
        else
            ++numWrong;
    }
    return (numCorrect * 1.0) / (numCorrect + numWrong); // ugly 2 - check for divide by zero
}

int NeuralNetwork::MaxIndex(double* vector, int len) // helper for Accuracy()
{
    // index of largest value
    int bigIndex = 0;
    double biggestVal = vector[0];
    for (int i = 0; i < len; ++i) {
        if (vector[i] > biggestVal) {
            biggestVal = vector[i];
            bigIndex = i;
        }
    }
    return bigIndex;
}
