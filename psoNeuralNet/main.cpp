/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   main.cpp
 * Author: min
 *
 * Created on July 19, 2016, 11:58 AM
 */

#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <limits>
#include <ios>
#include <iomanip>
#include <cmath>
#include "NeuralNetwork.h"
using namespace std;

void ShowVector(double* vector, int len, int valsPerRow, int decimals, bool newLine)
    {
    cout<<" ";
      for (int i = 0; i < len; ++i)
      {
        if (i % valsPerRow == 0) 
            cout<<endl;
        cout<< vector[i] << " ";
        
      }
      if (newLine == true) cout<<"";
    }

    void ShowMatrix(double** matrix, int numRows, int numCols, int decimals, bool newLine)
    {
        cout<<" ";
      for (int i = 0; i < numRows; ++i)
      {
        cout<< i << ": ";
        for (int j = 0; j < numCols; ++j)
        {
          cout<< matrix[i][j] << " ";
        }
        cout<<"\n ";
      }
      if (newLine == true) cout<<"";
    }
/*
 * 
 */
int main(int argc, char** argv) {
     cout<<"\nBegin neural network training with particle swarm optimization demo\n";
     cout<<"Data is a 30-item subset of the famous Iris flower set";
     cout<<"Data is sepal length, width, petal length, width -> iris species";
     cout<<"Iris setosa = 0 0 1, Iris versicolor = 0 1 0, Iris virginica = 1 0 0 ";
     cout<<"Predicting species from sepal length & width, petal length & width\n";

      // this is a 30-item subset of the 150-item set
      // data has been randomly assign to train (80%) and test (20%) sets
      // y-value (species) encoding: (0,0,1) = setosa; (0,1,0) = versicolor; (1,0,0) = virginica
      // for simplicity, data has not been normalized as you would in a real scenario

      double**trainData = new double*[24];
      trainData[0] = new double[7] { 6.3, 2.9, 5.6, 1.8, 1, 0, 0 };
      trainData[1] = new double[7] { 6.9, 3.1, 4.9, 1.5, 0, 1, 0 };
      trainData[2] = new double[7] { 4.6, 3.4, 1.4, 0.3, 0, 0, 1 };
      trainData[3] = new double[7] { 7.2, 3.6, 6.1, 2.5, 1, 0, 0 };
      trainData[4] = new double[7] { 4.7, 3.2, 1.3, 0.2, 0, 0, 1 };
      trainData[5] = new double[7] { 4.9, 3, 1.4, 0.2, 0, 0, 1 };
      trainData[6] = new double[7] { 7.6, 3, 6.6, 2.1, 1, 0, 0 };
      trainData[7] = new double[7] { 4.9, 2.4, 3.3, 1, 0, 1, 0 };
      trainData[8] = new double[7] { 5.4, 3.9, 1.7, 0.4, 0, 0, 1 };
      trainData[9] = new double[7] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };
      trainData[10] = new double[7] { 5, 3.6, 1.4, 0.2, 0, 0, 1 };
      trainData[11] = new double[7] { 6.4, 3.2, 4.5, 1.5, 0, 1, 0 };
      trainData[12] = new double[7] { 4.4, 2.9, 1.4, 0.2, 0, 0, 1 };
      trainData[13] = new double[7] { 5.8, 2.7, 5.1, 1.9, 1, 0, 0 };
      trainData[14] = new double[7] { 6.3, 3.3, 6, 2.5, 1, 0, 0 };
      trainData[15] = new double[7] { 5.2, 2.7, 3.9, 1.4, 0, 1, 0 };
      trainData[16] = new double[7] { 7, 3.2, 4.7, 1.4, 0, 1, 0 };
      trainData[17] = new double[7] { 6.5, 2.8, 4.6, 1.5, 0, 1, 0 };
      trainData[18] = new double[7] { 4.9, 2.5, 4.5, 1.7, 1, 0, 0 };
      trainData[19] = new double[7] { 5.7, 2.8, 4.5, 1.3, 0, 1, 0 };
      trainData[20] = new double[7] { 5, 3.4, 1.5, 0.2, 0, 0, 1 };
      trainData[21] = new double[7] { 6.5, 3, 5.8, 2.2, 1, 0, 0 };
      trainData[22] = new double[7] { 5.5, 2.3, 4, 1.3, 0, 1, 0 };
      trainData[23] = new double[7] { 6.7, 2.5, 5.8, 1.8, 1, 0, 0 };

      double** testData = new double*[6];
      testData[0] = new double[7] { 4.6, 3.1, 1.5, 0.2, 0, 0, 1 };
      testData[1] = new double[7] { 7.1, 3, 5.9, 2.1, 1, 0, 0 };
      testData[2] = new double[7] { 5.1, 3.5, 1.4, 0.2, 0, 0, 1 };
      testData[3] = new double[7] { 6.3, 3.3, 4.7, 1.6, 0, 1, 0 };
      testData[4] = new double[7] { 6.6, 2.9, 4.6, 1.3, 0, 1, 0 };
      testData[5] = new double[7] { 7.3, 2.9, 6.3, 1.8, 1, 0, 0 };

      cout<<"The training data is:" <<endl;
      ShowMatrix(trainData, 24, 7, 1, true);

      cout<<"The test data is:" <<endl;
      ShowMatrix(testData, 6, 7, 1, true);

      cout<<"\nCreating a 4-input, 6-hidden, 3-output neural network" <<endl;
      cout<<"Using tanh and softmax activations" <<endl;
      const int numInput = 4;
      const int numHidden = 6;
      const int numOutput = 3;
      NeuralNetwork *nn = new NeuralNetwork(numInput, numHidden, numOutput);

      int numParticles = 12;
      int maxEpochs = 700;
      double exitError = 0.060;
      double probDeath = 0.005;
      const int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;

      cout<<"Setting numParticles = " << numParticles << endl;
      cout<<"Setting maxEpochs = " << maxEpochs <<endl;
      cout<<"Setting early exit MSE error = " << exitError <<endl;
      cout<<"Setting probDeath = " << probDeath << endl;
              // other optional PSO parameters (weight decay, death, etc) here

      cout<<"\nBeginning training using a particle swarm\n";
      double* bestWeights = nn->Train(trainData, 24, numParticles, maxEpochs, exitError, probDeath);
      cout<<"Training complete" <<endl;
      cout<<"Final neural network weights and bias values:" <<endl;
      ShowVector(bestWeights, numWeights, 10, 3, true);

      nn->SetWeights(bestWeights, numWeights);
      double trainAcc = nn->Accuracy(trainData, 24);
      cout<<"\nAccuracy on training data = " << trainAcc;

      double testAcc = nn->Accuracy(testData, 6);
      cout<<"\nAccuracy on test data = " << testAcc;

      cout<<"\nEnd neural network training with particle swarm optimization demo\n";
      
              return 0;
    } // Main

  
