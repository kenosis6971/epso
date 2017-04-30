/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Random.cpp
 * Author: min
 * 
 * Created on July 19, 2016, 2:11 PM
 */

#include <limits>
#include <stdlib.h>
#include "Random.h"

Random::Random(int seed) {
    min = 0;
    max = std::numeric_limits<int>::max();
    range = max - min;
    srand(seed);
}


Random::~Random() {
}

double Random::NextDouble() {
    return (1.0 * rand() - min )/ range;
}


int Random::Next(int low, int hi) {
    return (rand() % ( hi - low) + low);
}

