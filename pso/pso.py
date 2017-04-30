# particleswarm.py
# python 3.4.3
# demo of particle swarm optimization (PSO)
# solves Rastrigin's function

import argparse
import random
import math    # cos() for Rastrigin
import copy    # array-copying convenience
import sys     # max float

W = 0.729    # inertia
C1 = 1.49445 # cognitive (particle)
C2 = 1.49445 # social (swarm)
Q = 1

def log(message) :
  if not Q :
    print(message)
    

# ------------------------------------

def show_vector(vector):
  for i in range(len(vector)):
    if i % 8 == 0: # 8 columns
      log("\n")#, end="")
    if vector[i] >= 0.0:
      log(' ')#, end="")
    log("%.4f" % vector[i])#, end="") # 4 decimals
    log(" ")#, end="")
  log("\n")

def error(position):
  err = 0.0
  for i in range(len(position)):
    xi = position[i]
    err += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
  return err

# ------------------------------------

class Particle:
  def __init__(self, dim, minx, maxx, seed):
    self.rnd = random.Random(seed)
    self.position = [0.0 for i in range(dim)]
    self.velocity = [0.0 for i in range(dim)]
    self.best_part_pos = [0.0 for i in range(dim)]

    for i in range(dim):
      self.position[i] = ((maxx - minx) *
        self.rnd.random() + minx)
      self.velocity[i] = ((maxx - minx) *
        self.rnd.random() + minx)

    self.error = error(self.position) # curr error
    self.best_part_pos = copy.copy(self.position) 
    self.best_part_err = self.error # best error

def Solve(max_epochs, n, dim, minx, maxx):
  rnd = random.Random(0)

  # create n random particles
  swarm = [Particle(dim, minx, maxx, i) for i in range(n)] 

  best_swarm_pos = [0.0 for i in range(dim)] # not necess.
  best_swarm_err = sys.float_info.max # swarm best
  for i in range(n): # check each particle
    if swarm[i].error < best_swarm_err:
      best_swarm_err = swarm[i].error
      best_swarm_pos = copy.copy(swarm[i].position) 

  epoch = 0
  w = W  # inertia
  c1 = C1 # cognitive (particle)
  c2 = C2 # social (swarm)
  done = False

  while epoch < max_epochs and not done:
    
    if epoch % 10 == 0 and epoch > 1:
      log("Epoch = " + str(epoch) +
        " best error = %.3f" % best_swarm_err)

    for i in range(n): # process each particle
      
      # compute new velocity of curr particle
      for k in range(dim): 
        r1 = rnd.random()    # randomizations
        r2 = rnd.random()
    
        swarm[i].velocity[k] = ( (w * swarm[i].velocity[k]) +
          (c1 * r1 * (swarm[i].best_part_pos[k] -
          swarm[i].position[k])) +  
          (c2 * r2 * (best_swarm_pos[k] -
          swarm[i].position[k])) )  

        if swarm[i].velocity[k] < minx:
          swarm[i].velocity[k] = minx
        elif swarm[i].velocity[k] > maxx:
          swarm[i].velocity[k] = maxx

      # compute new position using new velocity
      for k in range(dim): 
        swarm[i].position[k] += swarm[i].velocity[k]
  
      # compute error of new position
      swarm[i].error = error(swarm[i].position)

      # is new position a new best for the particle?
      if swarm[i].error < swarm[i].best_part_err:
        if (math.fabs(swarm[i].best_part_err - swarm[i].error) / swarm[i].best_part_err) < 0.00001:
            done = True
        swarm[i].best_part_err = swarm[i].error
        swarm[i].best_part_pos = copy.copy(swarm[i].position)

      # is new position a new best overall?
      if swarm[i].error < best_swarm_err:
        best_swarm_err = swarm[i].error
        best_swarm_pos = copy.copy(swarm[i].position)
    
    # for-each particle
    epoch += 1
  # while
  return best_swarm_pos, epoch
# end Solve

parser = argparse.ArgumentParser()

#-db DATABSE -u USERNAME -p PASSWORD -size 20
parser.add_argument("-w", "--initeria", help="Swarm inertia", dest="w", type=float)
parser.add_argument("-c1", "--cognitive", help="Cognitive weight (particle)", dest="c1", type=float)
parser.add_argument("-c2", "--social", help="Social weight (swarm)", dest="c2", type=float)
parser.add_argument("-q", "--quiet", help="only print final loss", dest="q", type=bool);
args = parser.parse_args()

W = args.w
C1 = args.c1
C2 = args.c2
Q = args.q

log("Configuration: W {} C1 {} C2 {} Q {}".format(
        W,
        C1,
        C2,
        Q   
        ))

log("\nBegin particle swarm optimization using Python demo\n")
dim = 3
log("Goal is to solve Rastrigin's function in " +
 str(dim) + " variables")
log("Function has known min = 0.0 at (")#, end="")
for i in range(dim-1):
  log("0, ")#, end="")
log("0)")

# TODO: make # of particles part of the genome
num_particles = 50
max_epochs = 100

log("Setting num_particles = " + str(num_particles))
log("Setting max_epochs    = " + str(max_epochs))
log("\nStarting PSO algorithm\n")

best_position, epochs = Solve(max_epochs, num_particles, dim, -10.0, 10.0)

log("\nPSO completed\n")
log("\nBest solution found:")
show_vector(best_position)
err = error(best_position)
err *= epochs # scale err by the number of steps it took to achieve it.
log("Error of best solution = %.6f" % err)
log("In %d" + str(epochs))
print("%.6f" % err)
log("\nEnd particle swarm demo\n")

