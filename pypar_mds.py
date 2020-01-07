# PyParticles : Particles simulation in python
# Copyright (C) 2012  Simone Riva
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

import pyparticles.pset.particles_set as ps

import pypar_mds_force as mf
import pyparticles.forces.damping as da
import pyparticles.forces.multiple_force as ml

import pyparticles.ode.leapfrog_solver as lps


import pyparticles.animation.animated_ogl as aogl

import scipy.spatial.distance as dist

from sklearn.datasets import load_digits

def get_distances(points, with_labels=True):
    np.random.seed(1234)
    result = load_digits(n_class=10, return_X_y=True)
    # result = result if with_labels else result[0]
    return result[0][:points], result[1][:points]


numberColorDictionary = {
    0 : [1.0, 0.0, 0.0, 1.0], #red
    1 : [0.0, 1.0, 0.0, 1.0], #green
    2 : [0.0, 0.0, 1.0, 1.0], #blue
    3 : [ 31.0 / 256.0, 170.0 / 256.0, 178.0 / 256.0, 1.0], #light blue
    4 : [ 237.0 / 256.0, 221.0 / 256.0, 38.0 / 256.0, 1.0], #yellow
    5 : [ 248.0 / 256.0, 27.0 / 256.0, 171.0 / 256.0, 1.0], #pink
    6 : [ 248.0 / 256.0, 134.0 / 256.0, 27.0 / 256.0, 1.0], #orange
    7 : [ 83.0 / 256.0, 27.0 / 256.0, 248.0 / 256.0, 1.0], #purple
    8 : [1.0, 1.0, 1.0, 1.0], # white
    9 : [ 253.0 / 256.0, 196.0 / 256.0, 181.0 / 256.0, 1.0] #creme
}

def mds():
    np.random.seed(1234)
    """
    MDS
    """


    #set main parameters
    dim = 2

    dt = 1
    steps = 10000

    number_of_points = 500

    FLOOR = -10
    CEILING = 10


    #get working data set and numbers
    vectors, numbers = get_distances(number_of_points)

    pset = ps.ParticlesSet(number_of_points, dim , mass=True, color=True)

    for i in range(number_of_points):
        color = numberColorDictionary[numbers[i]]
        if i % 2 == 0:
            pset.color[i][0] = color[0]
            pset.color[i][1] = color[1]
            pset.color[i][2] = color[2]
            pset.color[i][3] = color[3]

    #point in initial random positions
    pset.X[:] = np.random.rand(number_of_points, dim)

    #all points with same mass
    pset.M[:] = np.ones((number_of_points, 1))


	#calculate euclidean distances of points
    #matrix (n_o_p) x (n_o_p)
    distances = np.zeros((number_of_points, number_of_points))
    for i in range(number_of_points):
        for j in range(number_of_points):
            distances[i][j] = dist.euclidean([vectors[i]], vectors[j])


    pset.unit = 1
    pset.mass_unit = 1

    #build main forces in our universe
    force = mf.MdsForce(pset.size, distances)
    force.set_masses(pset.M)
    force.update_force(pset)


    damp = da.Damping(pset.size, dim=2, Consts=1.0)
    damp.set_masses(pset.M)
    damp.update_force(pset)


    mult = ml.MultipleForce(pset.size, dim=2)
    mult.append_force(force)
    mult.append_force(damp)


    solver = lps.LeapfrogSolver(mult, pset , dt )

    a = aogl.AnimatedGl()
   # a = anim.AnimatedScatter()

    a.trajectory = True
    a.trajectory_step = 1


    # a.xlim = ( FLOOR , CEILING )
    # a.ylim = ( FLOOR , CEILING )
    # a.zlim = ( FLOOR , CEILING )

    a.ode_solver = solver
    a.pset = pset
    a.steps = steps

    a.build_animation()

    a.start()



mds()