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
import scipy.spatial.distance as dist
import pyparticles.forces.force as fr

import pyparticles.pset.opencl_context as occ 

try:
    import pyopencl as cl
except:
    ___foo = 0


class MdsForce( fr.Force ) :
    r"""
    Compute the gravitational force between the particles

    The gravity between two particles is defined as follow:

    .. math::

        \mathbf{F}_{12}=-G \frac{m_1 m_2 }{r^2}\hat{r}_{12}

    Constructor
    
    :param    size:        the number of particles in the system
    :param    dim:         the dimension of the system
    :param    m:           a vector containig the masses
    :param    Const:       the gravitational constant
    """
    def __init__(self , size , dim=2 , m=None , Consts=1.0 ):
        
        self.__dim = dim
        self.__size = size
        self.__G = Consts
        self.__A = np.zeros( ( size , dim ) )
        self.__Fm = np.zeros( ( size , size ) )
        self.__V = np.zeros( ( size , size ) )
        self.__D = np.zeros( ( size , size ) )
        self.__M = np.zeros( ( size , size ) )
        if m != None :
            self.set_masses( m )

    def set_distances(self, d):
        """
        Set the distances used for computing the forces.
        """
        self.__D[:, :] = d

    def set_masses( self , m ):
        """
        Set the masses used for computing the forces.
        """
        self.__M[:,:] = m
        
        
    def update_force( self , p_set ):
        """
        Compute the force of the current status of the system and return the accelerations of every particle in a *size by dim* array
        """

        nr_of_points = len(p_set.X)
        # for i in range(nr_of_points):
        #     for j in range(i):
        #

        return self.__A
    
    
    def getA(self):
        """
        Return the currents accelerations of the particles
        """
        return self.__A
    
    A = property( getA , doc="Return the currents accelerations of the particles (getter only)")


    def getF(self):
        """
        Return the currents forces on the particles
        """
        return ( self.__A.T * self.__M[:,0] ).T
    
    F = property( getF , doc="Return the currents forces on the particles (getter only)")
