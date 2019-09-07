"""
n body problem in 3-space:
- Gravity is the only force
- Bodies cannot collide

Masses, positions, and momenta are bounded by positions_max, momentum_max,
mass_min, and mass_max.

Problems:
- If two bodies pass too closely together, the timestep becomes too large to
    accurately model the interaction.
    Potential solutions:
    - Make the timestep specific to each body. The larger the force on
        an object, the smaller the timestep. Always do the next timestep.
        - Problem still exists because timesteps can get arbitrarily small.
    - Make the objects combine if they get too close together.
        - Could pose a problem with conservation of energy

- The approximation errors compound and quickly get out of control. You can only
    model on the order of minutes or possibly hours. Days are out of the question
    unless the bodies are really far apart. Any longer, and the model will
    quickly diverge from reality.
"""

import random
import numpy as np
import scipy.constants as spc

n_bodies = 5

position_max = 1000
momentum_max = 1
mass_min = 0
mass_max = 1e11

class Body:
    def __init__(self):
        self.mass = (mass_max - mass_min)*random.random() + mass_min
        self.position = (
            position_max*(2*random.random() - 1),
            position_max*(2*random.random() - 1),
            position_max*(2*random.random() - 1)
        )
        self.momentum = (
            momentum_max*(2*random.random() - 1),
            momentum_max*(2*random.random() - 1),
            momentum_max*(2*random.random() - 1)
        )

class Ensemble:
    def __init__(self, n_bodies):
        self.total_mass = 0
        self.num_bodies = n_bodies
        self.time_passed = 0
        self.bodies = self._set_bodies(n_bodies)

    def _set_bodies(self, n_bodies):
        # t, s = O(n), O(n)
        bodies = list()
        for i in range(n_bodies):
            bodies.append(Body())
            self.total_mass += bodies[i].mass
        return bodies

    def get_total_mass(self): return self.total_mass

    def print_positions(self):
        # t = O(n)
        positions_str = str(); i = 0
        for body in self.bodies:
            x, y, z = self.bodies[i].position
            positions_str += "Body {}: ({:.2f}, {:.2f}, {:.2f}), ".format(i + 1, x, y, z)
            i += 1
        print(positions_str[:-2])

    def get_total_momentum(self):
        # t, s = O(n), O(1)
        p_total = [0, 0, 0]
        for body in self.bodies:
            px, py, pz = body.momentum
            p_total[0] += px
            p_total[1] += py
            p_total[2] += pz
        return tuple(p_total)

    def get_kinetic(self, body):
        # t = O(1)
        p = np.linalg.norm(body.momentum)
        return p*p/2/body.mass

    def get_total_kinetic(self):
        # t = O(n)
        kinetic = 0
        for body in self.bodies:
            kinetic += self.get_kinetic(body)
        return kinetic

    def get_potential(self, body):
        # t = O(n)
        potential = 0
        for body2 in self.bodies:
            if body == body2: continue

            x1, y1, z1 = body.position
            x2, y2, z2 = body2.position
            d = np.linalg.norm((x2 - x1, y2 - y1, z2 - z1))

            potential += -body.mass*body2.mass*spc.G/d
        return potential

    def get_total_potential(self):
        # t = O(n^2)
        potential = 0
        for body in self.bodies:
            potential += self.get_potential(body)
        return potential

    def get_energy(self, body):
        # t = O(n)
        return self.get_kinetic(body) + self.get_potential(body)

    def get_total_energy(self):
        # t = O(n^2)
        energy = 0
        for body in self.bodies:
            energy += self.get_energy(body)
        return energy

    def force_on_body(self, body):
        # t = O(n)
        force = [0, 0, 0]
        for body2 in self.bodies:
            if body == body2: continue

            x1, y1, z1 = body.position
            x2, y2, z2 = body2.position
            d_pointer = [x2 - x1, y2 - y1, z2 - z1]
            d_norm = np.linalg.norm(d_pointer)
            d_unit = [e/d_norm for e in d_pointer]
            magnitude = spc.G*body.mass*body2.mass/d_norm**2

            for i in range(3): force[i] += magnitude*d_unit[i]
        return tuple(force)

    def advance(self, timestep = 1):
        # t = O(n^2)
        new_positions = list()
        new_momenta = list()
        for body in self.bodies:
            m = body.mass
            x, y, z = body.position
            px, py, pz = body.momentum
            fx, fy, fz = self.force_on_body(body)

            new_positions.append((
                x + (px/m)*timestep,
                y + (py/m)*timestep,
                z + (pz/m)*timestep
            ))
            new_momenta.append((
                px + fx*timestep,
                py + fy*timestep,
                pz + fz*timestep
            ))

        for body in self.bodies:
            body.position = new_positions.pop(0)
            body.momentum = new_momenta.pop(0)

        self.time_passed += timestep

ensemble = Ensemble(n_bodies)
initial_energy = ensemble.get_total_energy()

while ensemble.time_passed < 30:
    print(80*"-")
    print("Time passed: {:.2f} min".format(ensemble.time_passed/60))
    ensemble.advance(0.1)
    new_energy = ensemble.get_total_energy()
    print("Change in energy since start: ", 100*(new_energy/initial_energy - 1), "%")
