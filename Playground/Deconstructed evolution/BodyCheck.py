""" Morphology Analysis 
TODO: - Remove self.population and self.bodies -> Make seperately callable
"""

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from database_components import (
    Base,
    Experiment,
    Generation,
    Genotype,
    Individual,
    Population,
)
from revolve2.modular_robot.body import Module
from revolve2.modular_robot.body.base import ActiveHinge, Brick, Core, Body
from revolve2.experimentation.evolution.abstract_elements import Morpho


class BodyCheck(Morpho):
    """
    Infer information from the robot's body, such as its 'nose', overall shape, etc.
    """
    def __init__(self) -> None:
        # Plotting variables
        self.colors = ["b", "r", "y"]
        self.legend = ["Brick", "Hinge", "Core"]
        self.marker = ["s", "^", "o"]
        self.sizes = [20, 20, 50]
        
    def findBodies(self, population: Population) -> list[Body]:
        """
        Returns all robot bodies in a population.
        """
        return [
            p.genotype.develop().body for p in population.individuals
            ]
        
    def findModules(self, body: Body) -> list[Module]:
        """
        Find all modules for a single robot.
        """
        modules = body.find_modules_of_type(Brick) + \
            body.find_modules_of_type(ActiveHinge) + \
                body.find_modules_of_type(Core)
                
        coords = np.array([
            np.array(body.grid_position(m)) for m in modules
            ]).astype(int)
        
        return coords
                
    def findModulesSep(
            self, body: Body
            ) -> list[list[Brick], list[ActiveHinge], list[Core]]:
        """
        Find all different types of modules for a single robot.
        """
        modules = [
            body.find_modules_of_type(Brick),
            body.find_modules_of_type(ActiveHinge),
            body.find_modules_of_type(Core),
            ]
        
        coords = [
            np.array([
                np.array(body.grid_position(mod)) for mod in modules[0]
                ]).astype(int),
            np.array([
                np.array(body.grid_position(mod)) for mod in modules[1]
                ]).astype(int),
            np.array([
                np.array(body.grid_position(mod)) for mod in modules[2]
                ]).astype(int),
            ]
        
        return coords
    
    def gridBody(self, body, coords):
        """
        Generate a bounding box for a robot body.
        
        :param modules: List of modules in a robot.
        """
        grid = np.zeros([40,40])
        for c in coords[:,:2]:
            c += np.array([20,20])
            grid[c[0],c[1]] += 1
        
        return grid
    
    def plot2D(self, body, idx, plt_out=False, ax=None):
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.set_title(f"Body no. {idx}")
            
        modules = self.findModules(body)
        grid = self.gridBody(body, modules)
        ax.imshow(grid)
        ax.set_xlabel("Y")
        ax.set_ylabel("X")
        if plt_out: plt.show()
        
    def plot3D(self, body, idx, plt_out=False, ax=None):
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.set_title(f"Body no. {idx}")
            
        allCoords = self.findModulesSep(body)
        for c_idx, coords in enumerate(allCoords):
            if len(coords) == 0: continue # No modules of certain type present
            
            x = coords[:,0]
            y = coords[:,1]
            z = coords[:,2]
            ax.scatter(x,y,z, color=self.colors[c_idx], 
                       marker=self.marker[c_idx], label=self.legend[c_idx],
                       s=self.sizes[c_idx])
            
        #ax.set_title(f"Body no. {idx}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        if plt_out: plt.show()      

    def plotFigs(
            self, body: Body, nose: int, idx: int) -> None:
        fig = plt.figure()
        fig.tight_layout()
        subfigs = fig.subfigures(1,2)

        ax1 = subfigs[0].add_subplot()
        ax2 = subfigs[1].add_subplot(projection="3d")        
        self.plot2D(body, idx, plt_out=False, ax=ax1)
        self.plot3D(body, idx, plt_out=False, ax=ax2)
        fig.suptitle(f"Body no. {idx}\nNose orientation: {nose}")
        
    def plotPop(self, population: Population):
        bodies = self.findBodies(population)
        for idx, body in enumerate(bodies):
            nose = population.individuals[idx].nose
            assert nose >= 0, "Nose not initialized. Call findNose() on population first."
            self.plotFigs(body, nose, idx)
    
    def findNose(self, population: Population):
        """
        Find the `nose` (frontal orientation) of the robots. The nose is in the longest 
        x or y direction and the closest from the core (i.e. a salamander).
        This method ignores height in the z-direction.
        """
        bodies = self.findBodies(population)
        for idx, body in enumerate(bodies):
            grid = self.findModules(body)[:,:2]
            min_x = np.min(grid[:,0])
            max_x = np.max(grid[:,0])
            min_y = np.min(grid[:,1])
            max_y = np.max(grid[:,1])
            width = max_x - min_x
            depth = max_y - min_y
            
            population.individuals[idx].nose = self.noseLoc(
                min_x, max_x, min_y, max_y, width, depth)
            
        return population
    
    def noseLoc(self, min_x, max_x, min_y, max_y, w, d) -> int:
        """
        Return an integer denoting the nose's direction:
                ^ (0)
          (3) < + > (1)
                ⌄ (2)
        """
        if w != d:
            if w > d:
                if abs(max_x) <= abs(min_x): nose = 0
                else: nose = 2
            else:
                if abs(max_y) <= abs(min_y): nose = 1
                else: nose = 3
        else: nose = np.random.randint(4) # Square grid -> random orientation
        
        return nose
