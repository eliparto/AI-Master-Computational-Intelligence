""" Morphology Analysis """

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from database_components import (
    Base,
    Experiment,
    Generation,
    Genotype,
    Individual,
    Population,
)
from typing import overload, Union

from revolve2.modular_robot.body import Module
from revolve2.modular_robot.body.base import ActiveHinge, Brick, Core, Body
from revolve2.experimentation.evolution.abstract_elements import Morpho

class BodyCheck(Morpho):
    """
    Infer information about the robot's body, such as its 'nose', overall shape, etc.
    """
    def __init__(self) -> None:
        # Plotting variables
        self.colors = ["b", "r", "y"]
        self.legend = ["Brick", "Hinge", "Core"]
        self.marker = ["s", "^", "o"]
        self.sizes = [20, 20, 50]
        
        self.epsilon = 1e-9 # Prevent division errors w/ bboxes
        
    def findBodies(self, population: Population) -> list[Body]:
        """
        Returns all robot bodies in a population.
        """
        return [
            p.genotype.develop().body for p in population.individuals
            ]
        
    def findModules(self, body: Body) -> list[Module]:
        """
        Find all modules' coordinates for a single robot.
        """
        modules = body.find_modules_of_type(Brick) + \
            body.find_modules_of_type(ActiveHinge) + \
                body.find_modules_of_type(Core)
                
        coords = np.array([
            np.array(body.grid_position(m)) for m in modules
            ]).astype(int)
        
        return coords
    
    def findHingeOrientations(self, body: Body) -> list[int]:
        """
        Find the orientations of hinges in a robot (0/1 denote rotation by 0 or 90 degrees.).
        """
        hinges = body.find_modules_of_type(ActiveHinge)
        orientations = np.array([
            np.array(
                self.quaternion_to_euler(hinge.orientation)
                ) for hinge in hinges
            ])[:,-1]
        
        orientations = np.round(orientations/np.pi, 1)
        
        return orientations
        
    def findModulesSep(
            self, body: Body
            ) -> list[list[Brick], list[ActiveHinge], list[Core]]:
        """
        Find all modules' coordinates for a single robot per module type.
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
    
    def findBBox(self, body: Body, offset: bool = True) -> npt.NDArray[np.int_]:
        """
        Find a robot's bounding box.
        :param offset: Toggle to return the absolute of the bounding box,
        returning the offsets w.r.t. axis centers.
        """
        coords = self.findModules(body)
        xMax = np.max(coords[:,0])
        xMin = np.min(coords[:,0])
        yMax = np.max(coords[:,1])
        yMin = np.min(coords[:,1])
        zMax = np.max(coords[:,2])
        zMin = np.min(coords[:,2])
        
        bbox = np.array([
            [xMin, xMax], [yMin, yMax], [zMin, zMax]
            ])
        
        if offset: return np.abs(bbox)
        else: return bbox 
    
    def gridBody(self, body, coords) -> npt.NDArray[np.int_]:
        """
        Generate a bounding box for a robot body.
        :param modules: List of modules in a robot.
        """
        grid = np.zeros([40,40])
        for c in coords[:,:2]:
            c += np.array([20,20])
            grid[c[0],c[1]] += 1
        
        return grid
    
    def show2D(self, body: Body, show: bool = True, 
               nameOut: Union[str, None] = None, 
               figTitle: Union[str, None] = None) -> None:
        """
        Visualize a body from a 2D top-down perspective.
        """
        fig, ax = plt.subplots()
        ax.set_aspect("equal", adjustable="box")
    
        bricks, joints, _ = self.findModulesSep(body)
        jointAngles = self.findHingeOrientations(body)
        x_lim, y_lim, _ = self.findBBox(body, offset=False)
        x_lim[0] -= 1
        x_lim[1] += 1
        y_lim[0] -= 1
        y_lim[1] += 1
        
        for joint, angle in zip(joints, jointAngles):
            patch = self.joint_patch(center=(joint[0], joint[1]))
            ax.add_patch(patch)
            ax.add_patch(
                patches.Circle(
                    (joint[0], joint[1]), radius=0.15, color="silver"))
        for brick in bricks:
            patch = self.brick_patch(center=(brick[0], brick[1]))
            ax.add_patch(patch)
            ax.add_patch(
                patches.Rectangle(
                    (brick[0]-0.4, brick[1]-0.4), width=0.8, height=0.8, color='skyblue'))
        ax.add_patch(
            patches.Circle(
                (1.0, 0), radius=0.5, color="hotpink"))
 
        ax.set_aspect('equal')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.axis('off')
        
        if figTitle != None:
            plt.title(figTitle)
        
        if show: plt.show()
        else: 
            if nameOut == None: nameOut = "temp.png"
            else: nameOut = nameOut + ".png" 
            plt.savefig(nameOut)
    
    def joint_patch(self, center, width=0.5, height=0.5, thickness=0.05, color='silver'):
        x, y = center
        w = width/2
        h = height/2
        t = thickness / 1
    
        # Create a plus by combining horizontal and vertical rectangles
        verts = [
            (x - t, y + h), (x + t, y + h),
            (x + t, y + t), (x + w, y + t),
            (x + w, y - t), (x + t, y - t),
            (x + t, y - h), (x - t, y - h),
            (x - t, y - t), (x - w, y - t),
            (x - w, y + t), (x - t, y + t)
        ]

        return Polygon(verts, closed=True, color=color)
    
    def brick_patch(self, center, size=1.0, thickness=0.1, color='skyblue'):
        x, y = center
        s = size / 2
        t = thickness / 1
    
        # Create a plus by combining horizontal and vertical rectangles
        verts = [
            (x - t, y + s), (x + t, y + s),
            (x + t, y + t), (x + s, y + t),
            (x + s, y - t), (x + t, y - t),
            (x + t, y - s), (x - t, y - s),
            (x - t, y - t), (x - s, y - t),
            (x - s, y + t), (x - t, y + t)
        ]

        return Polygon(verts, closed=True, color=color)
    
    def quaternion_to_euler(self, q) -> tuple[float]:
        """
        Convert quaterion data into angles about roll, pitch, and yaw axes.
        """
        w, x, y, z = q
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(2*(w*y - z*x))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    
        return roll, pitch, yaw  # Angles in radians
    
    def plot2D(self, body, idx, plt_out=False, ax=None) -> None:
        """
        Generate 2D plot of a robot (looking down from the z-axis.)
        """
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.set_title(f"Body no. {idx}")
            
        modules = self.findModules(body)
        grid = np.flip(self.gridBody(body, modules), axis=0)
        im = ax.imshow(grid)
        ax.set_xlabel("Y")
        ax.set_ylabel("X")
        ax.set_xlim(10,30)
        ax.set_ylim(10,30)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(20, 19, "c", ha="center", va="center", c="red")
        cbar = ax.figure.colorbar(im, ax=ax, orientation="horizontal",
                                  shrink=0.8)
        cbar.set_label("No. of modules")
        cbar.set_ticks(np.arange(0,np.max(grid)+1,1))
        if plt_out: plt.show()
        
    def plot3D(self, body, idx, plt_out=False, ax=None) -> None:
        """
        Generate 3D plot of a robot
        """
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
        """
        Plot a robot's 2D and 3D representation side by side.
        """
        fig = plt.figure(constrained_layout=False)
        fig.tight_layout()
        subfigs = fig.subfigures(1,2)

        ax1 = subfigs[0].add_subplot()
        ax2 = subfigs[1].add_subplot(projection="3d")        
        self.plot2D(body, idx, plt_out=False, ax=ax1)
        self.plot3D(body, idx, plt_out=False, ax=ax2)
        fig.suptitle(f"Body no. {idx}\nNose orientation: {nose}")
        
    @overload
    def plotPop(self, population: Population) -> None: ...
        
    @overload
    def plotPop(self, population: list[Body]) -> None: ...
    
    def plotPop(self, population: Union[Population, list[Body]]) -> None:
        """
        Plot a population of robots.
        """
        if isinstance(population, Population):
            bodies = self.findBodies(population)
            noses = [p.nose for p in population.individuals]
            
        elif isinstance(population, list):
            bodies = population
            noses = self.findNose(bodies)
            
        assert -1 not in noses, "Nose orientations not correctly initialized. Call findNose(population)."
        for idx, body in enumerate(bodies):
            self.plotFigs(body, noses[idx], idx)
    
    @overload
    def findNose(self, population: Population) -> Population: ...
    
    @overload
    def findNose(self, population: list[Body]) -> list[int]: ...
    
    def findNose(
            self, population: Population) -> Union[Population, list[Body]]:
        """
        Find the `nose` (frontal orientation) of the robots. The nose is in the longest 
        x or y direction and the closest from the core (i.e. a salamander).
        This method ignores height in the z-direction.
        TODO: Implement self.findBBox()
        """
        if isinstance(population, Population):
            bodies = self.findBodies(population)
        elif isinstance(population, list):
            bodies = population
        
        noses = []
        for idx, body in enumerate(bodies):
            grid = self.findModules(body)[:,:2]
            min_x = np.min(grid[:,0])
            max_x = np.max(grid[:,0])
            min_y = np.min(grid[:,1])
            max_y = np.max(grid[:,1])
            width = max_x - min_x
            depth = max_y - min_y
            
            noses.append(self.noseLoc(
                min_x, max_x, min_y, max_y, width, depth))
            
        if isinstance(population, Population):
            for idx, p in enumerate(population.individuals):
                p.nose = noses[idx]
            return population
        elif isinstance(population, list): return noses
    
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
    
    def xyz_symmetry(self, body: Body) -> list[float]:
        """
        Calculate a body's symmetry around its x-, y-, and z-axes.
        TODO: Implement nose orientation
        """
        bboxes = self.findBBox(body)
        symmetries = [
            1 - (
                np.abs(bbox[0] - bbox[1])/(bbox[0] + bbox[1] + self.epsilon)
                ) for bbox in bboxes
            ]
        return symmetries
    
    def count_bricks_hinges(self, body: Body) -> list[int]:
        """
        Return a robot's no. of bricks and hinges.
        """
        bricks, hinges, _ = self.findModulesSep(body)
        return [len(bricks), len(hinges)]
    
    def calc_size_volume(self, body: Body) -> list[int]:
        """
        Calculate a robot's size and bounding box and displacement volume.
        """
        # Calculate bounding box volume
        # Extract axis lengths from bounding boxes
        lengths = np.sum(self.findBBox(body), axis=1)
        lengths = np.clip(lengths, 1, 100) # Prevent side lengths of 0
        vol_bbox = np.prod(lengths)
        # Calculate displacement volume (assume 1 unit of displacement for every part)
        vol_disp = len(self.findModules(body))
        
        return [vol_bbox, vol_disp]
    
    def findLimbs(self, body: Body) -> list[int, float]:
        """
        Find the no. of limbs and avg. limb length for a robot.
        """
        _, hinge_coords, _ = self.findModulesSep(body)
        
        coords_set = set(map(tuple, hinge_coords))
        visited = set()
        limb_lengths = []
    
        directions = np.array([
            [1, 0, 0],  # +x
            [0, 1, 0],  # +y
            [0, 0, 1],  # +z
        ])
    
        for coord in coords_set:
            coord_arr = np.array(coord)
    
            for dir_vec in directions:
                next_coord = tuple(coord_arr + dir_vec)
                prev_coord = tuple(coord_arr - dir_vec)
    
                if next_coord in coords_set and prev_coord not in coords_set:
                    current = coord_arr.copy()
                    limb = [tuple(current)]
    
                    while True:
                        next_candidate = tuple(current + dir_vec)
                        if next_candidate in coords_set:
                            limb.append(next_candidate)
                            current += dir_vec
                        else:
                            break
    
                    if not any(c in visited for c in limb):
                        visited.update(limb)
                        limb_lengths.append(len(limb) - 1)                  
        
        limb_count = len(limb_lengths)
        avg_limb_len = 0
        if len(limb_lengths) > 0: avg_limb_len = np.average(limb_lengths)
            
        return [limb_count, avg_limb_len]
                
            
            