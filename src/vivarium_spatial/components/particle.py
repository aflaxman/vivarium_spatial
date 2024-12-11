from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import KDTree  # Changed to scipy KDTree
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData


class Basic(Component):
    name = "basic"

    @property
    def columns_created(self):
        return ["x", "y", "theta"]

    CONFIGURATION_DEFAULTS = {
        "particle": {"basic": {"step_size": 0.01, "overall_max_angle_change": 30}}
    }

    def setup(self, builder):
        config = builder.configuration.particle.basic
        self.step_size = config.step_size
        self.overall_max_angle_change = config.overall_max_angle_change

        # Register the max angle change pipeline
        self.max_angle_change = builder.value.register_value_producer(
            "particle.max_angle_change",
            source=lambda index: pd.Series(self.overall_max_angle_change, index=index),
        )

        self.randomness = builder.randomness.get_stream("particle.basic")

    def on_initialize_simulants(self, simulant_data):
        """Start new simulants at random location in unit square,
        with random direction in degrees [0, 360)"""
        pop = pd.DataFrame(index=simulant_data.index)

        # Generate random x,y coordinates in [0,1) x [0,1)
        pop["x"] = self.randomness.get_draw(pop.index, additional_key="x")
        pop["y"] = self.randomness.get_draw(pop.index, additional_key="y")

        # Generate random angles in [0, 360) degrees
        pop["theta"] = self.randomness.get_draw(pop.index, additional_key="theta") * 360.0

        self.population_view.update(pop)

    def on_time_step(self, event):
        """Update particle positions by moving random amounts in their theta directions"""
        pop = self.population_view.get(event.index)

        # Convert angles to radians for calculation
        theta_rad = np.deg2rad(pop["theta"])

        # Calculate position updates using trigonometry
        dx = self.step_size * np.cos(theta_rad)
        dy = self.step_size * np.sin(theta_rad)

        # Get max angle change from pipeline, which may be modified by whimsy
        max_angle = self.max_angle_change(pop.index)
        dtheta = (
            (self.randomness.get_draw(pop.index, additional_key="angle") - 0.5)
            * 2
            * max_angle
        )

        # Update positions
        pop["x"] = (pop["x"] + dx) % 1.0  # Wrap around unit square using modulo
        pop["y"] = (pop["y"] + dy) % 1.0
        pop["theta"] = (pop["theta"] + dtheta) % 360.0

        self.population_view.update(pop)

        return pop


class Whimsy(Component):
    """Component that gives each particle a whimsy value and modifies their max angle change"""

    CONFIGURATION_DEFAULTS = {"particle": {"whimsy": {"alpha": 1, "beta": 1}}}

    @property
    def columns_created(self):
        return ["whimsy"]

    def setup(self, builder):
        self.config = builder.configuration.particle.whimsy
        self.randomness = builder.randomness.get_stream("particle.whimsy")

        builder.value.register_value_modifier(
            "particle.max_angle_change",
            modifier=self.modify_max_angle_change,
            requires_columns=["whimsy"],
        )

    def on_initialize_simulants(self, simulant_data):
        """Initialize whimsy values using Beta distribution"""
        draws = self.randomness.get_draw(simulant_data.index)
        whimsy = stats.beta.ppf(draws, self.config.alpha, self.config.beta)
        pop = pd.DataFrame({"whimsy": whimsy})
        self.population_view.update(pop)

    def modify_max_angle_change(self, index, angle):
        pop = self.population_view.get(index)
        return angle * pop.whimsy


class SpatialIndex(Component):
    """Simple spatial indexing component that maintains a KDTree of current positions."""
    
    name = "particle_spatial_index"
    
    @property
    def columns_required(self) -> List[str]:
        return ["x", "y"]
    
    def setup(self, builder: Builder) -> None:
        self._current_tree = None
        self._current_positions = None
    
    def on_time_step(self, event: Event) -> None:
        """Rebuild the KDTree with current positions"""
        pop = self.population_view.get(event.index)
        if len(pop) < 2:
            self._current_tree = None
            self._current_positions = None
            return
            
        self._current_positions = pop[["x", "y"]].values
        self._current_tree = KDTree(self._current_positions)
    
    def get_neighbor_pairs(self, radius: float) -> Optional[Set[Tuple[int, int]]]:
        """Get all pairs of particles within radius using efficient pair query."""
        if self._current_tree is None:
            return None
            
        return self._current_tree.query_pairs(radius)
    
    def query_radius(self, radius: float) -> Optional[List[np.ndarray]]:
        """Get neighbor indices for each particle within radius."""
        if self._current_tree is None:
            return None
            
        return self._current_tree.query_ball_point(self._current_positions, radius)


class Collisions(Component):
    """Component for handling particle collisions using shared spatial index"""

    @property
    def columns_created(self) -> List[str]:
        return ["collision_count", "last_collision_time"]

    @property
    def columns_required(self) -> List[str]:
        return ["x", "y", "theta"]

    CONFIGURATION_DEFAULTS = {"particle": {"collisions": {"critical_radius": 0.01}}}

    def setup(self, builder):
        config = builder.configuration.particle.collisions
        self.critical_radius = config.critical_radius
        self.randomness = builder.randomness.get_stream("particle.collisions")
        self.clock = builder.time.clock()
        self.collisions = 0
        self.current_collisions = []
        
        self.spatial_index = builder.components.get_component("particle_spatial_index")

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        self.population_view.update(
            pd.DataFrame(
                {"collision_count": 0, "last_collision_time": pd.NaT}, 
                index=pop_data.index
            )
        )

    def on_time_step(self, event: Event) -> None:
        """Check for collisions and update particle directions"""
        pop = self.population_view.get(event.index)
        if len(pop) < 2:
            return

        # Use more efficient pair query for collisions
        pairs = self.spatial_index.get_neighbor_pairs(self.critical_radius)
        if not pairs:
            self.current_collisions = []
            return
            
        # Convert pairs to unique particle indices
        collided_particles = np.unique(np.array(list(pairs)).ravel())
        
        self.collisions += len(collided_particles)

        # Store collision locations
        self.current_collisions = [
            (pop.iloc[i]["x"], pop.iloc[i]["y"]) for i in collided_particles
        ]

        # Generate new random angles for all collided particles
        new_angles = self.randomness.get_draw(
            pd.Index(collided_particles), 
            additional_key="collision_theta"
        ) * 360.0

        # Update collision stats
        updates = pd.DataFrame({
            "theta": new_angles,
            "collision_count": pop.loc[new_angles.index, "collision_count"] + 1,
            "last_collision_time": pd.Series(self.clock(), index=new_angles.index),
        })

        self.population_view.update(updates)

    def on_simulation_end(self, event: Event) -> None:
        print(f"collisions total: {self.collisions}")


class Flock(Component):
    """Component for implementing flocking behavior using shared spatial index"""

    @property
    def columns_required(self) -> List[str]:
        return ["x", "y", "theta"]

    def __init__(self):
        super().__init__()
        self.current_flocks = []

    CONFIGURATION_DEFAULTS = {
        "particle": {"flock": {"radius": 0.05, "alignment_strength": 0.91}}
    }

    def setup(self, builder):
        config = builder.configuration.particle.flock
        self.flock_radius = config.radius
        self.alignment_strength = config.alignment_strength
        self.randomness = builder.randomness.get_stream("particle.flocking")
        self.clock = builder.time.clock()
        self.flock_updates = 0
        
        self.spatial_index = builder.components.get_component("particle_spatial_index")

    def on_time_step(self, event: Event) -> None:
        """Update particle directions based on neighboring particles"""
        pop = self.population_view.get(event.index)
        if len(pop) < 2:
            return

        # Get neighbor lists for each particle
        neighbor_lists = self.spatial_index.query_radius(self.flock_radius)
        if neighbor_lists is None:
            return

        # Track particles that will be updated
        particles_to_update = []
        new_thetas = []

        # Calculate new directions based on neighbors
        for i, neighbors in enumerate(neighbor_lists):
            if len(neighbors) > 1:  # Only update if particle has neighbors
                # Calculate average direction of neighbors (excluding self)
                neighbors = [n for n in neighbors if n != i]
                neighbor_thetas = pop.iloc[neighbors]["theta"].values

                avg_theta = np.mean(neighbor_thetas)
                # Interpolate between current direction and neighbor average
                current_theta = pop.iloc[i]["theta"]
                new_theta = ((1 - self.alignment_strength) * current_theta + 
                           self.alignment_strength * avg_theta)

                particles_to_update.append(i)
                new_thetas.append(new_theta)

        if particles_to_update:
            updates = pd.DataFrame(
                {"theta": new_thetas},
                index=pop.index[particles_to_update],
            )
            self.population_view.update(updates)

    def on_simulation_end(self, event: Event) -> None:
        print(f"flock updates total: {self.flock_updates}")